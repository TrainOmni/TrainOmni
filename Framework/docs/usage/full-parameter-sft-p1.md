# Full-parameter SFT P1 contract

本文定义真实 composite VLM“真全参数 SFT”的 Framework 接入边界。它只覆盖 P1；offline KD 和 multimodal DPO 不在此 contract 中。

## Recipe

目标 recipe 应显式配置 optimizer、诊断门禁和每个需要 checkpointing 的 component：

```yaml
stage:
  component_policy:
    vision_encoder:
      trainable: true
      dtype: bf16
      activation_checkpointing:
        use_reentrant: false
        config: {}
    connector:
      trainable: true
      dtype: bf16
    language_model:
      trainable: true
      dtype: bf16
      activation_checkpointing:
        use_reentrant: false
        config: {}

  optimization:
    optimizer: adamw
    optimizer_config:
      implementation: torch
      foreach: false
      kwargs: {betas: [0.9, 0.999], eps: 1.0e-8}
      quantization: null
    learning_rate: 1.0e-5
    weight_decay: 0.0
    max_steps: 8
    gradient_accumulation_steps: 1
    diagnostics:
      record_gpu_memory: true
      component_grad_norms: true
      component_update_probes: true
      update_probe_chunk_elements: 1048576
      require_finite_nonzero_gradients: true
      require_parameter_updates: true
      expected_trainable_numel: 1182802176
      required_components: [vision_encoder, connector, language_model]
      max_reserved_bytes: 16106127360
    config: {scheduler: constant}

  engine:
    backend: torch
    parallelism: single
    precision: bf16
    config: {device: cuda}
```

`expected_trainable_numel` 统计进入同一个 optimizer 的参数，而不是只统计 `requires_grad`。任一 required component 缺失、梯度非有限/为零或 optimizer step 后没有任何实际 bitwise 参数变化，run 会显式失败。

`max_reserved_bytes` 是硬安全门槛。建议为驱动、桌面和临时 kernel 留余量，而不是直接等于显卡标称字节数；上面的 15 GiB 仅展示字段形状，正式值仍应由目标机器 smoke 决定。

## Optimizer identity and exact resume

Native path 使用 `torch.optim.AdamW`。`foreach: false` 会直接传给构造器并同时写入 configured/actual metadata。Framework 不维护隐式 FP32 master weights；BF16 参数的 Adam state dtype 由 PyTorch 实际实现决定并在第一次 optimizer step 后审计。

以下信息会同时进入 `run-manifest.json.metadata` 和每个 checkpoint 的 `manifest.json.metadata`：

- optimizer name、implementation、Python class；
- package name/version；
- configured kwargs 和 optimizer actual defaults；
- state tensor dtype、tensor 数量和 numel；
- quantization configuration；
- `exact_resume: full_state_dict`；
- 每组件及总 `trainable_numel`。

Exact resume 先恢复完整 optimizer state，再比较 checkpoint 记录的 optimizer identity/state-dtype contract。旧 checkpoint 缺少该 contract，或者 class/version/config/state dtype 不一致时，恢复会失败，不会继续训练。

## AdamW8bit

可选 CUDA path：

```yaml
optimization:
  optimizer: adamw
  optimizer_config:
    implementation: bitsandbytes
    foreach: null
    kwargs: {betas: [0.9, 0.999], eps: 1.0e-8}
    quantization:
      bits: 8
      min_8bit_size: 4096
      percentile_clipping: 100
      block_wise: true
      paged: false
```

安装：

```powershell
python -m pip install -e ".[torch,bitsandbytes]"
```

该路径只构造 `bitsandbytes.optim.AdamW8bit`（或显式 `paged: true` 时的 PagedAdamW8bit）。Package 缺失、device 不是 CUDA或构造失败都会中止；绝不会降级为 torch AdamW。Bitsandbytes 官方当前提供 Windows x86-64 CUDA wheels，并说明 8-bit optimizer 需要 Pascal 或更新的 NVIDIA GPU；但 TrainOmni 的本地 CPU CI 只能验证选择与失败 contract。真实 P1 使用该路径前，必须在目标 Windows/CUDA 环境完成 uninterrupted/resume equality，才能声明 AdamW8bit exact conformance。

官方边界：

- <https://huggingface.co/docs/bitsandbytes/en/installation>
- <https://huggingface.co/docs/bitsandbytes/optimizers>

若 native AdamW 和 AdamW8bit 都无法满足单卡内存，使用 delegated backend 承担 ZeRO/CPU-offload optimizer；不要把 delegated state 冒充 native exact checkpoint。

## Composite activation checkpointing hook

Core 不再调用 composite 顶层的 `gradient_checkpointing_enable()`。当 recipe 为 component 配置 activation checkpointing 时，model plugin 必须实现：

```python
def configure_activation_checkpointing(self, bundle, requests):
    receipts = {}
    for component_id, request in requests.items():
        module = component_modules[component_id]
        enable_component_checkpointing(
            module,
            use_reentrant=request.use_reentrant,
            **dict(request.config),
        )
        receipts[component_id] = ActivationCheckpointingReceipt(
            component_id=component_id,
            implementation="plugin-specific-real-hook",
            use_reentrant=request.use_reentrant,
            metadata={"module": type(module).__qualname__},
        )
    return receipts
```

Vision 与 LLM 是两个独立 request。Plugin 必须在真实子模块/block 上启用 checkpointing 后才返回 receipt。Core 要求 receipt 精确覆盖 request，并验证 `enabled` 与 `use_reentrant`；缺 hook、漏 component、多余 component、错误 receipt 或悄悄改为 reentrant 都会在 optimizer 创建前失败。

## Evidence

每个 optimizer step 的 `metrics.jsonl` 包含扁平、便于监控的数值：

```text
trainable_numel
components/vision_encoder/grad_norm
components/vision_encoder/changed_elements
components/vision_encoder/changed_tensors
components/vision_encoder/abs_update
components/vision_encoder/abs_update_l1
gpu_memory/max_allocated_bytes
gpu_memory/max_reserved_bytes
```

Run/checkpoint metadata 同时保存结构化 `training_evidence`：每组件 trainable numel、finite grad norm、梯度 tensor 数、完整扫描规则、确定性 representative before/after、首个实际变化元素、最大绝对变化元素、精确 changed element/tensor count、最大/L1 absolute update，以及 CUDA current/peak allocated/reserved 字节。

Update gate 不再抽样单个元素。`optimizer.step()` 前，它按参数名和 flat index 的固定顺序，把每个 component 中 optimizer 持有的全部参数以原始 dtype snapshot 到 CPU；step 后按 `update_probe_chunk_elements` 分块复制当前值并进行逐元素 raw-byte 比较。因此一个 representative BF16 元素未变不会造成假阴性，只要组件中任意元素的实际存储位发生变化，`changed_elements` 就大于零。该比较没有摘要碰撞语义；`changed_elements` 是精确 bitwise count，`abs_update` 是最大绝对值变化，`abs_update_l1` 是 FP64 聚合的 L1 变化。

该强门禁以主机资源换确定性：单卡 1,182,802,176 个 BF16 参数的 before snapshot 约占 2.20 GiB 主机内存；step 后的比较临时量由 chunk 字段限制，不复制第二份完整模型，也不增加持久 GPU 参数副本。多进程时 metadata 同时记录 local count 和跨 rank 汇总 count；DDP 汇总会计算各 replica 的 optimizer-held 元素，FSDP2 则汇总 local shard。P1 当前单卡验收的汇总值就是模型本身的精确 count。

`require_parameter_updates=true` 要求每个 required component 同时满足跨 rank `changed_elements > 0` 与 `max_abs_update > 0`。这排除了仅发生正负零 raw-bit 翻转却没有数值变化的边界情况。组件完全不变、扫描期间 shape/dtype 改变或出现非有限 parameter delta 都会中止；它不会退化为只看 gradient，也不会通过调高 LR 改变判断。外层 evidence schema 已升级为 `trainomni.training-evidence.v2`，旧的 `v1` 单元素语义不应与新结果混用。

## P1 handoff checklist

VLMTrainer 需要：

1. 在 plugin 将 `vision_encoder` 和 `language_model` request 路由到真实子模块，并返回 typed receipts；
2. 将三个 component 设为 trainable，并配置准确的 `expected_trainable_numel`；
3. 使用固定 8 行、8 steps、step 4/8 checkpoint recipe；
4. 比较 uninterrupted step 8 与 step 4 resumed step 8 的完整 state；
5. 检查 `training_evidence`、CUDA peak、reload forward 和 generation；
6. 只有上述全部通过才记录 P1 complete。
