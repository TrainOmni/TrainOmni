# Offline dense-logit KD contract

本文定义 TrainOmni native torch 的离线完整词表 logit KD。该 objective 只消费经过完整身份预检的 BF16 raw-logit cache；它不会构建、加载或接受 live teacher，也不会回退 delegated backend。

## Objective and recipe

Stage 仍消费 canonical `sft` samples，loss implementation 显式选择 `offline-dense-logit-kd`：

```yaml
stage:
  stage_id: offline_dense_logit_kd
  stage_type: reasoning_distillation
  objective: sft
  objective_impl: offline-dense-logit-kd
  component_policy:
    vision_encoder: {trainable: false, dtype: bf16}
    connector:
      trainable: true
      learning_rate: 3.0e-4
      weight_decay: 0.0
      dtype: fp32
      gradient_clip: 1.0
    language_model: {trainable: false, dtype: bf16}
  optimization:
    optimizer: adamw
    optimizer_config:
      implementation: torch
      foreach: false
      kwargs: {}
    learning_rate: 3.0e-4
    weight_decay: 0.0
    max_steps: 16
    gradient_accumulation_steps: 1
    diagnostics:
      record_gpu_memory: true
      component_grad_norms: true
      component_update_probes: true
      require_finite_nonzero_gradients: true
      require_parameter_updates: true
      expected_trainable_numel: <connector-numel>
      required_components: [connector]
      max_reserved_bytes: 12884901888
    config: {scheduler: constant}
  engine:
    backend: torch
    parallelism: single
    precision: bf16
    attention_backend: eager
    config: {device: cuda}
  checkpoint: {every_steps: 8, resume_level: exact, export_formats: [torch]}
  inputs:
    model: artifact://qwen35_minicpm5_intergps_minimal_sft/step-00000064
```

Objective identity 使用现有 `RunSpec.metadata` 中的保留 namespace，因此全部字段天然进入 run fingerprint：

```yaml
metadata:
  offline_dense_logit_kd:
    cache_manifest: D:/path/to/cache/manifest.json
    cache_manifest_sha256: <sha256>
    cache_content_sha256: <sha256>
    teacher_state_sha256: b0d275f95163b3cda5f9b62251ce0d7dbf67dff54cdb64c7790461a614c038f5
    teacher_manifest_sha256: 58ecb70dfa729c038e0bfaf08fbff4f2c422c24fb25c25e48351b7a01c2c8664
    teacher_run_fingerprint: 70b11e5587983900834862a1041e099e585278b2698a02011bc03d4fc4ec0f03
    student_state_sha256: a35cf0528f25d977c6c0011cd6a450500a039682661f5700422eaf72b84fbcb2
    student_manifest_sha256: f6339790e2790393d25ee31e20078c4cee5190d230b703b37b07dd7653d268b1
    student_run_fingerprint: 3e0fb6d81d51fb4123382e605231f5a273ac1a201d79ed13edaca835f5858a74
    model_identity_sha256: <sha256>
    tokenizer_sha256: <sha256>
    processor_sha256: <sha256>
    data_sha256: <sha256>
    loss_positions_sha256: <sha256>
    temperature: 2.0
    ce_weight: 0.5
    kd_weight: 0.5
    vocab_size: 130560
    cache_dtype: bfloat16
    max_cache_bytes: <declared-cache-capacity-bytes>
```

改变 manifest/cache、teacher、student、model、tokenizer、processor、data、loss positions、temperature 或 loss weights 中任一字段都会改变 run fingerprint。Setup 会把 recipe 的 expected identity 与实际 manifest、资产和 checkpoint 逐项比较。

## Cache manifest

Manifest schema 为 `trainomni.offline-dense-logit-cache.v1`。Producer 可直接使用 `trainomni.objectives` 导出的 Pydantic types 和 digest helpers：

- `OfflineDenseLogitCacheManifest`；
- `CheckpointIdentity`、`ModelIdentity`、`AssetSetIdentity`；
- `CacheSampleIdentity`、`LogitTensorIdentity`；
- `asset_set_digest()`、`data_identity_digest()`、`loss_position_digest()`、`integer_tensor_digest()`。

Manifest 必须包含：

- teacher/student artifact、物理目录、state/manifest SHA-256、producer run fingerprint 和 `model_only` 语义；
- consumer model plugin/version 与基座 asset identity；
- tokenizer/processor 文件集合、大小和 SHA-256；
- 数据源路径/SHA-256、split fingerprint、reader/importer identity 和逐样本 canonical identity；
- 每条样本的 `input_ids`/`labels` digest、assistant target positions、target token IDs；
- 每条 BF16 logit tensor 的相对文件、`[positions, vocab]` shape 和 SHA-256；
- loss-position identity、总 position 数、总字节数和 cache content SHA-256。

每条样本使用独立 headerless BF16 raw 文件，且必须位于 manifest 目录以内。`content_sha256` 是按 manifest `samples` 顺序串联 raw tensor bytes 后的 SHA-256；每 tensor digest、shape 和 manifest digest同时消除边界歧义。当前合同面向同一 pinned little-endian 软件/硬件栈。

Setup 在 optimizer step 1 前流式验证：manifest、teacher/student checkpoint、实际 student input artifact、model/tokenizer/processor assets、数据文件、全部 logit tensor 的存在性/大小/digest、总内容 digest 及 recipe 显式声明的 `max_cache_bytes` 上限。任何异常都会中止，不会重建 cache 或启动 teacher；该上限没有隐式默认值，VLMTrainer 必须按真实 full-vocab cache 规模显式填写。

## Batch and position semantics

Model plugin 不需要读取 cache，也不得写入保留的 `kd_*` fields。Objective 在 `prepare()` 中按 `sample_id` 加载并验证 cache，然后产生：

```text
kd_teacher_logits       BF16 [1, N, V]
kd_assistant_positions  int64 [1, N]
kd_position_mask        bool  [1, N]
kd_target_token_ids     int64 [1, N]
kd_cache_identity       cache content SHA-256
```

V1 强制 batch size 1。`assistant_positions` 指模型展开后的 target-label positions；student prediction logits 取 `position - 1`。这适配 image tokens 插入后的复合 VLM 序列。Framework 会从真实 batch `labels` 重新推导 `tuple(nonzero(labels != -100))`，并要求它与 manifest `assistant_positions` 严格相等；随后按这些真实 positions 读取 `labels[p]`，要求与 `target_token_ids` 严格相等。该验证在 student model forward 和 optimizer step 1 之前执行，同时还会验证原始 batch `input_ids`/`labels` digest。Positions 自身也必须严格递增且大于零。

Model forward 接收原始 student inputs 和 labels，不接收 `kd_*` fields。冻结 language model 参数不会包裹 `no_grad()`；student logits 到 connector output 的 autograd 路径必须保持。Model output 必须提供 `[1, sequence, vocab]` logits。

## FP32 loss

Objective 在关闭 autocast 的区间把 student/teacher selected logits 转为 FP32，并严格计算：

```text
token_ce = mean CE(student_logits, target_ids)
teacher_kl = T^2 * mean KL(teacher || student)
weighted_ce = ce_weight * token_ce
weighted_teacher_kl = kd_weight * teacher_kl
total = weighted_ce + weighted_teacher_kl
```

`softmax`、`log_softmax`、KL accumulation 和 reduction 均为 FP32。`LossOutput.terms`、step metrics、最终 run result 和 checkpoint `metadata.objective_evidence` 保留 `token_ce`、`teacher_kl`、两个 weighted term 与 `total`；`loss_tokens=N` 驱动累计 token/cursor state。

Teacher logits必须保持 BF16、完整 vocab、无梯度。Teacher/student vocab、position/mask/target shapes 或 dtype 不一致，position 越界，target 越界，非有限 loss，以及 models mapping 中出现 live `teacher` 都会 fail closed。

## Checkpoint and exact resume

每个 checkpoint 的 `metadata.objective` 保存：

- manifest path/hash、cache ID/content digest、producer revision；
- teacher/student/model/tokenizer/processor/data/loss-position/logit identities；
- temperature、weights、KL direction、FP32/reduction semantics；
- `exact_resume: immutable_external_identity`。

Resume 会先重新预检所有外部文件，再加载训练 state，最后把 checkpoint objective identity 与当前 identity 做精确比较。Cache 缺失、内容改变、路径/身份改变或 checkpoint metadata 被篡改都会失败。Cache logits 不写入 checkpoint；student model、optimizer/scheduler、RNG、data cursor、step/microstep 和累计 loss tokens仍由现有 exact state registry恢复。

VLMTrainer 使用 `StageRunRequest.input_artifacts["model"]` 传入带物理 `uri` 的 student `ArtifactRef`，并设置 `trusted_input_artifacts=True`。Framework 会同时核对 artifact ref、物理路径、state/manifest digest；不能只在 YAML 中写逻辑 artifact 后绕过物理 lineage。

## Current boundary

该实现只声明 native torch offline full-vocab cached-logit KD 机制。它不包含 cache producer、live teacher、top-k/量化 cache、hidden-state loss、在线 KD、质量门、DPO 或大规模 distributed KD。Cache 生产、step-0/final evaluator、固定 16-step recipe 和真实 GPU acceptance 由 VLMTrainer 完成。
