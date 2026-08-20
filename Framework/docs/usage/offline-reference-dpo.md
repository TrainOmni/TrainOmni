# Offline-reference DPO contract

本文定义 TrainOmni native torch 的离线 reference-log-prob 原始 sigmoid DPO。训练进程只构建一个 policy，执行 chosen/rejected 两次 forward；reference 仅在独立 producer 阶段运行，训练时只消费经过完整身份预检的逐 token FP32 log-prob cache。Objective 不加载、不接受 live reference，也不会回退 delegated backend。

## Objective and recipe

Stage 消费 canonical `preference` samples，并显式选择 `offline-reference-dpo`：

```yaml
stage:
  stage_id: offline_reference_dpo
  stage_type: offline_preference
  objective: preference
  objective_impl: offline-reference-dpo
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
    optimizer_config: {implementation: torch, foreach: false, kwargs: {}}
    learning_rate: 3.0e-4
    weight_decay: 0.0
    max_steps: 16
    gradient_accumulation_steps: 1
    config: {scheduler: constant}
  engine:
    backend: torch
    parallelism: single
    precision: bf16
    attention_backend: eager
    config: {device: cuda}
  checkpoint: {every_steps: 8, resume_level: exact, export_formats: [torch]}
  inputs:
    model: artifact://qwen35_minicpm5_offline_dense_logit_kd_gate/step-00000016
```

全部 expected identity 和算法常量使用现有 `RunSpec.metadata` 保留 namespace，因此天然进入 run fingerprint：

```yaml
metadata:
  offline_reference_dpo:
    cache_manifest: D:/path/to/reference-cache/manifest.json
    cache_manifest_sha256: <sha256>
    cache_content_sha256: <sha256>
    preference_manifest_sha256: <sha256>
    preference_identity_sha256: <sha256>
    pair_identity_sha256: <sha256>
    data_identity_sha256: <sha256>
    reference_state_sha256: <sha256>
    reference_manifest_sha256: <sha256>
    reference_run_fingerprint: <sha256>
    policy_state_sha256: <sha256>
    policy_manifest_sha256: <sha256>
    policy_run_fingerprint: <sha256>
    model_identity_sha256: <sha256>
    tokenizer_sha256: <sha256>
    processor_sha256: <sha256>
    beta: 0.1
    loss_variant: sigmoid
    label_smoothing: 0.0
    sequence_reduction: sum
    pair_reduction: mean
    compute_dtype: float32
    reference_free: false
    auxiliary_ce: false
    expected_pair_count: 12
    expected_total_positions: 84
    vocab_size: 130560
    max_cache_bytes: 16777216
```

V1 将算法限制编码为 typed literals：beta 只能是 `0.1`，loss 只能是原始 sigmoid DPO，label smoothing 为零，sequence log-prob 使用 sum，pair 使用 mean，禁止 reference-free 和 auxiliary CE。改变任何 identity 或算法字段都会改变 run fingerprint；与 cache manifest 不一致时在 policy forward 前失败。

## Preference and reference-cache schemas

Preference manifest schema 为 `trainomni.offline-dpo-preference.v1`。它记录 source data/split identity、确定性 producer revision，以及有序 pair 的 source index、common prompt/media、chosen/rejected canonical identity、construction rule、judge、score 和 margin。每个 pair 有独立 canonical digest，整个 preference manifest 另有 aggregate identity 和文件 SHA-256。

Reference cache schema 为 `trainomni.offline-reference-dpo-cache.v1`，绑定：

- 唯一 policy/reference model-only checkpoint 的 artifact、物理路径、state/manifest SHA-256 和 producer run fingerprint；
- model plugin/version、model/tokenizer/processor assets；
- preference manifest、source data、split 和有序 pair identity；
- 每个 chosen/rejected branch 的 canonical identity、expanded `input_ids`/`labels` digest；
- `assistant_positions`、`causal_positions=p-1`、`target_token_ids=labels[p]`；
- common expanded media/model-input digest；
- 每个 branch 的 headerless little-endian FP32 per-token reference log-prob tensor、sequence sum、shape、size、SHA-256；
- aggregate pair count、position count、bytes 和 cache content SHA-256；
- 完整 DPO algorithm identity。

Raw content digest 顺序固定为 manifest pair order 中的 `chosen` tensor 后接 `rejected` tensor。Framework 会重新读取每个 FP32 tensor，验证 finite values 和 FP32 sequence sum。Reference/policy、preference、pair、data、model/plugin、tokenizer、processor 或 cache 任一 mismatch 均 fail closed。

Producer 可使用 `trainomni.objectives` 导出的 types/helpers：

- `OfflineDPOPreferenceManifest`、`PreferencePairIdentity`、`DPODataIdentity`；
- `OfflineReferenceDPOCacheManifest`、`CachedDPOPairIdentity`、`DPOBranchIdentity`；
- `ReferenceLogProbTensorIdentity`、`DPOCacheIdentity`、`DPOAlgorithmIdentity`；
- `dpo_data_identity_digest()`、`preference_pair_digest()`、`preference_manifest_digest()`、`model_inputs_digest()`、`integer_tensor_digest()`。

## Model-plugin paired batch

V1 强制 batch size 1。Model plugin 的 `ModelBatch.model_inputs` 必须且只能包含：

```text
chosen:
  input_ids, labels, attention_mask, pixel_values/image_grid/... model kwargs
rejected:
  input_ids, labels, attention_mask, pixel_values/image_grid/... model kwargs
dpo_pair_identity:
  sample_id
  preference_pair_sha256
  canonical_pair_sha256
  common_prompt_sha256
  media_sha256
  common_model_inputs_sha256
  chosen_canonical_sha256
  rejected_canonical_sha256
```

Objective 在 CPU prepare 阶段、model forward 前执行：

1. 验证 batch pair identity 与 preference/cache 的 chosen/rejected 方向；
2. 对两分支分别验证 `input_ids`/`labels` digest；
3. 从真实 labels 推导 `tuple(nonzero(labels != -100))`，与各自 manifest positions 严格等值，再验证 `labels[p] == target_token_ids`；
4. V1 要求 chosen/rejected target positions 和 target token 数相等；
5. 要求所有非 target positions 的 `input_ids` 相等；
6. 要求除 `input_ids`/`labels` 外的 common model inputs（包括 media tensors）结构和 tensor 值完全相等，并与 `model_inputs_digest` 相符；
7. 重新读取并哈希两分支 reference tensor。

这同时阻止 prompt/media 分叉、pair swap、截断、loss-mask 错位以及插件把不同图像送入两分支。Model plugin 不读取 reference cache；cache 注入由 objective 完成。

## FP32 sigmoid DPO

Policy chosen/rejected 分别 forward，冻结 language/vision 参数时不得使用包围 forward 的 `no_grad()`，两个分支都必须保留到 connector 的 autograd 路径。对每个 branch：

```text
logp_pi = sum_p log_softmax(policy_logits[p-1].float())[labels[p]]
logp_ref = sum_p cached_reference_logp[p]

rho_pi = logp_pi_chosen - logp_pi_rejected
rho_ref = logp_ref_chosen - logp_ref_rejected
delta = rho_pi - rho_ref
dpo_logit = 0.1 * delta
loss = softplus(-dpo_logit)
```

Log-softmax、gather、sequence sum、ratio、delta、reward 和 softplus 全部使用 FP32。Objective 输出并持久化：

- policy/reference chosen/rejected 四个 sequence log-prob；
- policy/reference log-ratio、delta、dpo_logit；
- chosen/rejected reward、reward margin、preference accuracy；
- loss、chosen/rejected target-token count、pair count。

`LossOutput.counts` 还提供 `preference_pairs`、`chosen_loss_tokens`、`rejected_loss_tokens` 和总 `loss_tokens`。

## Checkpoint and exact resume

Objective setup 在 optimizer step 1 前完成全部 external preflight。每个 checkpoint 保存：

- 完整 immutable objective identity 和 algorithm/cache/preference lineage；
- 最新全部 DPO metrics (`metadata.objective_evidence`)；
- 累计 pair/chosen/rejected token counters (`metadata.objective_counts`)；
- 三个 counter 对应的实际 StateRegistry objects，参与 strict exact resume；
- 原有 policy model、optimizer/scheduler、RNG、data cursor、step/microstep 和总 loss tokens。

Resume 先重新预检 preference/cache/checkpoint/assets，再验证 run fingerprint 和 checkpoint objective identity，最后严格加载所有 registry objects。Cache 缺失、篡改、pair direction 改变或 objective metadata 不一致不会自动重建或降级。

VLMTrainer 必须通过 `StageRunRequest.input_artifacts["model"]` 提供带物理 `uri` 的 policy `ArtifactRef`，并设置 `trusted_input_artifacts=True`。仅在 YAML 中写逻辑 artifact 不能绕过物理 lineage。

## VLMTrainer minimum integration

1. 双遍生成 deterministic canonical preference manifest/JSONL，完成 ground-truth cyclic-negative 语义检查；Framework 负责消费侧 identity 与 pair-direction enforcement，不替代数据 producer 的 A/B/C/D 业务校验。
2. 独立加载 reference checkpoint，逐 branch 缓存真实 positions 上的 FP32 per-token log-prob，并用上述 exported schemas 原子发布 cache。
3. Preference model plugin 产生本文精确定义的 paired batch 和 `dpo_pair_identity`；cache producer 与 policy consumer 必须使用相同 expanded inputs。
4. Recipe 填写全部 `metadata.offline_reference_dpo` 字段，并以 physical policy artifact 启动 native torch stage。
5. 配置 connector-only、16 steps、checkpoint step 8、step-8→16 exact resume；真实 gate 另行执行 baseline/final evaluation、freeze、resource、reload 和 generation acceptance。

## Claim boundary

Framework 只声明 native torch offline-reference original sigmoid DPO 的 cache consumer、数值、identity、failure、evidence 和 exact-resume 机制。它不包含正式 InterGPS preference/cache producer，不执行真实 12-pair/84-token cache，不启动目标 16-step GPU 训练，也不声明质量、泛化或人类偏好提升。Live-reference DPO、reference-free DPO、ORPO/SimPO/KTO、label smoothing、auxiliary CE、LoRA/full-parameter DPO 和大规模 distributed DPO 均在本 V1 边界之外。
