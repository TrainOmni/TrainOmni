# TrainOmni Framework v1 Implementation

- Date: 2026-08-20
- Status: implemented and verified
- Scope: VLM-first TrainOmni framework; no target-model-specific branch; audio understanding next; generation and Ascend deferred

## 1. Foundation decision

TrainOmni 采用 selective self-build，而不是 fork 一个大一统微调产品，也不是重写分布式/RL 基础设施：

- Own：canonical semantics、plugins、recipes、pipeline、lineage、exact resume、CLI。
- Reuse：PyTorch、Transformers、PEFT、FSDP2、DCP。
- Delegate：VeOmni/TRL/NeMo/veRL/AReaL/lmms-eval 等专用 backend。

原因和候选框架比较见 `docs/research/open-source-foundation-decision-2026-08.md`。

## 2. Implemented module map

```text
src/trainomni/
  checkpoint/   local atomic state, DCP, model-only reshard
  config/       strict schema, resolver, fingerprint
  contracts/    issues, artifacts, batch cost/plan
  data/         canonical, readers, importers, mixture, batches, rank grouping
  engines/      protocol, torch loop, PEFT, delegated command adapters
  evaluation/   normalized loss and external command evaluator
  models/       plugin protocol, bundle/build context, conformance, probe
  objectives/   masked LM and delegated algorithm requirements
  recipes/      Pipeline DAG, gates, artifact catalog
  registry/     explicit model/data plugin loading
  runtime/      stage/pipeline/eval/export/provenance/logging/seed
  cli.py        public CLI
```

## 3. Data path

```text
physical URI
  -> StatefulRecordReader
  -> SampleImporter
  -> CanonicalSample + ImportTrace
  -> deterministic weighted MixtureStream
  -> ModelPlugin.encode
  -> EncodedSample + SourceSpan + CostVector
  -> budget-aware StatefulBatchStream
  -> optional DistributedBatchStream
  -> ModelPlugin.collate
  -> ModelBatch
```

Checkpoint state contains every cursor/RNG/buffer needed to reproduce the next batch. JSONL uses byte offset rather than holding a file descriptor or rescanning from the beginning.

## 4. Objective and engine separation

`stage.objective` is the sample semantic contract (`cpt/sft/preference/prompt_only`)；`stage.objective_impl` is the loss/algorithm (`masked-causal-lm/dpo/distillation/grpo/ppo`)。这避免 model sample validation 与某个 Trainer 算法耦合。

Native torch engine owns ordinary loop semantics. Rollout、actor/reference/reward、多服务权重同步不是 batch loss，因此作为 delegated Stage 执行。Delegated command始终 `shell=False`，需要 `allow_external_command: true`，通过 result JSON 返回 metrics/artifacts。

VeOmni 使用专用 delegated bridge，而不是与 generic command 共享一个空 identity。Recipe 必须给出 immutable `backend_revision` 和精确 `trainomni.veomni-bridge.v1` API；请求携带版本化 backend contract。当前 capability 只覆盖 VLM supervised/preference 范围，并主动拒绝 `resume_level: exact`，直到真实 VeOmni data/RNG/topology conformance 完成。

## 5. Native torch engine

Preparation:

1. Seed Python/NumPy/Torch before model construction.
2. Build `ModelBundle` through the model plugin.
3. Exact-cover parameters with `ComponentCatalog`.
4. Apply component trainability/dtype/activation checkpointing.
5. Apply optional LoRA/QLoRA.
6. Move/wrap single, DDP or FSDP2.
7. Build component-aware optimizer and scheduler.
8. Register runtime state and optional resume.

Loop:

- autocast/TF32 and optional scaler；
- gradient accumulation with `no_sync`；
- named objective terms and normalized denominators；
- component gradient clipping；
- step/token budgets；
- checkpoint only at optimizer boundaries；
- JSONL metrics callbacks。

`torch.compile` is an engine option applied after parallel wrapping. Backend/model compatibility remains capability-tested rather than silently assumed.

## 6. Distributed data correctness

Variable-size VLM samples make “each rank independently packs its shard”容易产生不同 microbatch 数和 collective deadlock。实现先在每个 rank deterministically 生成相同 global batch sequence，再按 `world_size` 组成 batch group，每 rank 取一个。所有 rank 消费相同 global reader/mixture progress，因此 optimizer microstep 数一致，checkpoint 后也能精确恢复。

这不是最高吞吐的最终 sampler，但它是清晰、可验证的 correctness baseline；规模 backend 可替换数据 adapter，同时必须满足相同 state contract。

## 7. Checkpoint design

### Single/DDP

- sibling incomplete directory；
- state pickle fsync；
- SHA-256 manifest；
- remove `INCOMPLETE` then atomic rename；
- DDP per-rank checkpoint roots + barrier；
- explicit trusted load。

### FSDP2

- PyTorch DCP saves canonical model/optimizer FQNs and DTensor shards；
- duplicate `model_only` DCP entry enables topology-independent load/export；
- rank-local pickle sidecars preserve data/RNG/scheduler/scaler state；
- manifest verifies sidecar size/hash；
- exact runtime resume requires same world size；
- model-only load does not deserialize runtime pickle。

## 8. Pipeline and artifacts

Pipeline validates uniqueness, edge references and cycles before execution. Runtime performs deterministic topological scheduling, persists status atomically, retries stale `running/failed` stages on explicit resume, resolves artifact ID/selector/URI across edges, registers lineage, and applies gates only after result collection.

Executor reuse会从 durable state 重建 in-memory catalog，因此成功或失败后的同实例 resume 不会重复注册 artifact。自动两阶段 torch test 验证第一阶段的物理 checkpoint URI 被第二阶段实际加载，并在 resume 后保留 lineage 和 resume level。

`ArtifactRef` separates logical identity from physical URI. This lets a subsequent stage receive a local checkpoint path today and an object-store/DCP location through a future catalog without changing StageSpec。

## 9. Evaluation and export

- Loss evaluator aggregates named terms using each term denominator, not a misleading mean-of-means。
- External evaluator is shell-free and requires explicit authorization。
- Local exact checkpoint export requires trust; core restores only the model state with `strict=False` registry matching。
- DCP `model_only` can load into an unsharded model in one process, then model plugin emits HF/safetensors or another deployment format。

## 10. Security boundaries

- YAML/JSON never auto-imports Python。
- `--plugin` and `--data-plugin` are explicit trust decisions。
- local exact pickle load requires `--trusted-resume`/`--trusted-checkpoint`。
- delegated engine/evaluator commands use argv arrays and `shell=False`。
- delegated request manifests redact environment/token/password/secret/api-key fields。
- source fingerprints and checkpoint file hashes detect ordinary mutation/corruption；trust flags address code-execution risk, not just integrity。

## 11. What adding the target model means

Framework code is not expected to change. A target plugin supplies:

1. model/processor construction or composite assembly；
2. component prefix catalog；
3. canonical formatter and loss mask；
4. collator and cost model；
5. capability declaration；
6. state-dict/export mapping；
7. conformance fixtures。

Training strategy lives in recipe/pipeline files, not inside the plugin. Data schema conversion lives in a data importer, not in the model code。

## 12. Deferred scope

TrainOmni v1 采用 VLM-first 路线。音频已经有 canonical representation，但尚不声明 audio encoder/processor conformance；它排在目标 VLM 稳定之后。Diffusion、flow matching 和图像/音视频生成需要独立 objective/engine family，当前明确不实现。

Ascend multi-node was explicitly postponed. No torch-npu/HCCL claim is present. Future scale/Ascend work prioritizes a pinned VeOmni adapter and must validate device discovery、HCCL launch、AMP、SP/EP/FSDP2、checkpoint、operator coverage and multi-node exact/stage-boundary resume against the existing engine contract。
