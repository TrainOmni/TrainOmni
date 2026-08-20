# TrainOmni v1 Support Matrix

当前产品策略是 VLM-first。名称为 TrainOmni 是为了稳定长期公共协议，不表示 v1 已实现全部输入/输出模态或 diffusion 生成训练。

本文区分四种状态，避免把“接口存在”误写成“本地原生实现”：

- **Native**：TrainOmni 自有执行路径并有测试。
- **Delegated**：TrainOmni 管理 recipe、授权、lineage、metrics 和结果，但数值/编排由外部开源 backend 执行。
- **Plugin**：语义由公共协议定义，具体实现属于模型或数据插件。
- **Deferred**：本版本明确不实现。

## Training lifecycle

| Stage | Control plane | Default execution | State |
|---|---|---|---|
| Vision preparation | Stage/Pipeline | torch or delegated | Native control, pluggable execution |
| Modality alignment | Stage/Pipeline | torch masked causal LM | Native |
| Multimodal CPT | Stage/Pipeline | torch masked causal LM | Native |
| Capability curriculum | Stage/Pipeline | torch | Native |
| Instruction SFT | Stage/Pipeline | torch | Native |
| Reasoning distillation | Stage/Pipeline | TRL/NeMo/custom | Delegated |
| Reward/verifier | Stage/Pipeline | custom/TRL/veRL | Delegated |
| Offline preference | Stage/Pipeline | TRL/custom DPO | Delegated |
| Online RL/RLVR | Stage/Pipeline | veRL/TRL/custom GRPO/PPO | Delegated |
| Agentic RL | Stage/Pipeline | veRL/AReaL/custom | Delegated |
| Evaluation/export | Stage or CLI | internal/external/plugin | Native control |

## Data

| Capability | State | Owner |
|---|---|---|
| Canonical text/image/video/audio assets | Native representation | core |
| Multi-turn chat, tool call/result, JSON | Native | core |
| BBox/point grounding | Native | core |
| Preference pair and rollout/verifier metadata | Native | core |
| JSON/JSONL | Native reader | core |
| Parquet | Native optional reader | core + `pyarrow` extra |
| TAR JSON records | Native reader | core |
| HF Dataset, database, object store, project schema | Plugin | explicit data plugin |
| Deterministic weighted mixture/repeat | Native | core |
| Multi-budget batch planner | Native | core |
| Token/media packing representation | Plugin | model collator, core `BatchPlan` |
| Rank-stable variable-cost sharding | Native | core |
| Reader/mixture/look-ahead exact resume | Native | core |

## Model integration

| Capability | State |
|---|---|
| External zero-core-edit registration | Native |
| Explicit trust boundary | Native |
| Static capabilities and negative negotiation | Native |
| Exact parameter-to-component cover | Native |
| Canonical encode/collate | Plugin |
| Processor/chat template/loss mask | Plugin with conformance tests |
| Auxiliary teacher/reference/reward models | `ModelBundle` native contract |
| HF/plugin export | Native orchestration, plugin serializer |
| Public tiny LLaVA example | Verified |
| Audio encoder integration | Planned after target VLM | model/data plugin + conformance |
| Diffusion/image-video-audio generation | Deferred | future objective/engine family |

## Execution and optimization

| Capability | Single | DDP | FSDP2 | Delegated |
|---|---:|---:|---:|---:|
| Forward/backward | Native | Native | Native | Backend |
| Component freeze and param groups | Native | Native | Native | Backend contract |
| Gradient accumulation/clipping | Native | Native | Native | Backend contract |
| FP32/TF32/FP16/BF16 | Native | Native | Native | Capability negotiation |
| FP8 | — | — | — | NeMo/backend |
| Activation checkpointing | Native model hook | Native | Native | Backend |
| LoRA/QLoRA | Native optional PEFT | Native | Native | Backend |
| `torch.compile` | Native option | Native option | Backend/version dependent | Backend |
| TP/PP/CP/SP/EP | — | — | — | VeOmni/NeMo/veRL/custom |

## Checkpoint and recovery

| Capability | State |
|---|---|
| Single-process exact checkpoint | Native atomic local |
| DDP exact checkpoint | Native per-rank local + barrier |
| FSDP2 sharded model/optimizer | Native DCP |
| Scheduler/scaler/step/token/RNG/data state | Native |
| Uninterrupted vs resume equality | Verified single, DDP, FSDP2 |
| FSDP2 exact resume with changed world size | Not claimed; runtime state is topology-specific |
| DCP model-only reshard/load to single process | Native and verified |
| Safe deploy export | plugin-owned safe format; local exact pickle requires trust |

## Infrastructure

| Capability | State |
|---|---|
| Local process | Native |
| `torchrun` environment | Native engine contract |
| Windows CPU multi-process smoke | Native file-rendezvous launcher |
| Slurm/Kubernetes/Ray | Delegated launcher/backend |
| VeOmni scale backend | Pinned/versioned VLM command bridge implemented; native package execution and exact resume not yet claimed |
| Ascend/昇腾 multi-node | Deferred by user direction |

## Priority order

1. Target VLM: image/video understanding, alignment/CPT/SFT and reproducible scale-up.
2. Audio understanding: audio encoder/projector, temporal cost and conformance fixtures.
3. Continuous generative training: diffusion/flow-matching objectives and runtime; explicitly deferred.

## Capability claims rule

一个 model/engine plugin 只能声明通过 conformance 的能力。`parallelism: fsdp2`、`packing: true` 或 `resume_level: exact` 都不是提示词；声明后必须有对应 tiny smoke、错误路径和保存/恢复证据。核心遇到不支持组合会提前失败，不做静默 fallback。
