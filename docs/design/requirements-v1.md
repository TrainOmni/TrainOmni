# TrainOmni Framework Requirements v1

> 本文是可追踪验收基线，不等同于全部功能都由 core 原生执行。Framework v1 的实际 Native/Delegated/Plugin/Deferred 状态以 [support matrix](support-matrix-v1.md) 和 [verification report](../verification/verification-2026-08-20.md) 为准。

- 状态：Accepted baseline
- 日期：2026-08-20
- 依据：[Framework Blueprint v1](framework-blueprint-v1.md)、[VLM 完整训练生命周期](vlm-training-lifecycle-v1.md)
- 说明：`M1–M6` 是最早交付里程碑，不表示最终只支持该阶段

## 1. 数据与多模态语义

| ID | 最早里程碑 | 必须支持 | 验收证据 |
|---|---:|---|---|
| DATA-001 | M1 | versioned canonical sample、asset 和 objective schema | 正反例 fixture、稳定 hash、schema migration test |
| DATA-002 | M1 | image、multi-image、interleaved text-image | encode/inspect/2-step smoke |
| DATA-003 | M1 | text/image/video/audio/bbox/point/json/tool typed block 协议 | canonical fixture 不含模型专属 tag；未实现模态可被 capability gate 拒绝 |
| DATA-004 | M1 | CPT、SFT、preference、prompt-only；后续 reward/episode | discriminated validation 与 objective requirements |
| DATA-005 | M1 | JSON/JSONL reader 与可插拔 importer | 两种外部格式映射为相同 canonical hash |
| DATA-006 | M2 | Parquet/Arrow/HF Datasets 与 WebDataset reader | 同样本跨 reader 语义一致；分布式 shard test |
| DATA-007 | M2 | deterministic transform、坏样本 quarantine/drop/fail | 固定 seed 重放；drop reason/count 可追踪 |
| DATA-008 | M2 | dataset manifest、mixture、temperature/dynamic sampling、curriculum | 固定 seed 的 sample ID 序列和比例可复现 |
| DATA-009 | M2 | token/pixel/tile/frame 多维 cost 与 budget batching | 实际 batch 不越硬限制；估算误差有指标 |
| DATA-010 | M2 | stateful packer、padding-free/segment isolation | cross-segment attention 与 loss-mask 负例测试 |
| DATA-011 | M2 | turn/span/token loss weight 与 source span | inspect 输出可回溯到原始 block |
| DATA-012 | M2 | 基础 video/frame sampling | frame/time-range/resize trace 与 smoke |
| DATA-013 | M5 | episode/tool-returned media lineage | 多轮 replay 的 media 引用一致 |
| DATA-014 | M6 | audio encode/batching 实现 | 至少一模型一数据集 integration test |

## 2. 模型插件与参数策略

| ID | 最早里程碑 | 必须支持 | 验收证据 |
|---|---:|---|---|
| MODEL-001 | M1 | entry-point/显式路径插件发现，core 零修改 | 外部 toy plugin 安装、发现、卸载 test |
| MODEL-002 | M1 | 原生 Transformers VLM 与组合 ViT+connector+LLM | 各一 tiny model forward/backward |
| MODEL-003 | M1 | capabilities：modality/block/objective/packing/generation/engine | 不支持组合在全量权重加载前失败 |
| MODEL-004 | M1 | component catalog exact-cover、shared/tied relation | 所有参数恰好归属一个 component |
| MODEL-005 | M2 | per-component train/freeze/LR/WD/dtype/clip/activation checkpoint | resolved optimizer group 与参数快照一致 |
| MODEL-006 | M2 | full/freeze/LoRA/QLoRA 与 component-specific PEFT | 各模式 save/reload；target 不依赖用户 regex |
| MODEL-007 | M2 | formatter/processor/loss-mask/collator 子边界 | 同 canonical fixture 的 source span、labels 与 media order golden test |
| MODEL-008 | M2 | tokenizer/media-token 变更和初始化 manifest | resize 前后 shape/hash/init policy 可审计 |
| MODEL-009 | M2 | training checkpoint ↔ HF export adapter | `from_pretrained` + processor reload smoke |
| MODEL-010 | M3 | teacher/reference/reward model handles | 不进入错误 optimizer；身份进入 provenance |
| MODEL-011 | M4 | model-specific TP/PP/CP parallel plan | topology validate + distributed smoke |

## 3. 生命周期、Recipe 与 Objective

| ID | 最早里程碑 | 必须支持 | 验收证据 |
|---|---:|---|---|
| LIFE-001 | M1 | typed、versioned UserConfig → immutable ResolvedRunSpec | unknown field/invalid combination 负例；fingerprint 稳定 |
| LIFE-002 | M1 | alignment/CPT/SFT StageSpec | 三类 resolved recipe 与 tiny smoke |
| LIFE-003 | M2 | Stage DAG、artifact inputs/outputs、metric/manual gates | 顺序与分支 pipeline fixture |
| LIFE-004 | M2 | resolution/context/video curriculum 与 stage boundary | 每一 boundary 有 manifest、checkpoint、gate |
| LIFE-005 | M3 | distillation、reward/verifier、offline preference 分支 | teacher/reference lineage 和 preference smoke |
| LIFE-006 | M5 | online RL/RLVR stage | rollout/reward/backend integration smoke |
| LIFE-007 | M5 | agentic multi-turn stage | replayable tool/environment episode |
| OBJ-001 | M1 | masked/token-weighted causal LM | CPT/SFT mask、denominator、loss golden test |
| OBJ-002 | M2 | contrastive/feature alignment 扩展协议 | 外部 objective plugin test |
| OBJ-003 | M3 | sequence/logit/feature/on-policy distillation | 命名 loss 分量与 teacher provenance |
| OBJ-004 | M3 | DPO/MPO/KTO/ORPO/SimPO 可扩展 preference family | 至少两类多模态算法 smoke，其余 capability/adapter contract |
| OBJ-005 | M3 | pair/list/score reward objective | reward range/calibration/version manifest |
| OBJ-006 | M5 | PPO/GRPO/GSPO/RLOO/REINFORCE family 可委托 | engine 声明算法支持，不支持时早失败 |

## 4. Engine、分布式与性能

| ID | 最早里程碑 | 必须支持 | 验收证据 |
|---|---:|---|---|
| ENG-001 | M1 | LoopEngine 与 DelegatedStageEngine 两种协议 | fake backend contract suite |
| ENG-002 | M1 | torch 单卡/DDP、BF16/FP16、grad accumulation/clip | single/2-rank loss 与 effective batch cross-check |
| ENG-003 | M2 | PyTorch FSDP2 full shard | 2+ GPU forward/backward/save/resume |
| ENG-004 | M2 | activation checkpoint、compile/kernel capability switches | 明确兼容矩阵与性能/正确性 test |
| ENG-005 | M3 | TRL algorithm adapter | canonical → adapter，无公共 schema 泄漏 |
| ENG-006 | M4 | NeMo AutoModel delegated scale backend | 不改 canonical/model public protocol 即切换 backend |
| ENG-007 | M4 | TP/PP/CP/FP8 capability | topology-aware smoke 与 checkpoint restore |
| ENG-008 | M5 | veRL online-RL backend、vLLM/SGLang rollout | actor/reference/reward version 与权重同步 trace |
| ENG-009 | M6 | AReaL async agentic backend（按需求启用） | staleness policy、async episode 和 stage-boundary resume |
| ENG-010 | M1 | optional dependencies 分组 | core install 不 import NeMo/Ray/vLLM/eval stack |

## 5. Checkpoint、恢复与 Artifact

| ID | 最早里程碑 | 必须支持 | 验收证据 |
|---|---:|---|---|
| CKPT-001 | M1 | weights/optimizer/scheduler/scaler/step/RNG | uninterrupted vs resume local equivalence |
| CKPT-002 | M2 | reader/mixer/sampler/transform/packer/dataloader state | 下一 microbatch sample IDs/tensors 一致 |
| CKPT-003 | M2 | DCP sharded save、incomplete marker、atomic publish | 故障注入不发布损坏 checkpoint |
| CKPT-004 | M2 | exact/stage-boundary/weights-only/transfer 四种显式模式 | 禁止 exact 自动降级；lineage test |
| CKPT-005 | M2 | code/config/data/plugin/backend/environment manifest | artifact 可离线审计 |
| CKPT-006 | M4 | topology/world-size reshard | 改 DP topology 恢复；不支持组合早失败 |
| CKPT-007 | M2 | HF/adapter export 与 processor 同步发布 | reload + generation/forward smoke |
| CKPT-008 | M5 | rollout/reward/episode provenance | checkpoint 可定位生成它的 rollout model version |

## 6. 评测、检查与可观测性

| ID | 最早里程碑 | 必须支持 | 验收证据 |
|---|---:|---|---|
| OBS-001 | M1 | `validate`、`inspect data/model/batch`、`dry-run` | 人可读与 machine-readable 输出 snapshot |
| OBS-002 | M1 | loss/grad/LR/参数/数据/吞吐结构化指标 | event schema test |
| OBS-003 | M2 | token/pixel/frame、padding/pack、分段时延、坏样本指标 | fixed smoke 的完整 stage report |
| OBS-004 | M2 | in-loop generation/task metric plugin | 至少 caption/VQA 小集 adapter |
| OBS-005 | M2 | lmms-eval external benchmark adapter | 一个公开 tiny VLM + 小 benchmark smoke |
| OBS-006 | M4 | EvalScope/performance adapter | TTFT/TPOT/throughput report 与 artifact 关联 |
| OBS-007 | M5 | rollout latency/reward/staleness/episode trace | online-RL structured events |
| OBS-008 | M1 | structured issue 含 sample/source/plugin/stage/rank | 负例错误 snapshot |

## 7. 安全、维护与交付

| ID | 最早里程碑 | 必须支持 | 验收证据 |
|---|---:|---|---|
| SAFE-001 | M1 | recipe 默认不执行任意 `_target_`/Python | 恶意 config 负例 |
| SAFE-002 | M2 | media root traversal、URI allowlist、checksum/mime/decode policy | traversal/checksum bomb/格式错负例 |
| SAFE-003 | M1 | plugin/custom code 显式信任与版本记录 | 未授权插件拒绝；manifest 留痕 |
| SAFE-004 | M1 | secrets 脱敏，只保存引用 | resolved config/log snapshot 不含 token |
| MAINT-001 | M1 | public API/schema versioning 与 migration | 至少一个 fixture migration test |
| MAINT-002 | M1 | 每个 plugin/backend 的 conformance suite | CI 可单独运行 capability matrix |
| MAINT-003 | M2 | pinned tested dependency matrix | lock/constraints + smoke evidence |
| MAINT-004 | M1 | 产物位置、完成状态、验证证据、决策、问题、下一步 | `STATUS.md` 持续更新 |

## 8. 明确非目标

- 不开发自有 CUDA kernel、attention kernel、optimizer 或 rollout inference server。
- 不复制完整 Web UI、模型下载器、量化产品或部署平台；通过 exporter/provider 接口对接。
- 不保证任意 PyTorch/Transformers/NeMo/TRL/veRL 版本自由组合，只支持经过测试的版本矩阵。
- 不把模型专属 token/template、backend 配置或外部 Trainer row schema升级为公共协议。
- 不因单一模型 deadline 绕过 manifest、capability、inspect 和 conformance 门禁后再把临时代码固化进 core。

## 9. M1 架构内核通过门槛

M1 只有同时满足以下条件才完成：

1. canonical schema、config、plugin manifest、capability、artifact contract 均有版本和正反例测试；
2. 外部 tiny VLM plugin 在不修改 core 的情况下被发现并通过 component/encode/collate conformance；
3. `validate`、`inspect data/model/batch`、`dry-run` 可用；
4. torch engine 完成 single/DDP 2-step forward/backward；
5. local uninterrupted 与 save/resume 的下一 batch、LR、loss 和最终权重在阈值内一致；
6. optional backend 未安装时 core 命令仍可用并给出清晰缺依赖提示；
7. resolved config、manifest、验证命令、已知问题和依赖版本可以独立交接。
