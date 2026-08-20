# TrainOmni Framework 需求基线 v0.1

- 状态：Superseded by [Requirements v1](requirements-v1.md)
- 日期：2026-08-19
- 目标：支持自定义 ViT + LLM 的低摩擦 VLM 训练，同时保留扩展到 SOTA post-training 和规模化后端的路径

> 本文保留早期以目标小模型为中心的 P0/P1/P2 需求记录。当前通用框架范围、M1–M6 里程碑与验收条件以 v1 为准。

## 1. 优先级定义

- P0：首个可训练版本不可缺少；没有它就无法可信地完成 L0/SFT。
- P1：首个稳定版本需要；接口必须在 P0 预留。
- P2：规模化或前沿能力；不阻塞目标小模型的第一轮训练。

## 2. 功能需求

| ID | 优先级 | 需求 | 最小验收证据 |
|---|---|---|---|
| DATA-001 | P0 | JSONL、Parquet 或 Hugging Face Dataset 导入到统一 sample | 同一 SFT 样例经三种 reader 得到相同 canonical hash |
| DATA-002 | P0 | image、multi-image、interleaved text-image | schema、encode 和 2-step train smoke 全部通过 |
| DATA-003 | P0 | typed text/media/bbox/point/JSON/tool blocks | 模型专属 tag 不出现在 canonical fixture |
| DATA-004 | P0 | CPT、SFT、preference、prompt-only 四类 objective | schema 对必需字段做 discriminated validation |
| DATA-005 | P0 | dataset mixture、权重、seed、样本来源 trace | 固定 seed 的前 N 个 sample id 一致 |
| DATA-006 | P0 | block/turn 级 loss weight，派生 token loss mask | 可视化 trace 中 mask 与模板 token 对齐 |
| DATA-007 | P0 | deterministic transforms 和坏样本策略 | skip/resample/fail 策略均有测试且记录计数 |
| DATA-008 | P1 | streaming、WebDataset/对象存储、stateful shard | 中断恢复后下一样本一致 |
| DATA-009 | P1 | video 和 frame sampling | frame budget、时间区间与 processor 输出可追踪 |
| DATA-010 | P2 | audio 和全模态混合 | capability gate 与至少一个集成测试 |
| MODEL-001 | P0 | 自定义 ViT + connector + LLM 组装 | 能加载目标 checkpoint 并完成 forward/backward |
| MODEL-002 | P0 | 原生 Transformers VLM 适配 | 至少一个公开小 VLM smoke |
| MODEL-003 | P0 | 组件发现与稳定命名 | 参数无遗漏、无重复地归入 vision/connector/llm/other |
| MODEL-004 | P0 | 按组件 freeze、LR、weight decay、dtype、grad clip | trainable 参数快照与 config 一致 |
| MODEL-005 | P0 | full tuning 与 LoRA | 两条 2-step smoke 均可保存/重载 |
| MODEL-006 | P0 | 模型 capability 声明与训练前校验 | 不支持的 modality/objective 在启动前失败 |
| MODEL-007 | P1 | vision merger/resampler、多 projector 类型 | plugin fixture 覆盖至少两种连接器 |
| MODEL-008 | P1 | teacher/student distillation | teacher 不进入 optimizer，loss 项可分解记录 |
| RECIPE-001 | P0 | projector alignment、CPT、SFT recipe | 每种 recipe 有 resolved config 和 smoke test |
| RECIPE-002 | P0 | stage 可覆盖 model/data/optimizer/schedule/eval/checkpoint | resolved config 不依赖隐式全局变量 |
| RECIPE-003 | P1 | DPO/MPO/KTO 或等价 preference recipe | shared/different media pair 均能 collate |
| RECIPE-004 | P1 | sequence/on-policy distillation | teacher provenance 写入 run manifest |
| RECIPE-005 | P2 | GRPO/GSPO/RLVR 和 agentic multi-turn | rollout/reward/environment backend 集成测试 |
| PERF-001 | P0 | BF16、gradient accumulation/checkpointing | 与 FP32 小样例的 loss 差异在阈值内 |
| PERF-002 | P0 | length + visual-cost aware batching | batch 不超过 token/pixel budget |
| PERF-003 | P1 | packing 与 padding-free | segment attention 隔离和 loss mask 测试 |
| DIST-001 | P0 | 单卡和 DDP | 单卡/双卡一轮后的有效 batch 与 loss 可比 |
| DIST-002 | P0 | FSDP2 full-shard | 2+ GPU save/resume smoke，若无硬件则 CI 标记待执行 |
| DIST-003 | P1 | DeepSpeed ZeRO-2/3 兼容 backend | 与同 recipe 的参数更新 cross-check |
| DIST-004 | P1 | NeMo AutoModel backend | canonical sample 与 model plugin 无需修改即可切换 |
| DIST-005 | P2 | TP/PP/CP/SP/EP 或 Megatron backend | 规模化 benchmark 和 topology-aware checkpoint |
| CKPT-001 | P0 | 保存 model、optimizer、scheduler、RNG、step | resumed run 的下一 optimizer step 匹配连续 run |
| CKPT-002 | P0 | 保存 sampler、mixer、packer/dataloader state | resumed run 的 sample id 序列匹配连续 run |
| CKPT-003 | P0 | dataset/code/config/environment fingerprint | run manifest 可独立审计 |
| CKPT-004 | P0 | HF-compatible consolidated export | `from_pretrained` + processor 重载成功 |
| CKPT-005 | P1 | sharded DCP、异步保存、topology reshard | 改 DP topology 后能恢复 |
| EVAL-001 | P0 | loss、吞吐、token/pixel、grad/param metrics | structured log 含 recipe、data source、component 维度 |
| EVAL-002 | P0 | 训练前 sample/encoding trace | 人可读地显示内容、模板、token、labels、media shapes |
| EVAL-003 | P1 | generation eval 与 task metric plugin | 至少 caption/VQA/grounding 三类 adapter |
| UX-001 | P0 | YAML 配置 + typed validation + CLI override | 未知字段、非法组合直接报错 |
| UX-002 | P0 | dry-run：不训练，只解析、校验和估算 | 输出 resolved config、参数量、样本与显存风险 |
| UX-003 | P0 | 可读错误与 provenance | 错误包含 sample id、source、adapter 和 stage |
| UX-004 | P1 | recipe registry 和最小 Web/UI 薄层 | UI 只生成/校验 recipe，不复制训练逻辑 |

## 3. 非功能需求

### 3.1 可组合性

- Canonical data package 不 import Transformers、TRL、NeMo 或 ms-swift。
- Model plugin 不依赖具体 engine；engine 只消费标准的 model bundle、batch 和 objective hooks。
- 每个 backend 都通过 capability negotiation 明确支持的 objective、precision、parallelism 和 checkpoint mode。

### 3.2 可复现性

- 所有随机源显式 seed；rank/worker/sample 派生 seed 规则固定并记录。
- 运行目录保存原始 config 和 fully resolved config。
- 每个样本保留 `sample_id`、dataset fingerprint 和 transform trace；禁止训练前静默丢弃坏样本。
- “精确恢复”与“只恢复权重”是两个不同命令和状态，不能隐式降级。

### 3.3 可维护性

- 公共 Protocol 小而稳定，模型特例留在 plugin 内。
- Config schema 使用版本号；不兼容修改必须提供 migration。
- optional backend 使用 extras 安装，默认环境不引入 NeMo、Megatron、Ray、vLLM。
- 不通过任意 YAML `_target_` 默认执行用户代码；自定义 Python plugin 需显式 `--allow-custom-code`。

### 3.4 可观测性

- 至少记录 samples/s、text tokens/s、visual tokens 或 pixels/s、padding ratio、pack utilization。
- 分组件记录 trainable params、LR、grad norm 和 overflow/NaN。
- 记录坏样本、重试、跳过、媒体 decode latency 和 data wait time。

## 4. v0.1 明确不做

- 不开发自己的 CUDA kernel、FlashAttention、optimizer 或 rollout inference engine。
- 不复制完整的 ms-swift/LLaMA-Factory Web UI、量化与部署模块。
- 不在没有真实目标模型接口前固化 Qwen 专属 bbox tag 或 MiniCPM 专属 chat template。
- 不承诺首版支持 audio、超长视频、MoE expert parallel 或在线 multi-turn agent RL。
- 不把实验生成的 checkpoint、dataset、cache、run log 写进 `Framework/`。

## 5. v0.1 通过门槛

v0.1 只有同时满足以下条件才算“可用”，不能只以脚本跑通作为完成：

1. canonical schema、reader、validator 和 sample trace 有单元测试。
2. 目标 ViT + LLM plugin 能 dry-run，组件和参数策略审计无遗漏。
3. projector alignment 与 SFT 各完成至少 2 optimizer steps，loss 有限且目标组件发生更新。
4. 连续 4 steps 与 2 steps + save/resume + 2 steps 的 sample id、LR 和 loss 在定义阈值内一致。
5. checkpoint 可导出并以 HF API 重新加载执行一次 generation/forward。
6. README、resolved config、验证命令、已知问题和硬件/依赖版本齐全。

## 6. 等待外部输入

Research 任务需要提供：

- 目标 vision tower、merger/projector、LLM 的准确 checkpoint/revision 与 Transformers class。
- L0 训练阶段、冻结策略、loss、数据类型、上下文/像素预算和预计硬件。
- connector 输入/输出 shape、视觉 token 排列、position encoding 和 special token 约束。

Downloader 任务需要提供：

- 本地 checkpoint 路径、revision/commit、文件清单/hash、processor/tokenizer 和模型卡。
- checkpoint 是否含 remote code，以及离线加载所需的额外 Python 文件。
