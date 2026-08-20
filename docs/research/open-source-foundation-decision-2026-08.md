# TrainOmni 开源底座选型与 Build-vs-Adopt 决策

- 状态：Accepted for architecture v1
- 日期：2026-08-20
- 决策范围：VLM-first 的全模态训练框架，不针对某一个模型组合；音频理解后续，生成训练延后
- 核心决定：选择性自研（selective build），不 fork 大一统框架，也不从零重写数值计算、分布式与在线 RL 基础设施

## 1. 结论

TrainOmni 自研一个薄而权威的框架核心：它拥有数据语义、模型插件协议、训练阶段图、能力协商、运行清单、精确恢复和 CLI；具体数值执行复用成熟开源组件，并通过可替换 backend 接入。

默认技术栈：

| 层 | 默认选择 | TrainOmni 对它的使用方式 |
|---|---|---|
| 模型 ABI | PyTorch + Hugging Face Transformers | 硬依赖；模型插件最终产出普通 `nn.Module`、processor/tokenizer 和显式 forward schema |
| 单卡/DDP/FSDP2 | PyTorch 原生能力；Accelerate 负责 launcher/device glue | TrainOmni 自有训练循环和状态协议，不继承 `Trainer` 控制面 |
| 分布式 checkpoint | PyTorch Distributed Checkpoint（DCP） | 包装为 TrainOmni checkpoint manifest，补齐数据、recipe、lineage 与原子发布 |
| VLM/Omni 规模化训练 | VeOmni | 首选未来 scale engine；复用 FSDP2、SP/EP、动态 batching、DCP 与 GPU/NPU 路径，通过窄 adapter 隔离其快速演进 |
| 数据加载 | JSONL/Parquet/HF Datasets；TorchData `StatefulDataLoader` | 自有 canonical schema、mixture、packer 与跨 rank 状态聚合；不把第三方 row schema 当公共协议 |
| PEFT | Hugging Face PEFT | 作为 component policy 的一种实现，不让 LoRA target 字符串进入 canonical recipe |
| 离线 post-training | TRL | 算法适配器；复用 DPO/KTO/GRPO/KD 等实现，不采用其数据列格式作为框架协议 |
| 规模化监督训练备选 | NeMo AutoModel | NVIDIA 集群或 VeOmni 不适用时的可选 engine；用于 TP/PP/CP/FP8 与成熟 checkpoint 路径 |
| 在线 RL/RLVR | veRL | 首选 online-RL engine；训练/rollout/backend 组合由适配器管理 |
| 异步 agentic RL | AReaL | 后续可选 backend；服务化 agent workflow 和异步 rollout |
| Rollout | vLLM / SGLang | provider 插件，不与 objective 或 model plugin 绑定 |
| 多模态评测 | lmms-eval | 首选外部 benchmark adapter；EvalScope 作为更广评测/性能测试候选 |

一句话描述：**TrainOmni 自己决定“训练什么、数据是什么意思、如何复现”，开源后端决定“这一阶段如何高效执行”。**

## 2. 为什么既不 fork，也不纯手搓

### 2.1 不 fork ms-swift / LLaMA-Factory / Axolotl

这些项目适合直接完成常见微调任务，但作为二次开发母体有三个长期成本：

1. 模型接入往往同时修改 model registry、template、encode、collator 和参数系统；模型边界不是一个足够窄的插件。
2. canonical data 会被 `<image>` 占位符、并列 media 数组、特定 Trainer 列名或模板规则反向定义。
3. 框架同时承载训练、推理、量化、Web UI、模型 Hub 兼容等大量职责，升级时冲突面较大。

直接 fork 能更快得到“能跑”的第一条命令，却很难达到“以后接任意 VLM 只注册模型”的目标。因此这些框架被定位为：

- capability oracle：检查功能面是否漏项；
- baseline：做编码、loss 和吞吐回归对照；
- importer/exporter：必要时与其数据/recipe 互转；
- UX 参考：借鉴 CLI、配置和错误提示。

### 2.2 不纯手搓

以下能力不应自研：

- FSDP2、DDP、TP/PP/CP 通信原语；
- 分布式 optimizer state 重分片；
- FlashAttention、FP8、量化 kernel；
- vLLM/SGLang rollout 调度；
- PPO/GRPO 等大规模 actor/reference/reward orchestration；
- 已被广泛验证的 DPO/KTO 等算法公式实现；
- 大量公开 benchmark 的下载、prompt 和评分实现。

这些部分对数值正确性、硬件与版本组合高度敏感，复制代码会制造一个难以持续验证的私有分支。TrainOmni 只包装它们，并对输入输出、能力、版本和恢复语义加约束。

### 2.3 必须自研的部分

以下接口一旦交给某个外部框架，未来替换成本最高，因此由 TrainOmni 持有：

1. canonical multimodal sample 与数据资产语义；
2. dataset importer、mixture、deterministic transform、cost model、packer plan 和 sample trace；
3. Model Family Plugin、component catalog、capabilities 与 state-dict/export adapter；
4. ObjectiveSpec 与 StageSpec，不把算法和执行引擎写死；
5. Pipeline DAG、配置编译、静态校验和阶段间 lineage；
6. checkpoint manifest、exact/weights-only resume 语义与 run provenance；
7. engine、rollout、reward、eval provider 的窄适配器协议；
8. `validate`、`inspect`、`dry-run`、`train`、`resume`、`export` CLI。

## 3. 候选开源框架的职责判断

评分不是项目质量排名，而是判断它能否成为 TrainOmni 的稳定底座。

| 项目 | 强项 | 主要边界/风险 | 决策 |
|---|---|---|---|
| PyTorch | 数值计算、FSDP2、DCP、DTensor、compile | 不提供 VLM 数据/阶段控制面 | 硬底座 |
| Transformers | HF 模型/processor/checkpoint 生态 | `Auto*` 不等于所有模型具有一致 collator/forward | 硬底座与模型 ABI |
| Accelerate | launcher、device、DDP/FSDP/DeepSpeed glue | 若让其接管全部保存，难表达 TrainOmni 数据状态和 lineage | 只作默认 engine 的薄执行依赖 |
| TorchTitan | PyTorch-native 预训练、FSDP2/TP/PP/CP、DCP、可观测性 | 仍在快速开发；最新能力常要求 PyTorch nightly；当前仍偏 LLM recipe | 代码结构与规模化实现参考，不作稳定硬依赖 |
| VeOmni | Torch-native VLM/Omni/DiT、FSDP2、SP/EP、动态 batching、DCP、GPU/ROCm/Ascend | v0.x 快速演进；自定义组合模型仍需 parallel plan；不能替代 TrainOmni canonical/pipeline | P1 首选 scale/Ascend engine，固定 release/commit 后适配 |
| NeMo AutoModel | HF 兼容、FSDP2/TP/PP/CP/FP8、VLM recipe、精确 checkpoint | NVIDIA/CUDA 重、配置对象图和 family collator 仍有耦合 | P1 规模化备选 engine |
| OLMo-core | 清晰的预训练构件、data mix、checkpoint、模型扩展 | OLMo-native、LLM/pretraining 中心，不是通用 HF VLM 框架 | 数据/预训练设计参考 |
| ms-swift | 当前最广的一站式 VLM 训练与 RL 能力面 | model/template/encode/collator 与大参数面耦合 | oracle、baseline、可选桥接，不 fork |
| LLaMA-Factory | 低门槛 UX、大量模板、常见 SFT/偏好训练 | 更适合已支持模型的微调；研究级数据与阶段抽象较弱 | UX/recipe 参考，不 fork |
| Axolotl | HF 微调、配置化、近年的多模态与 GRPO 能力 | 同样以 fine-tuning product 为中心，canonical/模型协议不由我们控制 | baseline/UX 参考，不 fork |
| XTuner V1 | 大 MoE、长序列、Ulysses/FSDP/FP8 | 演进较快，设计重心不是通用小/中型 VLM 研发 | P2 性能后端候选 |
| TRL | SFT、DPO、KTO、GRPO、蒸馏等算法更新快 | Trainer 各自定义 dataset shape；不负责完整预训练/数据恢复 | post-training 算法 adapter |
| Molmo2 | VLM formatter→processor→packer→collator 分层与 media-aware CP | 单一模型家族研究代码 | 数据管线设计参考 |
| veRL | FSDP/FSDP2/Megatron、vLLM/SGLang、VLM、多轮工具、丰富 RL 算法 | 运行时和依赖重；不应承担 CPT/SFT 控制面 | online-RL 首选 engine |
| AReaL | 异步 agentic RL、外部 agent workflow、FSDP2/TP/CP | 专用异步 RL 系统，监督预训练不是核心 | agentic-RL 可选 engine |
| slime | Megatron + SGLang 大规模 RL、训练/推理解耦 | 对集群和 Megatron 侵入较深 | P2 专项 backend 观察项 |
| OpenRLHF | Ray-based RLHF、VLM/多轮能力 | 与 veRL/AReaL 职责重叠，首版同时支持收益低 | 暂不首选，保留适配可能 |
| lmms-eval | image/video/audio 多模态 benchmark | 不是训练内轻量 validation loop | 外部 benchmark 首选 adapter |
| EvalScope | 模型能力、agent、性能压测和报告；可桥 VLMEvalKit | 多模态自定义模型路径依赖其 backend 约束 | P1/P2 eval/perf adapter |

## 4. 默认 engine 的精确边界

`torch` engine 是首个可用 backend，但不是重新实现一个通用 Trainer。

它负责：

- 调用模型插件产出的 `nn.Module` 与 `ModelBatch`；
- 标准 forward/backward、gradient accumulation、optimizer/scheduler、AMP；
- 单进程、DDP、FSDP2；
- activation checkpointing、`torch.compile` 的能力开关；
- DCP 保存模型和 optimizer shard；
- 调用 TrainOmni hook、metrics、eval gate 和 checkpoint manager。

Accelerate 只负责：

- 进程启动与 rank/device 环境；
- 基础 prepare/unwrap/accumulate/autocast glue；
- DDP/FSDP2 常见兼容层。

Accelerate 不拥有：

- recipe schema；
- canonical dataset；
- batch/packing 语义；
- exact-resume 定义；
- run directory 与 checkpoint manifest；
- stage transition；
- model export。

如果 Accelerate 的某个版本阻碍 DCP、FSDP2 或精确恢复，`torch` engine 可以直接调用 PyTorch API，而不改变上层接口。也就是说，Accelerate 是实现细节，不是公共 ABI。

### 4.1 VeOmni scale engine

VeOmni 与常规微调产品不同，它的 model-centric distributed recipe、Torch-native FSDP2、Ulysses SP、EP、动态 batching、DCP 和 Ascend 路径与 TrainOmni 的长期规模化目标高度一致。TrainOmni 不 fork VeOmni，也不让其配置或数据格式成为公共协议；adapter 负责把已解析的 Stage、Model Plugin 和 artifact 投影到一个固定版本的 VeOmni 执行请求，再收回 metrics/checkpoint/lineage。

当前已实现受控的 VeOmni VLM command bridge：强制 immutable backend revision、固定 bridge API、shell-free 执行、版本化 request/result contract，并在 conformance 前拒绝声称 exact resume。这仍不等于已经运行真实 VeOmni package。正式提升为 native scale backend 前必须通过：自定义 ViT+LLM 组合模型、数据语义一致性、2+ rank FSDP2/SP、DCP 恢复、不同 world-size model-only load，以及后续 Ascend 多机矩阵。[VeOmni repository](https://github.com/ByteDance-Seed/VeOmni)；[VeOmni paper](https://arxiv.org/abs/2508.02317)

## 5. 数据底座决策

### 5.1 存储格式与语义协议分离

canonical sample 是逻辑协议，不要求所有数据重写成一种物理格式。reader 首批支持：

- JSON / JSONL：小规模、自定义 annotation 和调试；
- Parquet / Arrow / Hugging Face Datasets：可索引数据与高效列扫描；
- WebDataset TAR：海量 media streaming；
- 用户 Python reader plugin：数据库、对象存储或项目私有数据。

所有 reader 输出同一种 canonical sample；下游不感知原始存储。

### 5.2 StatefulDataLoader 只能解决一部分恢复

TorchData `StatefulDataLoader` 提供 `state_dict/load_state_dict` 和 worker state 聚合，但官方文档明确指出它不负责跨 rank 聚合。TrainOmni checkpoint manager 因此仍需保存：

- 每 rank reader/shard cursor；
- mixture sampler RNG 与已消费计数；
- batch planner/packer buffer；
- transform RNG；
- microbatch/gradient accumulation position；
- world-size/topology 与恢复策略。

WebDataset 适合 streaming/resampling，但精确 epoch、跨 rank 去重和中途恢复需要明确 shard policy；因此它是 reader backend，不是 sampler/checkpoint 协议。

## 6. Post-training 与 RL 的接法

### 6.1 TRL：算法库，不是 canonical trainer

TrainOmni 的 adapter 把 canonical preference/prompt sample 和 ModelBatch 投影成具体 TRL Trainer 所需输入，并把：

- resolved config；
- reference/teacher model identity；
- algorithm-specific state；
- optimizer 与数据状态；
- 产出 checkpoint；

重新纳入 TrainOmni manifest。若某算法可以稳定抽出 `compute_loss`，优先作为 Objective 实现复用；只有 rollout/orchestration 与 Trainer 深耦合时才委托整个 stage。

### 6.2 veRL/AReaL：整个 stage 的 delegated engine

在线 RL 包含 actor、reference、reward、rollout 和权重同步，不能假装成普通 `compute_loss`。TrainOmni 对这类 backend 的控制粒度是 stage：

1. 导出模型、processor 和 rollout contract；
2. 生成 backend-specific resolved config；
3. 启动/监控 backend；
4. 收集 metrics、sample/reward provenance 与 checkpoint；
5. 验证输出并登记到 pipeline lineage。

这样不复制 veRL/AReaL 内部 orchestration，也不让其配置扩散到其他阶段。

## 7. 模型接入成本的硬目标

一个符合 Transformers 约定的新 VLM 家族应通过独立 plugin package 接入，核心代码零修改。插件最少实现：

1. `ModelFactory`：加载/组装 model 与 processor；
2. `ComponentCatalog`：稳定标识 vision/connector/LLM 等组件；
3. `ContentFormatter` + `ProcessorAdapter`：canonical blocks 到模型输入；
4. `BatchCollator`：精确 forward kwargs；
5. `Capabilities`：modality、objective、packing、parallelism 等；
6. `CheckpointAdapter`：训练 state 与可部署 checkpoint 的映射；
7. conformance fixtures：至少一组 encode、forward、loss-mask、save/load 测试。

只有模型具有特殊并行切分时才额外提供 `ParallelPlan`。算法、dataset reader、CLI 和 stage compiler 不因新模型改变。

验收条件不是“注册表里有名字”，而是：

- 不改 `trainomni` core 即可发现插件；
- `trainomni inspect-model` 能显示组件和能力；
- canonical fixture 能稳定 encode/collate；
- tiny forward/backward 与 save/load 通过；
- 不支持的 recipe 在加载完整权重前报错。

## 8. 依赖与发布策略

为避免“大而全”依赖重现，按 extras 分包：

```text
trainomni-core       torch, transformers, safetensors, typed config/runtime
trainomni-data       datasets, pyarrow, torchdata, optional webdataset
trainomni-peft       peft
trainomni-trl        trl
trainomni-nemo       nemo-automodel and NVIDIA stack
trainomni-rl-verl    verl + chosen rollout runtime
trainomni-rl-areal   areal
trainomni-eval       lmms-eval / EvalScope adapters
trainomni-dev        schema, test, lint and documentation tooling
```

这里是依赖分组设计，不要求真的发布多个 wheel；首版可以用一个项目的 optional dependency groups 实现。每个 engine 维护经过验证的版本矩阵和最小 smoke，不保证任意版本自由组合。

### 8.1 开源与许可证策略

首选运行时本身均采用宽松开源许可：PyTorch/TorchTitan 为 BSD-3-Clause；Transformers、Accelerate、PEFT、TRL、NeMo AutoModel、veRL 与 AReaL 为 Apache-2.0；lmms-eval 的主 pipeline 沿用 MIT，而其新增多模态 model/task 代码采用 Apache-2.0。这个组合允许以“依赖 + 适配器”的方式集成，不需要复制整个项目源码。

但“框架开源”不自动解决下列许可问题：

- 模型权重、tokenizer 和 processor 可能有独立 license/acceptable-use 条款；
- benchmark 和训练数据有各自 license、地域、隐私和再分发限制；
- CUDA、Transformer Engine、某些 kernel/container 和在线 judge 服务可能有额外条款；
- 复制第三方源码时必须保留 NOTICE/copyright；能通过依赖调用就不 vendoring；
- artifact manifest 必须记录 code/model/data license identifier 和来源，未知许可默认不进入可发布产物。

正式发布 TrainOmni 前仍需生成第三方依赖清单和 NOTICE；本文是工程选型判断，不替代法律审查。

## 9. 风险与复审条件

| 风险 | 当前控制措施 | 何时复审 |
|---|---|---|
| 自有控制面也可能逐渐膨胀 | 公共协议保持窄；backend-specific config 隔离在 adapter namespace | 核心包开始依赖具体 RL/集群运行时时 |
| 默认 torch loop 重复生态工作 | 只实现 CPT/SFT 通用 loop；复杂算法委托 TRL/veRL | 维护成本超过上层协议收益时 |
| Accelerate FSDP2/DCP 版本漂移 | 它不是 ABI；保留直接 PyTorch 实现路径 | exact-resume 或 optimizer state 无法验证时 |
| NeMo 与 torch engine 语义不一致 | 同一 canonical fixture、loss 和 checkpoint lineage cross-check | 引入 NeMo backend 前后 |
| backend 输出不可完全恢复 | manifest 标出 `exact`、`stage-boundary`、`weights-only` 三档恢复能力 | 每个 backend 接入时 |
| 插件仍需大量定制 | conformance suite 和“core 零修改”门禁 | 第二、第三个异构 VLM 接入后 |

会推翻本 ADR 的证据只有两类：

1. 某个开源框架提供稳定、窄、模型无关的数据/插件/阶段协议，并能覆盖全生命周期且升级成本显著更低；
2. 实际接入三个差异明显的 VLM 后，TrainOmni 的公共抽象比直接适配现有框架更复杂且无法共享。

在此之前，不因单一模型的特殊 forward 或单次训练 deadline 改变核心边界。

## 10. 主要官方资料

- [PyTorch TorchTitan](https://github.com/pytorch/torchtitan)：PyTorch-native 规模化训练、FSDP2/TP/PP/CP、DCP、checkpointable data loading；仓库也明确标注仍在快速演进。
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [Hugging Face Accelerate FSDP](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp)
- [TorchData StatefulDataLoader](https://meta-pytorch.org/data/main/torchdata.stateful_dataloader.html)
- [WebDataset](https://github.com/webdataset/webdataset) 与 [WIDS](https://github.com/webdataset/wids)
- [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) 与 [repository structure](https://docs.nvidia.com/nemo/automodel/get-started/repo-structure)
- [OLMo-core](https://github.com/allenai/OLMo-core)
- [ms-swift](https://github.com/modelscope/ms-swift) 与 [custom model architecture](https://swift.readthedocs.io/en/v4.0/Customization/Architecture.html)
- [TRL](https://github.com/huggingface/trl) 与 [dataset formats](https://huggingface.co/docs/trl/en/dataset_formats)
- [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)
- [XTuner](https://github.com/InternLM/xtuner)
- [Molmo2](https://github.com/allenai/molmo2)
- [veRL](https://github.com/verl-project/verl)
- [AReaL](https://github.com/inclusionAI/AReaL)
- [slime](https://github.com/THUDM/slime)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [EvalScope](https://github.com/modelscope/evalscope)
