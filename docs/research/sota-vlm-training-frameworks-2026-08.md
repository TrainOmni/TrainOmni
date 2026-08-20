# SOTA VLM 训练方式与开源框架调研

- 状态：阶段性调研归档；选型结论已收敛到 [开源底座选型 ADR](open-source-foundation-decision-2026-08.md)
- 调研日期：2026-08-19
- 适用范围：可组合 ViT + LLM、原生 VLM；预训练、SFT、偏好优化和在线 RL
- 结论置信度：框架能力以官方仓库、官方文档和论文为准；目标模型细节仍等待 Research 任务输入

> 本文是第一轮广度调研。训练阶段的规范定义见 [VLM 完整训练生命周期](../design/vlm-training-lifecycle-v1.md)，当前模块边界和分期功能见 [Framework Blueprint v1](../design/framework-blueprint-v1.md)。

## 1. 结论先行

TrainOmni 不应 fork 一个“大而全”的现有项目继续堆功能。推荐采用分层组合：

1. 用自有的、模型无关的 canonical multimodal data contract 保存原始语义。
2. 用窄接口的 Model Family Plugin 负责模型组装、chat/template、processor、loss mask、collator 和导出。
3. 用 Recipe 表达训练阶段，而不是把 `freeze_vit`、`dpo_beta` 等参数散落到命令行。
4. v0.1 默认采用 PyTorch + Transformers + FSDP2；TRL 作为 post-training 算法参考或可替换后端。
5. ms-swift 作为能力覆盖和兼容性基准，不作为 TrainOmni 的核心依赖。
6. NeMo AutoModel 作为规模化、FSDP2/TP/PP/CP、精确恢复和 HF checkpoint 互操作的首选参考/后端候选。
7. Molmo2 的 DataFormatter → MultimodalPreprocessor → Packer → model-specific Collator 分层，是数据管线设计的主要参考。
8. 在线 RL 不在 v0.1 自研；先为 TRL / veRL / EasyR1 保留 rollout、reward、environment 接口。

这套组合保留了开源生态的训练算法与分布式能力，又避免让数据协议被某个模型的 `<image>` 标签或某个 Trainer 的列名锁死。

## 2. 当前 SOTA VLM 训练生命周期

现在的高性能 VLM 通常不是一次 SFT 得到，而是多阶段课程。一个完整框架需要表达以下阶段及其状态转移。

### 2.1 视觉与连接器准备

- 复用已有 ViT，或先做视觉预训练/继续预训练。
- 只训练 projector / merger / resampler，让视觉 token 对齐 LLM 表示空间。
- 支持冻结 ViT 和 LLM、单独训练连接器；不同组件有不同学习率、weight decay、dtype 和梯度裁剪。

### 2.2 多模态预训练

- 对 interleaved text-image、caption、OCR、document、grounding、video 等混合数据做 next-token training。
- 从短上下文到长上下文逐步扩展 token budget、视觉分辨率和视频帧数。
- 支持按数据源采样权重、温度采样、质量过滤、去重和可复现的动态 mixture。

Qwen3-VL 的公开技术报告给出了 merger-only、全参数短上下文、全参数 32K、最终 256K 的分阶段课程；STEP3-VL-10B 报告了 1.2T token 的统一全参数多模态预训练。它们说明“阶段、上下文、数据 mixture、冻结策略”必须是一等配置对象，而不只是一个训练脚本。[Qwen3-VL report](https://arxiv.org/html/2511.21631)；[STEP3-VL-10B](https://arxiv.org/html/2601.09668)

### 2.3 Instruction SFT 与 reasoning/distillation

- 多轮图文对话、OCR/文档、grounding、视频、工具使用等任务混合 SFT。
- assistant-only、completion-only、指定 span/token weight 等多粒度 loss mask。
- reasoning cold-start、长 CoT SFT、teacher sampling、sequence KD 或分布级 KD。

TRL 当前已把 SFT、DPO、GRPO 和 on-policy distillation 纳入统一的 post-training 库；其 SFTTrainer 支持 conversational VLM、预编码 labels、assistant mask、packing 与 padding-free，但原始样本仍需被转换为 Trainer 需要的列格式。[TRL repository](https://github.com/huggingface/trl)；[TRL SFTTrainer](https://huggingface.co/docs/trl/main/sft_trainer)

### 2.4 Offline preference optimization

- DPO、MPO、KTO、ORPO、SimPO 等偏好学习。
- 多模态 chosen/rejected 可以共享 prompt/media，也可能各自引用不同 media。
- 必须允许 pair-level 元数据、margin、quality、teacher、judge 和安全标签。

InternVL3.5 的 Cascade RL 先用约 20 万 preference pairs 做 MPO，再用约 7 万 queries 做在线 GSPO，表明 offline preference 和 online RL 是可组合而不是互斥的阶段。[InternVL3.5](https://arxiv.org/html/2508.18265)

### 2.5 Online RL / RLVR / Agentic RL

- GRPO、GSPO、PPO、RLOO 等在线训练。
- rollout engine 与 trainer 解耦；支持 colocate 或 server 模式、权重同步、采样版本标记和 off-policy 修正。
- rule reward、model judge、execution verifier、多奖励组合及 reward provenance。
- 多轮视觉工具调用，tool result 可以返回新的图像/视频帧。

TRL GRPO 已支持 VLM 图像输入、vLLM colocate/server 和多模态工具返回；ms-swift 提供更广的 GRPO family、多轮 scheduler 和 environment 插件；veRL/EasyR1 则是更适合规模化 RL 的独立后端候选。[TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)；[ms-swift](https://github.com/modelscope/ms-swift)；[veRL](https://github.com/verl-project/verl)；[EasyR1](https://github.com/hiyouga/EasyR1)

## 3. SOTA 训练框架必须支持什么

### 3.1 数据与语义

- image、multi-image、video、audio 和 interleaved document；v0.1 至少实现 image/multi-image。
- 文本、media reference、bbox、point、JSON、tool call/result 等 typed content block。
- CPT、SFT、preference、prompt-only/rollout 四种顶层 objective。
- 坐标使用明确的 coordinate space，禁止在原始数据中写模型专属 bbox token。
- URI、相对路径、Hub/对象存储引用、checksum、license、source、split 和数据 fingerprint。
- streaming、map-style、mixture、deterministic transform、样本 trace 和坏样本策略。

### 3.2 模型与训练策略

- 原生 `AutoModelForImageTextToText` 和自定义 ViT + projector + LLM 组装。
- 组件目录：`vision_encoder`、`connector`、`language_model` 及可选 `vision_merger`、`audio_encoder`。
- 按组件 freeze/unfreeze、学习率、dtype、activation checkpoint、LoRA target 和 layer-wise policy。
- 模型能力声明：支持哪些 modality、content block、objective、attention 实现、packing 和并行策略。
- 训练前静态校验，不能依靠运行到第一个 batch 才发现模型不支持 video 或 bbox。

### 3.3 效率与分布式

- BF16/FP16、TF32、FlashAttention、gradient accumulation/checkpointing。
- length/pixel/frame-aware batching；packing 与 padding-free；不同 pack 之间 attention 隔离。
- DDP 与 FSDP2 为基础；DeepSpeed 作为兼容路径；TP/PP/CP/SP/EP/Megatron 作为规模化后端。
- 对多模态模型做视觉 token 与文本 token 的联合 cost estimation，不能只按 `input_ids` 长度组 batch。
- 明确处理 context parallel 下的 media shard、position id、loss mask 和负载均衡。

### 3.4 可恢复、可验证、可交付

- checkpoint 必须包含 model、optimizer、scheduler、scaler、RNG、sampler、mixer、packer、dataloader 和 recipe state。
- dataset manifest/fingerprint、代码版本、完整 resolved config、环境和 checkpoint lineage。
- 恢复后下一批样本、loss mask 和 optimizer step 可复现。
- 训练中 generation/eval hook、task metrics、吞吐和 token/pixel 利用率。
- 保存 sharded training checkpoint，同时可导出 Hugging Face `save_pretrained` 格式。

NeMo AutoModel 默认保存 dataloader、RNG 和 step scheduler 状态，并基于 PyTorch Distributed Checkpoint 提供 FSDP2 checkpoint、异步保存、重分片与 HF 导出，是这部分最完整的开源参考。[NeMo checkpointing](https://docs.nvidia.com/nemo/automodel/latest/development/checkpointing)

## 4. 开源框架对比

评分含义：`强` 为官方一等能力；`中` 为可用但需要适配或覆盖有限；`弱` 为主要依靠用户自行实现。评分用于选架构，不等同于项目质量排名。

| 项目 | 原始多模态数据自由度 | CPT/SFT | Preference | Online RL | 分布式/长上下文 | 新模型接入成本 | TrainOmni 定位 |
|---|---:|---:|---:|---:|---:|---:|---|
| ms-swift | 强 | 强 | 强 | 强 | 强 | 中-高 | 能力 oracle、兼容性和回归基准 |
| TRL | 中 | 强 | 强 | 强 | 中 | 中 | post-training 算法层/可选 backend |
| LLaMA-Factory | 中 | 强 | 强 | 中 | 中 | 中 | 易用性、CLI/UI 和 recipe UX 参考 |
| XTuner V1 | 中 | 强 | 规划中 | 中 | 强 | 中-高 | 超大 MoE/长序列后端候选 |
| VeOmni | 强 | 强 | 中 | 中 | 强 | 中 | VLM/Omni 规模化与未来 Ascend 首选 engine 候选 |
| NeMo AutoModel | 强 | 强 | 中 | 弱-中 | 强 | 中 | NVIDIA 规模化 HF/FSDP2 backend 备选 |
| Molmo2 | 强 | 强 | 项目特定 | 项目特定 | 强 | 高 | 数据/packing/collator 设计参考 |
| veRL / EasyR1 | 中 | 弱 | 中 | 强 | 强 | 中 | 在线 RL backend，不承担通用 CPT/SFT |

### 4.1 ms-swift

优点：

- 官方仓库当前覆盖数百种 LLM/MLLM，CPT、SFT、DPO、KTO、GKD、GRPO family、量化、评测和部署。
- 支持 image/video/audio、grounding、agent、multimodal packing、padding-free、DeepSpeed、FSDP/FSDP2 与 Megatron。
- `vit/aligner/llm` 可以独立控制，能力面最接近“一站式 SOTA 框架”。

限制：

- 新模型注册包含 `ModelMeta`、模型加载函数、`model_arch` 和 `TemplateMeta`；多模态模板经常需要自定义 `_encode`、`_post_encode`、`_data_collator`。
- 数据统一格式仍使用 `<image>`、`<bbox>` 等占位符和并列 media 数组，语义、序号、模型模板较易耦合。
- 一个项目同时承担训练、RL、推理、量化、Web UI 和多个模型 hub，作为二次开发底座会引入较大依赖面。

证据：[ms-swift repository](https://github.com/modelscope/ms-swift)；[custom model](https://swift.readthedocs.io/en/v3.11/Customization/Custom-model.html)；[custom dataset](https://swift.readthedocs.io/en/v3.10/Customization/Custom-dataset.html)；[arguments/packing](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)

结论：不 fork；建立小规模 cross-check recipe，验证 TrainOmni 编码和 loss 与其结果的一致性。

### 4.2 TRL

优点：

- Hugging Face 原生，Trainer 接口薄，SFT/DPO/GRPO/KTO/Distillation 等 post-training 算法更新快。
- SFT VLM 支持 `image`/`images`、预编码 `input_ids`/`labels`、自定义 collator、packing 和 padding-free。
- GRPO 支持 VLM、vLLM colocate/server、reward function、tool/environment 和多模态 tool response。

限制：

- 各 Trainer 有自己的 dataset shape；官方也要求用户先转换为相应格式。
- 不负责 ViT + LLM 组装、原始多模态 annotation、全流程 exact-resume data state 或大规模预训练课程。
- “模型能被 Transformers/processor 接受”不等于其所有 VLM 结构都经过测试。

证据：[TRL repository](https://github.com/huggingface/trl)；[dataset formats](https://huggingface.co/docs/trl/en/dataset_formats)；[SFTTrainer](https://huggingface.co/docs/trl/main/sft_trainer)；[GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)

结论：采用其算法，不采用其列格式作为 canonical contract。

### 4.3 LLaMA-Factory

优点：

- 覆盖继续预训练、多模态 SFT、RM、PPO、DPO/KTO/ORPO，支持 full/freeze/LoRA/QLoRA。
- CLI、Web UI、数据集注册和大量模板让入门体验很好。
- Apache-2.0，生态和用户量大。

限制：

- 模型/数据扩展仍围绕内建模板和 dataset info 体系；适合常规模型微调，不是最自由的模型研发抽象。
- 在线 RL、精确数据恢复和多后端能力不是其最突出的设计中心。

证据：[LLaMA-Factory repository](https://github.com/hiyouga/LlamaFactory)

结论：借鉴 CLI/recipe UX 和数据导入器，不作为核心运行时。

### 4.4 XTuner V1

优点：

- 面向 ultra-large MoE 和长序列，重点优化 FSDP、Ulysses、FP8 和 GPU/NPU。
- 已实现 multimodal pre-training、multimodal SFT 与 GRPO。

限制：

- MPO、DAPO、multi-turn agentic RL 仍列在 roadmap。
- 设计中心是大规模 MoE，不是小型可组合 ViT + 1B LLM 的低摩擦研究循环。

证据：[XTuner repository](https://github.com/InternLM/xtuner)

结论：保留为 P2 后端与性能实现参考。

### 4.5 NeMo AutoModel

优点：

- Hugging Face checkpoint/model 兼容，PyTorch-native FSDP2/DTensor，并支持 TP、PP、CP、FP8、LoRA/QLoRA。
- 配置化自定义 Dataset、typed VLM loader、多数据源 MetaDataset、model-specific collator。
- DCP checkpoint 覆盖 optimizer、dataloader、RNG、scheduler，可异步保存并重分片/导出 HF。

限制：

- NVIDIA/CUDA 优化是重心；低成本单卡开发的安装和心智负担高于纯 HF stack。
- 当前 VLM collator 仍有较多 model-family-specific 函数，TrainOmni 仍需要自己的 plugin boundary。

证据：[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)；[dataset overview](https://docs.nvidia.com/nemo/automodel/datasets/overview)；[VLM loader](https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/datasets/vlm/loader)；[checkpointing](https://docs.nvidia.com/nemo/automodel/latest/development/checkpointing)

结论：P1 首选规模化 backend；v0.1 先让核心 data/model contract 不依赖 NeMo。

### 4.6 VeOmni

优点：

- 面向 VLM/Omni 预训练和后训练，而非只做现成模型微调；采用 model-centric distributed recipe。
- Torch-native FSDP2、Ulysses SP、EP/MoE、动态 batching、DCP，并公开支持 GPU、ROCm 和 Ascend。
- 已覆盖 Qwen VL/Omni、音频处理以及 DiT/Wan/LTX 等不同模型族，证明其执行抽象不只围绕文本 causal LM。

限制：

- 当前仍处于 v0.x 快速演进，trainer、parallel abstraction 和 Transformers 版本发生过 breaking changes，必须固定 release/commit。
- 自定义 ViT+connector+LLM 仍需模型 build/parallel plan；VeOmni 的数据、config 和 trainer ABI 不应成为 TrainOmni 公共协议。
- 现有功能覆盖不等于目标组合模型、exact data resume 或 Ascend 多机已经通过本项目 conformance。

证据：[VeOmni repository](https://github.com/ByteDance-Seed/VeOmni)；[paper](https://arxiv.org/abs/2508.02317)；[Ascend installation](https://veomni.readthedocs.io/en/latest/get_started/installation/install_ascend_x86.html)

结论：提升为 P1 首选 scale/Ascend engine 候选；保留 TrainOmni torch engine 作为本地 correctness oracle，通过窄 adapter 集成，不 fork。

### 4.7 Molmo2

优点：

- 原始数据保持最少处理，DataFormatter 生成带 media 的消息，MultimodalPreprocessor 负责 token/weight，之后 packing，最后 model-specific collator。
- message tree 可复用同一 media 上的多个 annotation；subsegment id 隔离 pack 内 cross-attention。
- packer state 被纳入 checkpoint，context parallel 对 ViT 工作量做 media-aware shard。

限制：

- 是单一模型家族的研究代码，不是通用训练产品；直接抽取会带入项目特定数据类和模型假设。

证据：[Molmo2 repository](https://github.com/allenai/molmo2)；[context parallel](https://github.com/allenai/molmo2/blob/main/docs/context_parallel.md)

结论：借设计，不复制项目结构。

## 5. 关键架构判断

### 5.1 Canonical data 不能等于 tokenizer 输入

`<image>`、Qwen bbox token、processor tensor 和 `input_ids` 都是派生物。canonical sample 保存“这里引用哪个 media、这个框在哪个坐标系、哪个内容参与 loss”，Model Family Plugin 才负责序列化。

### 5.2 Collator 是模型插件的一部分，但 Packer 不是

不同 VLM 的 `pixel_values`、grid、frame、position id 差异很大，通用 collator 只能提供协议。packing 的顺序、预算、segment id 与 checkpoint state 应由框架层管理，再把 pack plan 交给模型 collator。

### 5.3 Objective 与 Engine 分离

SFT、DPO、GRPO 描述的是目标函数和所需 batch/rollout；FSDP2、NeMo、veRL 描述的是执行方式。Recipe 选择 Objective 和 Engine，二者不能写死为一套 Trainer。

### 5.4 “精确恢复”从第一版就设计

只保存 model/optimizer 不够。mixture RNG、sampler、流式 shard、packer buffer、数据增强 RNG 和当前 microstep 都会改变后续 token。接口若不预留 `state_dict/load_state_dict`，以后很难补齐。

## 6. 暂不确定项

- “Qwen3.5-ViT”具体指哪一个视觉 encoder/merger checkpoint，以及从 Qwen3.5 原生 VLM 拆出还是独立 vision tower。
- “MiniCPM5-1B”作为语言模型时的 Transformers class、chat template、RoPE/attention、embedding resize 和权重许可约束。
- projector 类型、视觉 token 数、是否复用 Qwen merger、目标 context length 和首轮算力预算。
- L0 阶段只做连接器对齐，还是允许 ViT top layers/LLM embedding 联合训练。

这些属于 Research 任务应给出的模型路线输入；框架设计通过能力声明和 component policy 保持中立。

## 7. 下一步

1. 实现 canonical schema validator 和 3 个最小样例（SFT、preference、prompt-only）。
2. 定义 Model Family Plugin、Recipe、Engine Backend 和 stateful data component Protocol。
3. 用本地目标 checkpoints 做只加载不训练的 capability probe。
4. 先完成单进程 2-step smoke，再做 DDP/FSDP2 与 exact-resume 测试。
5. 建立 ms-swift 编码/loss cross-check，避免“自研接口正确、语义却偏了”。
