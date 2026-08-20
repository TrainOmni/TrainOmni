# 论文证据地图与推荐阅读顺序

## 1. 如何读这批论文

不要按发布时间从头堆论文。建议围绕六个问题阅读：

1. 视觉信息如何进入 LLM？
2. 什么决定视觉 token 和分辨率效率？
3. 全参数联合训练怎样组织阶段和数据？
4. 小模型与大模型的规律哪里不同？
5. SFT、蒸馏、偏好和 RL 各解决什么？
6. 榜单提升是否真来自视觉能力？

以下全部优先列原论文或官方材料。

## 2. 第一层：建立架构与训练共同语言

### 2.1 BLIP-2

论文：[BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)

- **回答的问题**：冻结视觉编码器和 LLM 时，如何用轻量 Q-Former 对齐两个预训练模态。
- **应学到**：query bottleneck、两阶段对齐、低成本 bootstrapping 的经典设计。
- **不要误推**：BLIP-2 证明 connector-only 可建立能力，不证明它能达到今天全参数联合训练的上限。

### 2.2 MM1

论文：[MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611)

- **回答的问题**：在受控设置下，架构、视觉编码器、分辨率、token 和数据混合谁更重要。
- **应学到**：视觉编码质量、分辨率、视觉 token 数和 caption/interleaved/text 数据组合的影响通常大于微调 connector 细节。
- **不要误推**：其消融条件和模型规模并非所有 VLM 的全局最优。

### 2.3 PaliGemma

论文：[PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726)

- **回答的问题**：较干净的视觉-语言预训练模型怎样作为多任务 transfer base。
- **应学到**：prefix-LM/生成式多任务配方、分辨率阶段和广泛 transfer 评测。
- **不要误推**：transfer base 的能力与完成聊天、推理、偏好后训练后的产品 VLM 不等价。

### 2.4 LLaVA-OneVision

论文：[LLaVA-OneVision](https://arxiv.org/abs/2408.03326)，[官方训练说明](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md)

- **回答的问题**：如何用统一表示覆盖单图、多图和视频。
- **应学到**：从已有图像能力迁移到多图/视频、任务和数据整合的开放路线。
- **不要误推**：统一接口并不自动解决长视频 token、时间建模和数据质量。

### 2.5 Chameleon

论文：[Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818)

- **回答的问题**：图像离散 token 与文本 token 能否在单一 early-fusion 模型中统一理解和生成。
- **应学到**：离散统一建模的潜力，以及稳定训练/模态平衡的复杂性。
- **不要误推**：若目标只是视觉理解，不必为“统一生成”承担全部 tokenizer 与训练成本。

## 3. 第二层：理解数据、分辨率和小模型规律

### 3.1 Idefics3

论文：[Idefics3](https://arxiv.org/abs/2408.12637)

- **回答的问题**：原生分辨率、文档合成数据、数据污染和高效微调怎样共同影响 8B VLM。
- **应学到**：Docmatix 的文档数据生产方式；动态分辨率；对 MathVista 的污染审计；数据与评测细节的重要性。
- **不要误推**：LoRA/DoRA 的成功不代表全参数更新没有上限收益，论文自身也承认这一点。

### 3.2 SmolVLM

论文：[SmolVLM: Redefining small and efficient multimodal models](https://arxiv.org/abs/2504.05299)

- **回答的问题**：小模型如何选择 ViT、视觉 token 压缩、数据混合和推理数据。
- **应学到**：大 ViT/更多 token/更多 CoT 对小 LLM 并非单调更好；模型容量决定最佳视觉带宽。
- **不要误推**：小模型节省成本的配方不能直接扩展成大模型的最佳策略，反之亦然。

### 3.3 Molmo/PixMo

论文：[Molmo and PixMo](https://arxiv.org/abs/2409.17146)

- **回答的问题**：高质量人工详细 caption、pointing 和开放数据能否替代一部分不透明海量语料。
- **应学到**：标注信息密度、视觉证据与数据可开放性。
- **不要误推**：高质量小数据能减少噪声，不代表 web-scale 覆盖完全不需要。

### 3.4 DataComp 与 Interleaved 语料

- [DataComp](https://arxiv.org/abs/2304.14108)：固定训练框架比较数据策展；重点读 filtering 和 fixed-compute evaluation。
- [OBELICS](https://arxiv.org/abs/2306.16527)：文档级 interleaved 图文构建。
- [MMC4](https://arxiv.org/abs/2304.06939)：网页多图与文本对齐。
- [MINT-1T](https://arxiv.org/abs/2406.11271)：trillion-token 级多模态 interleaved 规模化。

共同启示：pair caption、interleaved 文档和高质量专项监督承担不同作用，不能把它们合并为一个“多模态 token 数”。

## 4. 第三层：看完整前沿训练流水线

### 4.1 Kimi-VL

论文：[Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491)

- **回答的问题**：从视觉侧预训练、联合训练、cooldown、长上下文到 SFT/RL 的完整开放路线如何组织。
- **应学到**：原生分辨率 ViT；逐阶段的大规模联合训练；保留文本分布；高质量 cooldown；CoT 冷启动和在线 RL。
- **方法意义**：它非常接近本项目“完整 L0”所指的生命周期，而不只是一个架构论文。

### 4.2 Qwen3-VL

论文：[Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)

- **回答的问题**：大规模联合预训练、超长上下文、视觉多层注入、强弱蒸馏和双类型 RL 如何组合。
- **应学到**：短 connector 对齐之后仍需要多阶段全参数训练；视觉 token/长短样本 loss 平衡；off-policy + on-policy 蒸馏；reasoning RL 与 general RL 分工。
- **方法意义**：提供从基础能力到通用后训练的高覆盖参考，但其规模与数据不可直接复制。

### 4.3 InternVL3.5

论文：[InternVL3.5: Advancing Open-Source Multimodal Models](https://arxiv.org/abs/2508.18265)

- **回答的问题**：视觉 token 动态压缩、一致性训练、SFT 和级联 RL 如何形成效率/能力平衡。
- **应学到**：ViR/ViCO；预训练和 SFT 的不同数据混合；MPO 离线偏好后接 GSPO 在线 RL；对中等难度 query 的筛选。
- **方法意义**：特别适合学习“效率对齐”和“离线到在线”的后训练结构。

### 4.4 GLM-4.1V / GLM-4.5V

论文：[GLM-4.1V-Thinking and GLM-4.5V](https://arxiv.org/abs/2507.01006)

- **回答的问题**：视觉推理模型如何组合预训练、thinking SFT 和强化学习课程。
- **应学到**：预训练决定上限，RL 负责稳定激活；推理任务的课程和不同 reward 类型。
- **不要误推**：RL 不能替代低层视觉表征和数据覆盖。

### 4.5 Seed1.5-VL

论文：[Seed1.5-VL Technical Report](https://arxiv.org/abs/2505.07062)

- **回答的问题**：大规模多模态基础模型的架构、数据、预训练和后训练怎样协同。
- **应学到**：完整技术报告应披露的模块；从通用理解到视觉推理/Agent 能力的阶段关系。
- **不要误推**：未公开数据和未充分消融部分只能作为工程先例，不是可复制因果结论。

## 5. 第四层：追踪 2026 原生多模态与长视频方向

### 5.1 Kimi K2.5

论文：[Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276)

- **回答的问题**：视觉是否应该从基础预训练早期进入；如何做联合图文 RL 和视觉 Agent。
- **应学到**：约 15T 混合 token 的原生联合训练；固定预算下早期中等视觉比例的优势；可验证视觉 reward、开放任务 reward model、thinking 切换与工具能力。
- **最重要限制**：证据来自超大模型与超大 token 规模。对小模型，最合理的行动是验证这一趋势，而不是直接宣布同一比例最优。

### 5.2 Molmo2

论文：[Molmo2: Open Vision-Language Models for Images and Video](https://arxiv.org/abs/2601.10611)，[官方仓库](https://github.com/allenai/molmo2)

- **回答的问题**：图像 caption/pointing、图像+视频+多图 SFT、长上下文 SFT 如何分阶段；怎样提高 packing 效率。
- **应学到**：全部参数的分组学习率；任务权重和输出长度缩放；时间戳、视频 pooling；长上下文对短任务的权衡；message tree/packing 的系统收益。
- **方法意义**：在视频和工程透明度上非常值得逐节复现式阅读。

## 6. 第五层：推理与视觉 Agent

### 6.1 R1-OneVision 与 Vision-R1

- [R1-OneVision](https://arxiv.org/abs/2503.10615)
- [Vision-R1](https://arxiv.org/abs/2503.18013)

用它们理解“长 CoT/强化学习范式迁移到视觉”的早期方法，包括冷启动数据、可验证问题和 GRPO 类训练。阅读时重点检查：视觉是否真参与、教师数据质量、推理长度成本，以及提升来自 SFT 还是 RL。

### 6.2 DeepEyes

论文：[DeepEyes: Incentivizing Thinking with Images via Reinforcement Learning](https://arxiv.org/abs/2505.14362)

用它理解模型在推理时主动 crop/zoom、重新观察和调用工具。它代表“固定一次图像编码”之外的视觉推理路线。评估时必须把工具次数、额外视觉 token 和环境失败计入成本。

## 7. 第六层：评测可信度

### 7.1 统一评测工具

- [VLMEvalKit](https://arxiv.org/abs/2407.11691)
- [lmms-eval](https://arxiv.org/abs/2407.12772)

用来统一 dataset adapter、prompt 和答案提取，但运行器不能自动消除数据污染、输入预算差异和模型专属 prompt 调优。

### 7.2 视觉依赖与难度

- [MMStar](https://arxiv.org/abs/2403.20330)：减少可被纯文本先验回答的样本。
- [MMMU-Pro](https://arxiv.org/abs/2409.02813)：提高专业多模态推理难度，并抑制 shortcut。

两者应与无图/错图/遮挡对照一起使用，不能只报最终分数。

## 8. 最短必读清单

如果只读十篇，建议顺序：

1. MM1：先学会区分真正重要的设计变量；
2. Idefics3：分辨率、文档数据和污染；
3. SmolVLM：小模型规律；
4. Molmo/PixMo：高质量视觉数据；
5. Kimi-VL：完整训练生命周期；
6. Qwen3-VL：大规模联合预训练、蒸馏和 RL；
7. InternVL3.5：视觉 token 效率与级联 RL；
8. Kimi K2.5：原生早期联合和 Agentic 趋势；
9. Molmo2：视频、长上下文和训练系统；
10. MMStar + MMMU-Pro：识别“模型没看图也能答”的评测陷阱。

## 9. 证据账本：当前可形成的结论

| 判断 | 证据强度 | 主要来源 |
|---|---|---|
| Connector-only 适合热身，不是完整能力路线 | E2 | BLIP-2、Kimi-VL、Qwen3-VL |
| 视觉编码器/分辨率/token 数比 connector 微结构更关键 | E3（特定设置） | MM1、SmolVLM |
| Post-hoc 路线的前沿主阶段通常全参数联合训练 | E2 | Qwen3-VL、Kimi-VL、Molmo2 |
| 纯文本 replay 对抗语言遗忘重要，但比例无统一值 | E2 | Qwen3-VL、Kimi-VL、InternVL3.5、SmolVLM |
| 数据质量、专项监督和去污染不能靠规模替代 | E2/E3 | DataComp、Idefics3、PixMo |
| 长样本需要采样或 loss 归一化，避免主导梯度 | E2 | Qwen3-VL、InternVL3.5、Molmo2 |
| 离线偏好后接在线 RL 是稳定且节省在线算力的路线 | E2 | InternVL3.5、Qwen3-VL |
| 中等难度在线 query 更有学习信号 | E2 | InternVL3.5、Qwen3-VL、Kimi-VL |
| 小模型不能简单照搬大模型的 ViT/token/CoT 配方 | E3（小模型设置） | SmolVLM |
| 早期持续视觉联合训练可能优于后期集中注入 | E3（超大规模设置） | Kimi K2.5 |
| 公榜分数必须配视觉依赖与污染测试 | E2 | MMStar、MMMU-Pro、Idefics3 |

## 10. 仍缺乏公开答案的问题

- 同一数据和算力下，原生 early-joint 与强 post-hoc adaptation 的严格规模化对比；
- 1B、3B、7B 各档最优 ViT/LLM 参数比与视觉 token 带宽；
- 合成 caption/CoT 的教师能力、数量和多样性的最优组合；
- 多层视觉注入收益的等参数、等 FLOPs 因果消融；
- 多模态在线 RL 在开放任务上与人类偏好的长期一致性；
- 视觉 Agent 工具能力和 base perception 能力的公平拆分；
- 数据许可、隐私和教师生成内容在大规模可再发布项目中的完整治理方案。

这些应被列为研究假设，不应在模型选型前伪装成既定配方。

