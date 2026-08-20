# 推荐论文与证据地图（历史窄口径版）

> 归档说明：本文围绕 connector alignment 展开，已由主目录中的完整训练方法和新版证据地图取代。

> 只把论文、官方模型卡和官方实现当作技术事实来源。博客用于理解工程背景；本地 checkpoint 检查单独标记。访问日期：2026-08-19。

## 1. 建议阅读顺序

### 第一组：先建立 L0 的共同语言

1. [LLaVA 1.5 — Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)  
   看什么：冻结 CLIP 与 LLM、只训练 `mlp2x_gelu` 的 feature alignment；558K 数据、1 epoch、LR 1e-3 的经典基线。  
   对本项目的意义：直接给出最小 connector-only 参照，但其 LLM 更大，不能假设 1B MiniCPM 有相同上限。  
   工程交叉核验：[官方 pretrain script](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh)。

2. [TinyLLaVA: A Framework of Small-scale Large Multimodal Models](https://arxiv.org/abs/2402.14289)  
   看什么：小 LLM + vision encoder + two-layer GELU MLP；feature alignment 仍采用双冻结、connector-only。  
   最重要的提醒：小模型只训练 connector 可能对齐不足；部分解冻能带来明显收益，MLP 在该设置下优于 resampler。  
   对本项目的意义：它是 1B 级目标比原始 LLaVA 更相关的证据。

3. [NVIDIA Nemotron Nano V2 VL](https://arxiv.org/abs/2511.03929)  
   看什么：明确命名的 Stage 0；冻结 vision/LLM，只训 MLP，使用约 2.2M 多样本，任务含 caption、VQA、grounding、OCR/doc。  
   超参锚点：Stage 0 约 LR 2e-4、batch 1024、weight decay 0.01、warmup 0.1。  
   对本项目的意义：说明现代工业路线仍保留 adapter-only stage，但数据任务比 LLaVA caption-only 更丰富。

4. [Seed1.5-VL Technical Report](https://arxiv.org/abs/2505.07062)  
   看什么：Stage 0 同样冻结 vision/LLM 只训练 MLP；报告中该阶段数据和 token 规模远大于单卡项目。  
   对本项目的意义：支持“L0 是有价值的初始化步骤”，但也说明不能照抄大厂 token 预算。

### 第二组：理解 connector-only 为什么会触顶

5. [VILA: On Pre-training for Visual Language Models](https://arxiv.org/abs/2312.07533)  
   看什么：冻结 LLM 能有一定 zero-shot，但会损伤 in-context learning；交错图文数据和 text-only replay 的作用。  
   对本项目的意义：为 L0.5 解冻/LoRA 和 L1 文本 replay 提供依据。

6. [TinyAlign: Scaling Down Large Vision-Language Models via Memory-Aligned Knowledge Distillation](https://arxiv.org/abs/2505.12884)  
   看什么：轻量 LLM 下冻结双 backbone、只训 connector 会形成 alignment bottleneck；论文用额外记忆/检索机制缓解。  
   对本项目的意义：不是要求采用 TinyAlign，而是给 connector-only 的能力边界提供证据。

7. [MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611)  
   看什么：image encoder、分辨率、视觉 token 数常比 connector 架构本身更重要；caption/interleaved/text-only 数据混合很关键。  
   对本项目的意义：消融优先级不能只围绕“换一个更复杂 projector”。

### 第三组：理解不同桥接范式

8. [BLIP-2](https://arxiv.org/abs/2301.12597)  
   看什么：冻结 vision 和 LLM，使用 Q-Former 分两阶段做视觉语言表征与生成对齐。  
   判断：比 MLP 更强也更复杂，适合作为 L0 后备路线，不适合作为首个可复现基线。

9. [PaliGemma](https://arxiv.org/abs/2407.07726)  
   看什么：SigLIP + Gemma 的完整 transfer/pretrain 路线。  
   判断：说明高质量 VLM 往往需要联合训练，不应拿 connector-only 的资源预期衡量最终效果。

10. [MobileVLM](https://arxiv.org/abs/2312.16886)  
    看什么：面向 1.4B/2.7B LLM 的高效视觉语言模型与轻量 projector。  
    对本项目的意义：后续若 MLP 已证实为瓶颈，可参考其高效 connector 思路。

### 第四组：数据质量与小模型工程

11. [Molmo and PixMo](https://arxiv.org/abs/2409.17146)  
    看什么：细致人类 caption、数据质量优先、少于百万图文对也能形成有竞争力的开放 VLM；其正式训练会更新全部参数。  
    配套：[官方 Molmo/PixMo 介绍](https://allenai.org/blog/molmo)、[PixMo-Cap dataset card](https://huggingface.co/datasets/allenai/pixmo-cap)。

12. [SmolVLM 官方工程说明](https://huggingface.co/blog/smolvlm)  
    看什么：小模型的视觉 token 压缩、文档数据、训练 checkpoint 的多指标选择。  
    对本项目的意义：单卡项目要控制视觉 token 数，并按下游评测而非 train loss 选模型。

13. [nanoVLM](https://github.com/huggingface/nanoVLM)  
    看什么：最小纯 PyTorch VLM 的 processor、pixel shuffle、projector、embedding 拼接和训练循环。  
    对本项目的意义：非常适合读代码和写单测，但不是成熟分布式训练框架的替代品。

## 2. 证据账本

| 结论 | 证据 | 类型 | 置信度 | 如何用于本项目 |
|---|---|---|---|---|
| L0 可定义为冻结 vision+LLM、只训 connector | LLaVA、TinyLLaVA、Nemotron、Seed1.5-VL | 多篇一手论文/官方脚本 | 高 | 采用为严格 L0 边界 |
| MLP2x-GELU 是合理最小基线 | LLaVA 官方脚本、TinyLLaVA | 一手实现/论文 | 高 | 默认 connector |
| 小 LLM 上 connector-only 可能对齐不足 | TinyLLaVA、TinyAlign | 一手论文 | 中高 | 预留 L0.5，设置停止条件 |
| connector 架构未必是最大影响因子 | MM1 | 一手论文 | 中高 | 优先做分辨率/数据/encoder 消融 |
| text-only replay 有助于保留语言能力 | VILA、MM1 | 一手论文 | 中高 | L0.5/L1 混入 5–10% 起测 |
| 当前 Qwen 资产不是独立 ViT | 本地 config/header + Qwen 官方卡 | 本地/官方 | 高 | 配置写 source checkpoint/module path |
| Qwen visual 输出 1024，MiniCPM hidden 1536 | 本地 config/header | 本地事实 | 高 | 必须新增 projector |
| 视觉 token 应在 MiniCPM 使用连续 1D position | 两侧架构检查 | 工程推断 | 高 | framework 接口契约与单测 |
| MiniCPM-SFT 比 final 更适合主训练 | MiniCPM 官方阶段说明 + 训练风险判断 | 官方事实+推断 | 中 | 做 SFT vs final 小对照 |
| PixMo-Cap 优先于盲用 LLaVA558K | PixMo/Molmo 与两者数据卡 | 一手资料+工程判断 | 中 | pilot 优先质量与可审计性 |

## 3. 需要谨慎表述的内容

- 大模型论文中的 batch、token 数和学习率是锚点，不是当前 16 GB 单卡的可直接复刻配置。
- “connector-only 有效”不等于“connector-only 足够”。论文通常把它当初始化阶段，后续仍会解冻更多模块。
- “开放数据集”不自动等于“所有图片均可商用”。必须区分 annotation/data-table license 与图片原始权利。
- 原 Qwen3.5-0.8B 可作为 teacher 是研究建议，不是上述论文已经验证过的本项目结论。
- shared-token Procrustes/ridge 初始化是本项目可做的低成本实验，不应包装成既定行业标准。

## 4. 数据与模型官方入口

- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)
- [LLaVA-Pretrain 558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
- [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap)
- [PixMo-CapQA](https://huggingface.co/datasets/allenai/pixmo-cap-qa)
