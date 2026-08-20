# 架构、视觉编码与视觉 Token 方法

## 1. 架构不是 connector 选型题

完整 VLM 架构至少包含六个耦合决策：

```text
像素/帧
  → 分辨率与切片策略
  → 视觉编码器及其训练目标
  → token 压缩/重采样
  → 视觉-语言融合位置
  → LLM 与位置编码
  → 输出头、工具和结构化坐标
```

MM1 的受控研究把这件事说得很清楚：在其设置下，视觉编码器、输入分辨率、视觉 token 数和数据混合比 connector 结构本身更重要。因此不应把大量研究预算消耗在“线性层还是两层 MLP”上：[MM1](https://arxiv.org/abs/2403.09611)。

## 2. 四类融合范式

| 范式 | 做法 | 优点 | 主要风险 | 公开代表 |
|---|---|---|---|---|
| 连续 token 早融合 | 将 ViT 特征投影到 LLM hidden size，与文本 token 拼接，进入 decoder self-attention | 简单、可扩展、图文深层交互 | 上下文/注意力成本随视觉 token 快速上升 | Qwen、Kimi、InternVL、Molmo、LLaVA |
| Cross-attention | 文本流保持不变，在若干 LLM 层插入对视觉特征的 cross-attention | 可保留原 LLM 主干，视觉 KV 可独立管理 | 改造层多；训练和 checkpoint 兼容复杂 | Flamingo、Llama 3.2 Vision |
| Query/Resampler | 用少量可学习 query 压缩任意数量图像特征，再接入 LLM | 固定 token 预算、适合多图 | 固定瓶颈可能丢小字和密集位置细节 | BLIP-2、Idefics2、部分 MiniCPM-V |
| 离散统一 token | 图像先离散为 code，与文本在同一词表/序列建模 | 理论上统一理解和生成 | tokenizer 重构误差、序列很长、稳定性与数据要求高 | Chameleon |

参考：[BLIP-2](https://arxiv.org/abs/2301.12597)、[Chameleon](https://arxiv.org/abs/2405.09818)、[Llama 3.2 Vision model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md)。

**跨项目共同模式**：若目标是纯理解型、decoder-only、开放可复现 VLM，连续视觉 token 早融合是当前最稳妥的主路线；但这个结论不是说 cross-attention 或离散 token 在任何预算下都较差。

## 3. 视觉编码器决定“看见什么”

### 3.1 不能只拿通用对比学习 ViT 冻结到底

通用图文对比预训练擅长全局语义，但 OCR、细粒度属性、坐标、计数和视频时序需要更密集的空间信息。前沿路线常继续训练甚至专门预训练 ViT：

- Kimi-VL 使用原生分辨率 MoonViT，并对视觉侧进行独立大规模训练后再联合训练：[Kimi-VL](https://arxiv.org/abs/2504.07491)。
- Qwen3-VL 继续训练视觉编码器并在联合阶段更新全部参数：[Qwen3-VL](https://arxiv.org/abs/2511.21631)。
- Kimi K2.5 将视觉编码扩展到时空建模，而不是仅把逐帧 CLIP 特征交给 LLM：[Kimi K2.5](https://arxiv.org/abs/2602.02276)。

### 3.2 视觉训练目标是能力配方

可组合的目标包括：

- 图像/视频 caption 的自回归 CE：连接视觉细节与自然语言；
- 图文对比：增强全局语义与检索；
- masked image modeling：补足局部纹理和密集表征；
- OCR/文档布局：文字、阅读顺序、表格和公式；
- grounding/pointing：框、点、区域和指代；
- 时间与动作：帧顺序、时间戳、事件区间；
- 3D/空间关系：相对位置、深度、视角变换。

**研究建议**：不要用一个“ViT 总分”选视觉骨干。应先用能力契约给视觉侧做专项 probing，然后决定继续训练目标和数据比例。

## 4. 原生分辨率、切片与位置

### 4.1 三种分辨率策略

| 策略 | 描述 | 适用性 |
|---|---|---|
| 固定 resize | 全部缩放到固定方形 | 简单；小字、长图和纵横比损失明显 |
| 动态切片 + overview | 高分辨率图分块，同时保留缩略全图 | 实用成熟；块边界与 token 数膨胀需处理 |
| 原生可变分辨率/packing | 保持纵横比，把不同图像 patch 打包进 batch | 长期更优雅；位置编码、mask 和 kernel 更复杂 |

[Pixtral](https://arxiv.org/abs/2410.07073)、[Idefics3](https://arxiv.org/abs/2408.12637) 和 [SmolVLM](https://arxiv.org/abs/2504.05299) 展示了动态/原生分辨率与 token 压缩的不同工程取舍。

### 4.2 位置不能只靠一维序列索引

至少要保留：

- 图像内二维坐标；
- 多个 tile 的全局坐标与原图尺寸；
- 多图的 image id 和先后关系；
- 视频帧时间戳、帧率或相对时间；
- GUI/grounding 输出坐标的归一化约定。

否则，LLM 虽能“看到”局部 token，却很难稳定恢复它们在原图或时间轴中的关系。

## 5. 视觉 Token 预算是核心缩放变量

粗略地，单图视觉 token 数满足：

```text
N_visual ≈ ceil(H / patch_h) × ceil(W / patch_w) / compression_ratio
```

对 decoder self-attention，视觉 token 会同时增加 prefill 计算、KV cache 和长文本竞争。高分辨率不等于有效高分辨率；如果压缩太强，细节在进入 LLM 前已经丢失。

### 5.1 主要压缩方法

| 方法 | 机制 | 强项 | 风险 |
|---|---|---|---|
| Pixel shuffle / spatial merge | 邻域 patch 在通道维合并 | 保持规则空间网格，OCR 友好 | 压缩过强会伤小物体/定位 |
| Pooling | 平均/注意力池化 | 快、稳定 | 细节损失不可逆 |
| Learned resampler | 固定 query 从全部 patch 读取信息 | token 上限固定 | 复杂页面可能超过瓶颈容量 |
| Token pruning/router | 按图像或层动态保留 token | 可形成精度-成本自适应 | 训练不稳，可能误删关键小区域 |
| 多层特征注入 | 不增加输入长度，将浅/中层 ViT 特征注入 LLM 中间层 | 补低级与高分辨率细节 | 融合层与尺度对齐复杂 |

Qwen3-VL 的 DeepStack 属于多层视觉特征注入；InternVL3.5 的视觉 token 路由与一致性训练探索了按样本动态压缩：[Qwen3-VL](https://arxiv.org/abs/2511.21631)、[InternVL3.5](https://arxiv.org/abs/2508.18265)。

SmolVLM 的实验还提醒：小 LLM 不一定能有效消费超大 ViT 或过多视觉 token，最优压缩率随语言模型容量和任务改变，不能照搬大模型：[SmolVLM](https://arxiv.org/abs/2504.05299)。

### 5.2 应报告曲线而非单点

同一 checkpoint 至少测 3–4 个视觉 token 预算，绘制：

- OCR/grounding/通用 VQA 对 token 数的增益；
- prefill 时延、峰值显存和吞吐；
- 高分辨率与多图/视频之间的预算竞争。

这样才能判断模型是在真正提高编码效率，还是仅靠更多 token 堆分。

## 6. 多层视觉特征

最后一层 ViT 更偏全局语义，浅层和中层保留局部纹理、边缘和空间细节。常见方案：

- 选择多个固定层并拼接/加权；
- 将不同层特征映射到不同 LLM 深度；
- 对 OCR、grounding 使用更浅特征，对语义问答使用更深特征；
- 通过辅助 loss 让被选特征具备可解码的文字/位置能力。

**未决问题**：多层融合的提升究竟来自额外参数、额外监督还是更完整表征，需要等计算量消融，而不是只比较最终总分。

## 7. 图像、视频与多图是否共享编码器

共享编码器有利于图像能力迁移到视频，也减少系统复杂度；但视频还需要：

- 采帧和镜头边界策略；
- 明确时间戳/相对时间 token；
- 帧内空间编码 + 帧间时间编码；
- temporal pooling/merge，避免长视频 token 爆炸；
- 快动作的高帧率路径和长视频的稀疏路径；
- 防止仅凭字幕或音频先验作答的反事实评测。

Molmo2 报告了图像、短视频、多图到长上下文的分阶段训练，并分析时间戳、视觉注意力和 pooling 对视频能力的作用：[Molmo2](https://arxiv.org/abs/2601.10611)。

**研究建议**：基础图像能力稳定后再渐进启用视频长度；不要从第一天就把超长视频和高分辨率文档混在一个 token 预算里竞争。

## 8. 输出表示

一个完整 VLM 不应只支持自由文本，还需定义：

- bounding box / point 的坐标 token 或归一化数字格式；
- 多区域引用及 region id；
- 时间段与帧索引；
- tool call schema；
- thinking 与 final answer 的分隔和可见性策略；
- 拒答、不确定性和证据引用格式。

结构化输出必须在预训练/SFT 数据、tokenizer、decode constraint 和 evaluator 中保持同一约定。

## 9. 架构研究的最小消融矩阵

模型绑定之前，至少保留以下问题作为正式消融：

1. 冻结 ViT vs 继续训练 ViT vs 全参数联合；
2. 最后一层特征 vs 多层特征；
3. 固定分辨率 vs 动态切片/原生分辨率；
4. 两到三个视觉 token 压缩率；
5. 拼接早融合 vs 一个受预算约束的替代融合 baseline；
6. 纯图文 vs 加 grounding/OCR 辅助目标；
7. 无纯文本 replay vs 不同比例 replay；
8. 图像-only curriculum vs 渐进加入多图/视频；
9. 固定视觉预算 vs 动态 token 路由；
10. 正确图/无图/错图的视觉依赖差值。

这些消融的优先级高于在若干 connector 深度和激活函数之间做大规模网格搜索。

