# 完整 L0 训练生命周期：P0–P10

## 1. 总体原则

完整 L0 不是“预训练 + SFT”两个盒子，而是从能力定义到数据回流的闭环。下面 P0–P10 是职责分解，不要求每段都单独保存 checkpoint；是否合并由规模、初始化方式和稳定性决定。

## 2. 阶段总表

| 阶段 | 目的 | 主要参数更新 | 主要产出 |
|---|---|---|---|
| P0 | 能力契约、评测、数据治理 | 无 | target/eval/data spec |
| P1 | 视觉编码器专项增强 | ViT，必要时视觉 heads | 能看清文字、位置、时序的 ViT |
| P2 | 可选模态对齐热身 | connector 为主 | 稳定初始化，不是能力终点 |
| P3 | 全参数联合多模态预训练 | ViT + connector + LLM | 基础多模态能力主体 |
| P4 | 高质量 cooldown/mid-training | 通常全参数 | 激活知识、推理与细粒度能力 |
| P5 | 高分辨率、长上下文、多图、视频扩展 | 通常全参数 | 长程和动态视觉能力 |
| P6 | 通用多模态 SFT | 通常全参数或资源受限 PEFT | 指令遵循与对话能力 |
| P7 | 推理冷启动与蒸馏 | LLM 为主，视觉侧视数据而定 | 稳定长/短思维模式 |
| P8 | 离线偏好优化 | 通常全参数 | 质量、风格、安全与稳定性 |
| P9 | 在线 RL / Agent RL | policy，必要时 value/reward | 可验证推理与工具闭环 |
| P10 | 效率对齐、交付门禁、数据飞轮 | 可继续训练/蒸馏/压缩 | Pareto 最终点与下一轮数据 |

## 3. P0：能力、评测和数据先行

### 输入

- 模型/算力档位；
- 目标场景和不做事项；
- 可用数据、许可和隐私边界；
- 推理 token/时延约束。

### 必做

- 冻结公开评测版本和 prompt；
- 建立私有、动态、反事实评测；
- 建立图像/文本联合去重；
- 定义日志、checkpoint、数据版本和训练恢复规范；
- 对候选数据做小规模 scaling/quality pilot。

### 门禁

评测不能与训练数据独立去重、推理预算不能公平对齐时，不进入 P1/P3。

## 4. P1：视觉专项预训练/继续训练

P1 的目的不是训练一个通用分类 ViT，而是补足最终 VLM 的视觉瓶颈。任务组合取决于能力契约：caption/contrastive 提供语义，OCR/layout 提供文字与结构，grounding/pointing 提供位置，视频目标提供时间。

### 建议训练信号

- 高质量详细 caption 与短 caption 混合；
- 图文对比和难负样本；
- 文档 OCR、阅读顺序、表格/公式结构；
- box/point/mask 到自然语言的双向 grounding；
- 帧序、时间段、动作和事件描述；
- 多分辨率、多纵横比和图像增强。

### 门禁

在不接 LLM 或接轻量 probing head 的条件下，视觉侧必须在目标 OCR/位置/视频 probing 上显示足够信息，否则 connector 无法恢复已经丢失的视觉细节。

## 5. P2：可选 connector alignment warm-up

只有在 ViT 与 LLM 来自独立预训练、embedding 空间差异大或直接联合训练不稳定时，才需要独立 P2。

典型做法是冻结两个骨干，用 caption/图文描述训练 projector、resampler 或 cross-attention adapter。目标是让 loss、图像 token 和梯度尺度进入合理区域。

### 何时应缩短或跳过

- 原生联合预训练模型；
- 联合训练已经稳定；
- P2 数据太窄，可能把视觉表征压到 caption 风格；
- P2 成为主要算力阶段，却仍冻结骨干。

**关键结论**：P2 通过只说明接口可学，不说明 VLM 已具备 OCR、推理、grounding 或视频能力。

## 6. P3：全参数联合多模态预训练

这是 post-hoc adaptation 路线中决定上限的核心阶段。

### 数据

- 图文对与详细 recaption；
- interleaved webpage/document；
- OCR、文档、图表、数学、知识问答；
- grounding、计数、空间；
- 纯文本 replay；
- 根据能力目标逐步加入多图、视频、GUI。

### 目标函数

主目标通常仍是因果 next-token prediction：

```text
L = Σ_d w_d · Normalize_d(Σ_t m_t · CE(y_t, p_t)) + Σ_k λ_k L_aux,k
```

其中 `d` 是数据/任务域，`w_d` 控制混合；`m_t` 决定哪些 token 参与 loss；`Normalize` 防止长样本支配；辅助目标可包括对比、grounding 或一致性损失。

### 参数更新

公开前沿共同模式是此阶段更新 ViT、connector 和 LLM，而不是一直冻结骨干。Qwen3-VL 披露在主要联合阶段训练全部参数；Kimi-VL 也在联合训练中更新完整模型：[Qwen3-VL](https://arxiv.org/abs/2511.21631)、[Kimi-VL](https://arxiv.org/abs/2504.07491)。

推荐采用参数组学习率而非“一刀切”：预训练成熟的 LLM/ViT 用较小 LR，新 connector/输出头可用较大 LR；具体比例必须由梯度、loss 和遗忘曲线确定，公开报告不存在跨架构通用常数。

### 防止文本能力退化

纯文本 replay、语言/代码/数学回归集和分域 loss 是三件不同的事，都需要。Qwen3-VL、Kimi-VL、InternVL3.5 均在联合训练中保留文本数据；但公开比例差异很大，说明比例必须由模型容量和目标数据确定，不能抄一个固定百分比。

## 7. P4：高质量 cooldown / mid-training

在大规模宽分布预训练之后，降低学习率并提高高质量、可核验、学术/合成数据占比，用更少 token 激活：

- 多步知识和 STEM 推理；
- 高精度 OCR/文档/图表；
- 复杂 grounding、空间和计数；
- 多图比较与更长描述；
- 接近最终交互格式、但仍保持预训练多样性的样本。

Kimi-VL 明确使用 cooldown 数据提高知识与推理质量：[Kimi-VL](https://arxiv.org/abs/2504.07491)。

### 风险

过度使用同一教师生成的问答会导致风格坍缩、答案模板化和 benchmark leakage。高质量不等于“所有数据都变成聊天 QA”。

## 8. P5：分辨率、上下文、多图和视频渐进扩展

推荐的 curriculum 是先建立稳定图像语义，再提高空间和序列难度：

1. 常规分辨率单图；
2. 高分辨率 OCR/文档与动态切片；
3. 多图交错与跨图比较；
4. 短视频、时间戳与密集动作；
5. 长视频/多页文档/超长上下文。

同时延长 context 的主要风险不是 OOM，而是数据分布和注意力行为改变。Molmo2 报告长上下文训练能提升长视频 QA，但也观察到对部分 caption 能力的权衡：[Molmo2](https://arxiv.org/abs/2601.10611)。因此每次扩长都要回归短图、caption 和文本能力。

## 9. P6：通用多模态 SFT

SFT 把基础知识转化为可控交互行为。核心不是样本数量最大，而是覆盖和格式一致：

- 开放式与简答 VQA；
- OCR/文档/图表；
- grounding/坐标/计数；
- STEM/知识；
- 多轮、多图和视频；
- GUI/tool schema；
- 拒答、安全、不确定性；
- 高质量纯文本指令，守住语言能力。

对对话样本通常只对 assistant response 计算生成 loss；system/user 和图像占位 token 用 mask 排除。若希望模型复述 OCR 或生成结构化视觉目标，则对应 token 需要明确纳入监督。

## 10. P7–P9：推理与偏好/强化学习

这三阶段详见 [05_posttraining_distillation_and_rl.md](./05_posttraining_distillation_and_rl.md)。简化关系：

```text
高质量推理示范/教师轨迹
  → P7 冷启动与蒸馏
      → P8 离线 chosen/rejected 偏好学习
          → P9 在线采样 + 可验证/模型奖励
```

离线阶段提供稳定起点，在线阶段发现模型自身分布上的错误。对于资源有限项目，P8 可能先于昂贵的 P9；但若目标包含 agentic 工具使用和真正的自我纠错，最终仍需要环境或在线策略数据。

## 11. P10：效率对齐和数据飞轮

最终 checkpoint 不是“准确率最高的那个”，而是满足能力和成本约束的 Pareto 点。P10 包括：

- 视觉 token 动态压缩与一致性训练；
- 推理长度控制；
- 小模型蒸馏、量化和 serving 校准；
- 红队、视觉幻觉和工具越权评测；
- 线上/人工错误聚类；
- 将失败样本转成下一轮预训练、SFT、偏好或 RL 数据；
- 保持私有 holdout，不让数据飞轮吞掉最终评测集。

## 12. 训练系统的研究性要求

虽然当前不做工程实现，但完整方法必须预留：

- sequence/sample packing，避免短样本浪费；
- 分辨率和序列长度 bucketing；
- 图像解码失败、空图、极端纵横比的数据监控；
- 分参数组梯度范数和 update norm；
- loss 按域、长度、分辨率、语言、教师来源拆分；
- checkpoint 可恢复、数据游标可恢复；
- 数据版本、过滤规则、teacher prompt 和 reward 版本可追溯；
- eval prompt 与训练 prompt 隔离。

Molmo2 报告 message-tree 数据组织与 packing 带来显著训练效率收益，说明系统设计本身会改变可承受的数据配方：[Molmo2](https://arxiv.org/abs/2601.10611)。

## 13. 缩放策略

“本地不做全量训练”不等于方法上跳过 scaling study。真正的大训练前应在外部算力上做固定框架的多尺度试验：

- 至少两个模型规模；
- 至少三个数据/token 预算；
- 关键数据域增减；
- 视觉 token/分辨率曲线；
- 冻结/解冻和不同 LR group；
- 对 SFT/RL 做能力增益与文本回归对照。

缩放 pilot 的作用是估计边际收益和暴露失败模式，而不是用极小训练绝对分数预测最终榜单。

## 14. 常见失败及回滚位置

| 症状 | 可能原因 | 优先检查阶段 |
|---|---|---|
| 能聊天但不看图 | 文本先验、视觉 loss 太弱、图文错配 | P0/P3/P6 |
| OCR 强但通用语义差 | 文档数据或高分辨率过度占比 | P1/P3/P4 |
| 图像能力增、文本能力降 | 文本 replay/参数 LR/采样失衡 | P3/P6 |
| 长视频高分、短图退化 | 长上下文 curriculum 覆盖旧分布不足 | P5 |
| CoT 很长但正确率不增 | 教师风格模仿或 RL 长度投机 | P7/P9 |
| grounding 格式正确但坐标错 | 坐标编码/视觉分辨率/奖励过宽松 | P1/P6/P9 |
| 公榜高、私榜低 | 污染、prompt 过拟合、教师泄漏 | P0/P4/P6 |
| RL 崩溃或输出模板化 | reward hacking、难度两极化、KL 不足 | P8/P9 |

