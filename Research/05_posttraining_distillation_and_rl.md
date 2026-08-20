# 多模态后训练：SFT、蒸馏、偏好与强化学习

## 1. 后训练解决什么

联合预训练决定模型“可能知道和看见什么”，后训练决定它是否能：

- 按指令选择和表达视觉证据；
- 在难题上稳定展开推理；
- 控制答案长度、格式、不确定性和拒答；
- 使用 crop/zoom、OCR、代码、搜索、GUI 等工具；
- 从自己的错误分布而非教师静态分布中继续学习。

后训练不能补回 ViT 已丢失的像素细节，也不能用奖励掩盖数据污染。

## 2. P6：通用多模态 SFT

### 2.1 数据组成

SFT 集应覆盖：简答与开放回答、OCR/文档/图表、grounding、STEM、多图/视频、多轮、工具 schema、安全与纯文本回归。一个总榜单高分的数据混合不一定是好 SFT，因为最终模型还要学会格式、拒答和任务切换。

### 2.2 Loss mask

常规聊天 SFT 对 assistant response 计算 CE，对 system/user prompt、图像占位和 padding mask。结构化 grounding/工具轨迹要明确哪些观察、思考、动作、结果 token 参与 loss，避免把环境返回文本当作模型需要预测的答案。

### 2.3 全参数还是 PEFT

- 全参数：更有机会让视觉、connector 与语言表示共同适配，是追上限的默认研究基线；
- LoRA/DoRA：节省显存、适合快速域适配和消融；
- 只训 LLM：适合已有原生视觉表征很强、数据主要改变行为的情况；
- 只训 connector：仅适合接口热身，不适合作为完整 SFT 结论。

Idefics3 采用参数高效适配并明确讨论了完全解冻可能带来的上限空间，这恰好说明 PEFT 的价值是成本，不应自动等价于最高能力：[Idefics3](https://arxiv.org/abs/2408.12637)。

## 3. P7：推理冷启动

在线 RL 前，模型需要一批高质量、可读、可验证的视觉推理示范。来源可包括：

- 强教师对同一图像/问题的多次 rollout；
- 程序/规则验证后的正确轨迹；
- 人工纠错与过程标注；
- 将答案反推为证据定位、计算和结论步骤；
- 图像 crop/zoom 或 OCR 工具的成功轨迹。

过滤不能只看 final answer：还要检查步骤是否引用正确对象、坐标/数值是否一致、是否通过语言先验猜中，以及中间工具结果有没有被真正利用。

### 3.1 Thinking 与 non-thinking

建议把长思维和直接回答作为两个可控模式：

- 简单感知/OCR 不应强制长 CoT；
- 复杂 STEM/Agent 可以启用 thinking；
- 训练和评测都要记录思维 token 成本；
- final answer 应能在不暴露内部长轨迹时保持可核验。

SmolVLM 的小模型研究显示，更多 CoT 数据不一定提升小 VLM，甚至可能伤害表现：[SmolVLM](https://arxiv.org/abs/2504.05299)。这意味着小模型首先受容量和视觉信息瓶颈约束，不能把“大模型长思维配方”原样缩小。

## 4. 强到弱蒸馏

小/中模型追赶同档前沿时，蒸馏往往比单纯堆弱标签更有效，但至少要区分两类：

### 4.1 Off-policy response distillation

学生学习教师已经生成的正确答案/推理轨迹。优点是便宜、稳定；风险是学生只模仿教师分布，不会处理自身特有错误。

### 4.2 On-policy distillation

先让学生在当前 policy 上生成，再用教师对学生输出打分、修正或提供 token-level/sequence-level KL 信号。它更贴近学生实际分布，但成本高，教师校准和隐私风险更复杂。

Qwen3-VL 报告组合使用 off-policy 与 on-policy 的强到弱蒸馏，为这条两步路线提供了前沿实例：[Qwen3-VL](https://arxiv.org/abs/2511.21631)。

### 4.3 蒸馏不应只复制答案风格

建议同时蒸馏：

- 视觉证据选择；
- 正确的结构化坐标/时间段；
- 必要而不过长的推理；
- 不确定性与拒答；
- 工具调用时机；
- 多种等价表达，避免单一教师口吻。

## 5. P8：离线偏好优化

对同一输入构造 chosen/rejected，可以优化 correctness、视觉忠实、简洁、格式、安全和工具策略。DPO 类方法不需要在线 rollout，适合先做稳定暖启动。

InternVL3.5 的 Cascade RL 先做离线 MPO，再做在线 GSPO；其 MPO 同时考虑偏好、质量和生成目标，避免只拉开 chosen/rejected 而破坏基本生成：[InternVL3.5](https://arxiv.org/abs/2508.18265)。

### 偏好对设计

高价值 rejected 不是随机垃圾，而是“很像对但在关键视觉证据上错”：

- OCR 字符差一位；
- 框住相邻物体；
- 数值计算对但读错图例；
- 图像证据正确但结论过度；
- 工具动作格式正确但目标元素错误；
- final answer 正确但过程使用错图，属于偶然猜中。

避免把长度、固定标题或教师口吻变成 chosen 的捷径。

## 6. P9：在线多模态 RL

在线 RL 的核心价值是让模型在自己的 rollout 分布上探索、失败和修正。训练任务应分成两类奖励。

### 6.1 可验证奖励

适用：数学、代码、OCR、计数、选择题、框/点、时间区间、GUI 环境成功。

可使用：

- exact match / normalized edit distance；
- 数值容差和单位检查；
- IoU、point distance、F1；
- 程序执行与测试用例；
- 环境 task success、动作合法性、步数成本。

### 6.2 开放式模型奖励

适用：详细描述、开放 VQA、解释质量、安全和审美。需要：

- 多 rubric，而不是一个笼统“好不好”；
- judge 对图像可见，避免只评语言流畅；
- 与人工偏好校准；
- 多 judge/规则交叉；
- 对 reward hacking 和 length bias 做 adversarial audit。

Qwen3-VL 报告将可确定验证的 reasoning RL 与覆盖 caption/VQA/OCR/文档/grounding 的 general RL 分开，并组合规则与模型奖励：[Qwen3-VL](https://arxiv.org/abs/2511.21631)。Kimi K2.5 也披露了面向 IoU、编辑距离、计数误差等视觉任务的 outcome reward，以及开放任务的生成式 reward model：[Kimi K2.5](https://arxiv.org/abs/2602.02276)。

## 7. 在线样本难度与课程

全对的题几乎没有优势信号，全错的题也常缺乏有效正样本。前沿报告普遍进行难度筛选或 curriculum：

- Qwen3-VL 对每个 query 多采样，并过滤全错和过易样本；
- InternVL3.5 的在线阶段偏向中等成功率 query；
- Kimi-VL 使用课程/优先级采样和正确性奖励。

来源：[Qwen3-VL](https://arxiv.org/abs/2511.21631)、[InternVL3.5](https://arxiv.org/abs/2508.18265)、[Kimi-VL](https://arxiv.org/abs/2504.07491)。

这形成一个可迁移原则：把在线算力优先花在“当前 policy 偶尔能做对，但不稳定”的题上，同时为完全做不对的簇回流 SFT/预训练数据。

## 8. Agentic 视觉推理

静态 VQA 与“会用视觉工具”是两个层级。Agent 路线需要模型学习循环：

```text
观察 → 判断信息是否足够 → 选择 crop/zoom/OCR/code/search/GUI 动作
     → 读取工具结果 → 更新状态 → 给最终答案
```

[DeepEyes](https://arxiv.org/abs/2505.14362) 探索通过强化学习让模型在推理时调用视觉工具、放大并重新观察。这类方法特别适合超高分辨率、小目标和视觉 token 受限场景，但必须单独报告工具调用成本和失败率。

## 9. RL 稳定性与安全边界

必须监控：

- reward 与真实评测的相关性；
- KL、entropy、输出长度和重复；
- 分域 reward，防止一个易任务主导；
- 正确图/错图 reward 差，防止语言奖励压过视觉；
- 工具调用次数、非法动作、超时和越权；
- 安全拒答的 over-refusal/under-refusal；
- policy、reward model、judge 和数据版本。

若 reward 上升而私有正确率、视觉依赖差值或人工偏好不升，应判定为 reward hacking，不是能力提升。

## 10. 推荐的后训练分叉

### 基础路线

`高质量通用 SFT → 离线偏好 → 可验证任务在线 RL`

适合先建立稳定的通用 VLM，不强求长思维和工具。

### 推理路线

`通用 SFT → 视觉 CoT 冷启动 → off-policy 蒸馏 → on-policy 蒸馏 → reasoning RL`

适合 STEM、图表、文档和多步空间问题。

### Agent 路线

`通用 SFT → 工具轨迹 SFT → 离线动作偏好 → 环境在线 RL`

适合 GUI、视觉搜索、crop/zoom 和代码执行。

三条路线可共享 base checkpoint，但不建议一开始混成一个 reward；先分别证明能力，再做统一策略或 mode token。

## 11. 必做消融

1. SFT-only vs SFT + preference vs SFT + preference + online RL；
2. final-answer-only 过滤 vs 同时验证过程；
3. off-policy vs 加 on-policy 蒸馏；
4. 单一 reward vs 多 rubric；
5. 随机 query vs 中等难度 curriculum；
6. 强制长 CoT vs thinking toggle；
7. 无工具 vs 工具可用、并对齐总推理成本；
8. 正确图/错图 reward 与性能差；
9. 多模态后训练前后文本能力回归；
10. reward 模型与独立人工/私有 evaluator 的一致性。

