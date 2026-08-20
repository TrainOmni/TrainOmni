# TrainOmni Research：完整 L0 方法研究

更新时间：2026-08-20  
角色：researcher  
范围：纯调研；当前不绑定具体 ViT/LLM，不执行本地全量训练，不修改其他角色目录。

## 结论先行

本项目把 **L0** 定义为：从基础视觉/语言模型与原始数据出发，完成架构、预训练、中期训练、指令微调、推理蒸馏、偏好优化、在线强化学习、评测和数据飞轮，使最终 VLM 在相同参数/算力档位逼近或进入业界前沿 Pareto 面的完整能力建设流程。

这一定义可以使用，但它与很多开源项目中的 **Stage 0** 不同。后者通常只表示“冻结 ViT 和 LLM、训练 projector/connector”的短暂模态对齐。为避免继续混淆，本研究保留项目级名称 L0，内部训练阶段统一写作 **P0–P10**；connector warm-up 只叫 P2，不再叫 L0。

最重要的研究判断：

1. 匹配 SOTA 不是“把图像 token 接进 LLM”即可实现，而是数据、视觉表征、全参数联合训练、后训练、评测和数据飞轮的系统工程。
2. “SOTA”不是单一总分。必须先限定参数/推理预算档位，再同时验证感知、OCR/文档、知识与推理、空间/grounding、视频、Agent、幻觉、文本能力回归和效率。
3. 对已有 ViT + LLM 的主流路线，connector-only 只是可选热身；真正决定上限的是大规模全参数联合多模态训练及其后训练。PEFT 可以降低成本，但目前没有充分证据把它当作追平同档 SOTA 的默认方案。
4. 前沿路线正在从“后接视觉”转向更早、更长时间的原生图文联合训练；但这需要极大数据与算力，不能直接把超大模型结论外推到小模型。
5. 数据质量、混合比例、样本/损失权重和评测防污染，与模型结构同等重要。公开视频榜单的单点提升不能独立证明视觉能力。

## 主文档

- [00_scope_and_definition.md](./00_scope_and_definition.md)：L0、SOTA、证据等级和边界。
- [01_capability_target_and_evaluation.md](./01_capability_target_and_evaluation.md)：能力树、对照组、评测矩阵与阶段门禁。
- [02_architecture_and_visual_tokens.md](./02_architecture_and_visual_tokens.md)：融合架构、ViT、分辨率、视觉 token 与视频设计。
- [03_complete_training_pipeline.md](./03_complete_training_pipeline.md)：P0–P10 完整训练生命周期及参数更新策略。
- [04_data_system_and_curriculum.md](./04_data_system_and_curriculum.md)：数据分类、清洗、配比、课程、损失归一化与治理。
- [05_posttraining_distillation_and_rl.md](./05_posttraining_distillation_and_rl.md)：SFT、长思维、强蒸弱、偏好优化与在线 RL。
- [06_evidence_and_reading_guide.md](./06_evidence_and_reading_guide.md)：按问题组织的论文阅读顺序和证据账本。
- [07_decisions_risks_and_open_questions.md](./07_decisions_risks_and_open_questions.md)：已形成判断、风险、需要后续实证回答的问题。

## 推荐的完整 L0 生命周期

```text
P0 能力契约、评测和数据治理
 └─ P1 视觉编码器专项预训练/继续训练
     └─ P2 可选的 connector 对齐热身
         └─ P3 全参数联合多模态预训练（主要算力阶段）
             └─ P4 高质量 cooldown / mid-training
                 └─ P5 分辨率、长上下文、多图和视频扩展
                     └─ P6 通用多模态 SFT
                         └─ P7 推理冷启动与强模型蒸馏
                             └─ P8 离线偏好优化
                                 └─ P9 在线多模态 RL / Agent RL
                                     └─ P10 效率对齐、回归验证与数据飞轮
```

这不是要求每个项目机械地执行十一个独立 checkpoint，而是一张职责完整性检查表。相邻阶段可以合并；但要声称完整 L0，就不能遗漏能力契约、联合训练、后训练、反事实评测和持续数据闭环。

## 证据使用规则

文档使用四种标记：

- **报告事实**：技术报告或官方材料明确披露。
- **跨项目共同模式**：至少两个独立前沿项目出现的做法。
- **研究建议**：基于多个事实综合出的可执行假设，仍需消融验证。
- **未决问题**：公开材料不足，不能假装已有标准答案。

本轮优先采用官方技术报告、论文和官方仓库；不把二手博客或榜单宣传当作训练方法证据。

## 历史材料

[archive/2026-08-19_model_specific_stage0](./archive/2026-08-19_model_specific_stage0/) 保存第一版针对具体 ViT/LLM 的接口审计和 connector-only 路线。它们已经退出主线，仅供未来模型接入时参考。
