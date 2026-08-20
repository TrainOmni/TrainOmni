# 工程准备与交付检查表（历史模型接入版）

> 归档说明：当前 researcher 只做纯调研，不做本地全量训练或工程实现。本文留作未来工程团队参考，不属于当前主交付。

## A. 资产门槛

- [x] 记录 Qwen 本地路径、config hash、weight hash。
- [x] 记录 MiniCPM 本地路径、config/index hash。
- [ ] MiniCPM `model-00000-of-00001.safetensors` 存在且大小/hash 与下载记录一致。
- [ ] 补充 MiniCPM5-1B-SFT，并固定 repo revision/hash。
- [ ] 固定 `transformers>=5.6` 的精确版本、PyTorch/CUDA 版本和 lockfile。
- [ ] 在 WSL2 内完成一次两模型的离线加载，不隐式联网。
- [ ] 把模型许可证与数据许可证分开归档。

## B. 模型 Adapter 合约

- [ ] `QwenVisionAdapter` 只加载 `model.visual.*`，不常驻完整 Qwen LLM。
- [ ] 对照原 Qwen 模型验证视觉模块输出 shape/数值。
- [ ] connector 配置显式记录 input/output dim、激活、norm、bias。
- [ ] MiniCPM unused token 映射写入独立配置，不覆盖 tokenizer 原文件。
- [ ] 单图、变长图片、无图片三条路径都有单测。
- [ ] 视觉 placeholder 数和 feature group 数强断言一致。
- [ ] Qwen processor 的 grid 信息不丢失。
- [ ] MiniCPM 使用连续 1D position ids。
- [ ] visual/user/pad label 均为 `-100`。
- [ ] 无图输入与原 MiniCPM logits/生成结果在容差内一致。

## C. 冻结与优化器

- [ ] Qwen visual、MiniCPM 全部 `requires_grad=False`。
- [ ] optimizer 参数组只包含 connector/可选边界 embedding。
- [ ] 训练一步后打印 trainable names、参数量、梯度 norm。
- [ ] frozen 模块无梯度，训练前后摘要一致。
- [ ] connector FP32/BF16 数值稳定性完成 100-step 对照。
- [ ] gradient checkpointing 开启，`use_cache=False`。
- [ ] grad accumulation 与 scheduler 以 optimizer step 而非 micro-step 计数。

## D. 数据协议

- [ ] 原始样本协议至少支持 `id/images/messages/source/license`。
- [ ] 每个数据集有 manifest：repo、revision、split、hash、过滤版本。
- [ ] 图片加载失败有可追踪的 quarantine 日志，不静默换图。
- [ ] caption/VQA/OCR 的采样比例由 recipe 管理。
- [ ] 图像分辨率/视觉 token 数记录进 batch metrics。
- [ ] train/val/test 按 image hash/URL/近重复去重。
- [ ] 与 VQAv2/TextVQA/MMStar 等评测集做 overlap 审计。
- [ ] prompt/chat template 版本化。

## E. 运行记录

- [ ] 每次 run 保存 git/代码快照标识；当前 workspace 若非 git repo，则保存文件清单和 hash manifest。
- [ ] 保存模型 revision/hash、processor/tokenizer hash、数据 manifest hash。
- [ ] 保存完整 recipe、seed、hostname、GPU、driver、CUDA、PyTorch、Transformers。
- [ ] 每 N steps 记录 loss、LR、grad norm、tokens/s、samples/s、peak VRAM。
- [ ] 分别记录 image tokens 和 text tokens，避免只有 samples/s。
- [ ] 保存首批展开后的 token/label 可读审计样本。
- [ ] checkpoint 使用原子写入并验证可重新加载。
- [ ] resume 测试覆盖 optimizer、scheduler、sampler/RNG 状态。

## F. L0-Sanity 验收

- [ ] 64 条样本可过拟合。
- [ ] 正确图像 loss < shuffled-image loss，且 gap 随训练扩大。
- [ ] blank/noise/prompt-only 对照完成。
- [ ] connector 参数确实变化，冻结权重不变化。
- [ ] save/resume 后损失轨迹可复现。
- [ ] 无图文本路径无回归。
- [ ] 未出现 NaN/Inf、label 泄漏或 placeholder 错位。

## G. L0-Pilot 验收

- [ ] 50k–100k 数据的训练/验证曲线完整。
- [ ] LR 2e-4/5e-4/1e-3 至少完成低成本筛选。
- [ ] Linear 与 MLP2x 基线完成。
- [ ] 256 与 448 分辨率收益/成本完成对照。
- [ ] MiniCPM final 与 SFT 至少完成同预算小对照。
- [ ] 固定生成集、VQA、OCR、image-shuffle 指标齐全。
- [ ] 100-step 性能报告给出峰值显存、吞吐和预计主实验耗时。
- [ ] 根据退出条件明确选择 L0-Main 或 L0.5，而不是只报最低 loss。

## H. checkpoint 交付内容

- [ ] `adapter.safetensors`（默认只含 connector）。
- [ ] `adapter_config.json`（结构、dims、activation、norm、token map）。
- [ ] `base_models.json`（两侧 repo/revision/hash/module path）。
- [ ] processor/tokenizer 配置或不可变引用。
- [ ] training recipe 与数据 manifest hash。
- [ ] eval results、日志位置、已知限制。
- [ ] 最小离线推理命令和一条 golden sample。
- [ ] 兼容性测试：CPU load、目标 GPU load、无图文本、单图生成。

## I. 当前明确依赖

| 依赖方 | 需要内容 | 状态 |
|---|---|---|
| downloader | 补齐 MiniCPM final 权重与 hash | 阻塞 smoke test |
| downloader | 下载 MiniCPM5-1B-SFT | 主实验建议，未开始 |
| framework | Qwen visual adapter + MiniCPM embedding splice | 未实现 |
| framework | 变长视觉 token collator、label mask、单元测试 | 未实现 |
| research/data | smoke/pilot 数据 manifest 与许可审计 | 未开始 |
