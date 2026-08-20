# Connector-only Stage 0 训练路线（历史材料）

> 归档说明：本文原先误把 connector-only warm-up 称为项目 L0。它现在只代表完整路线中的一个可选初始化阶段，不能作为 SOTA VLM 的完整 L0。

## 1. 目标与非目标

L0 的目标是用最少可训练参数证明三件事：视觉信息能进入 MiniCPM、模型会利用它、整个训练与保存链路可复现。

L0 不追求：

- 在综合 VLM benchmark 上达到成熟产品水平；
- 通过 connector 单独学会复杂 OCR、grounding、长视频或视觉推理；
- 重新训练 Qwen 视觉表征；
- 修复 MiniCPM 本身不具备的知识或推理能力。

## 2. 推荐架构

```text
image
  -> Qwen3.5 processor
  -> frozen Qwen visual tower
  -> frozen Qwen 2x2 merger        # [B, Nv, 1024]
  -> trainable MLP2x-GELU          # 1024 -> 1536 -> 1536
  -> splice into MiniCPM embeds    # replace one image placeholder with Nv vectors
  -> frozen MiniCPM5-1B
  -> next-token CE on assistant answer only
```

默认 connector：`Linear(1024,1536) -> GELU -> Linear(1536,1536)`。建议记录输入/输出范数；若数值尺度严重不匹配，再把 `LayerNorm(1024)` 作为明确消融，不静默加入。

### 2.1 forward 的最小契约

每个 batch 必须携带：

- `pixel_values` 与 Qwen processor 产生的 image grid；
- MiniCPM `input_ids`；
- 每个样本唯一的 image placeholder 位置；
- 展开后的 `attention_mask`、`position_ids`、`labels`；
- 每图视觉 token 数 `Nv`。

强制断言：

1. connector 输入最后一维是 1024，输出是 1536；
2. placeholder 数与图片数一致；
3. 展开后 `inputs_embeds/attention_mask/position_ids/labels` 序列长度一致；
4. visual token、user token、padding 的 label 均为 `-100`；
5. 无图片样本走原始 MiniCPM 路径，输出在容差内不变。

## 3. 分阶段执行

### 3.1 L0-Sanity：先证明实现正确

数据：64 条，然后 256 条人工核验的短 caption/VQA；统一 256×256；target 尽量少于 64 tokens。

要求：

- 64 条能明显过拟合；
- connector 梯度非零、有限，无 NaN/Inf；
- Qwen visual 和 MiniCPM 参数梯度为 `None`；
- frozen 权重训练前后 hash/逐张量摘要一致；
- 正确图像的 loss 明显低于 batch 内随机错配图像；
- blank/noise image 的输出不得与正确图像完全等价；
- save/resume 后下一个 step 的 loss 在数值容差内一致。

若过拟合失败，不允许用更大数据掩盖实现错误。

### 3.2 L0-Pilot：找稳定配方

数据：50k–100k 高质量样本；优先 PixMo-Cap 子集，混入短 VQA/OCR。建议先固定 384 或 448 分辨率。

建议超参网格：

| 参数 | 首选 | 最小搜索 |
|---|---:|---|
| optimizer | AdamW | 固定 |
| betas | (0.9, 0.999) | 固定 |
| precision | BF16 | BF16；不稳再验证 FP32 connector |
| micro-batch | 1 | 1 / 2（以 OOM 实测） |
| effective batch | 128 | 128 / 256 |
| learning rate | 5e-4 | 2e-4 / 5e-4 / 1e-3 |
| warmup ratio | 0.03 | 0.03 / 0.10 |
| schedule | cosine | 固定 |
| weight decay | 0.0 | 0.0 / 0.01 |
| grad clip | 1.0 | 固定 |
| epochs | 1 | 1；小数据可 2–3 |
| max text tokens | 512 | 256 / 512 |

LLaVA/TinyLLaVA 的 connector-only 配方使用 1e-3；Nemotron Nano V2 VL stage 0 使用 2e-4。模型、batch、数据规模不同，因此本项目应搜索 2e-4 到 1e-3，而不是机械复制单点。

### 3.3 L0-Main：只在 pilot 通过后扩量

建议 300k–600k 样本，1 epoch。初始任务混合可设为：

- 70% 高质量 caption；
- 20% 短 VQA / knowledge-light QA；
- 10% OCR / document QA。

这是工程起点而非论文规定比例。采样时按数据源设上限，防止一个大而嘈杂的数据源支配训练。所有验证/测试 benchmark 必须先从训练数据中做 URL、image hash 和近重复去重。

## 4. 数据路线

### 4.1 推荐层级

| 层级 | 数量 | 作用 |
|---|---:|---|
| hand-checked | 64–256 | 过拟合与接口测试 |
| smoke | 2k–5k | 测吞吐、checkpoint、评测脚本 |
| pilot | 50k–100k | 超参和架构选择 |
| main | 300k–600k | 稳定 L0 基线 |

### 4.2 数据源判断

- **PixMo-Cap**：约 717k，长而细致的人类语音转写式 caption；适合作为质量优先的主源。数据表是 ODC-BY-1.0，但图像通过 URL 引用，底层图片权利仍要单独审计。
- **PixMo-CapQA**：从 caption 构造的 QA，可补充短问答；同样要保留来源和生成方式。
- **LLaVA-Pretrain 558K**：最接近经典 LLaVA reproduction，方便横向比较；其图片来自 LAION/CC/SBU，数据卡明确提示遵循源许可且部分 URL 已失效，不应把“能下载”误写成“可自由商用”。
- **COCO captions**：适合 smoke/pilot 的稳定基线；仍需遵循 COCO 图片和 annotation 许可。

每个 manifest 至少记录：`dataset_name, repo, revision, split, original_id, image_uri, image_sha256, text_sha256, license_note, filter_version, benchmark_overlap`。

### 4.3 是否缓存视觉特征

视觉塔冻结时可以离线缓存 merger 输出，能显著减少多轮 projector sweep 的计算；代价是磁盘：

- 256 分辨率：约 128 KiB/图，100k 约 12.5 GiB；
- 448 分辨率：约 392 KiB/图，100k 约 38.3 GiB；
- 448 分辨率：500k 约 191 GiB。

以上未计 shard 索引和元数据。只做一次训练时未必值得缓存；做 3 个 LR × 3 个 connector 消融时更可能划算。缓存必须绑定 Qwen checkpoint hash、processor config、resize 规则和 dtype。

## 5. 单卡 16 GB 工程预算

仅加载 MiniCPM BF16 权重约 2.16 GB，Qwen visual BF16 约 0.20 GB，3.94M connector 参数和 Adam 状态相对很小。真正的显存压力来自 24 层 MiniCPM 的 activation、视觉 token 数和序列长度。

即使 LLM 参数冻结，为了把梯度传回 connector，计算图仍需产生输入梯度；不能把“冻结 LLM”理解成“没有 LLM activation”。建议：

- micro-batch 从 1 开始；
- gradient accumulation 达到 effective batch 128；
- 开启 gradient checkpointing；
- `use_cache=False`；
- 先用 256/384 和 max text 256；
- SDPA 作为默认，WSL2 环境稳定后再评估 FlashAttention 2；
- dataloader 采用本地 shard、pinned memory、持久 worker；
- 先跑 100 steps，记录 peak VRAM、step time、视觉 token/样本、文本 token/样本、I/O wait。

不要在 benchmark 前承诺 600k 训练的完成时间。按实测 `samples/s` 和有效样本数计算，并额外预留评测、保存和失败重跑时间。

## 6. 评测与选择 checkpoint

### 6.1 必须有的低成本评测

- held-out caption/VQA teacher-forced loss；
- 正确图像 vs batch-shuffled 图像 loss gap；
- prompt-only、blank image、noise image 对照；
- 200–1,000 条人工可读固定生成集；
- VQAv2、TextVQA/OCR、MMStar 各抽固定小集做迭代回归；
- 无图文本固定集，验证 frozen L0 的输出不变。

### 6.2 checkpoint 选择

不能只看 train loss。建议 pilot 评分：

```text
score = 0.35 * caption/VQA heldout
      + 0.25 * image-shuffle sensitivity
      + 0.20 * OCR/TextVQA subset
      + 0.20 * human-readable fixed set
```

各项先归一化到同一尺度。文本回归和权重完整性是硬门槛，不纳入加权后“相互抵消”。

## 7. 最小消融矩阵

按价值/成本排序：

1. Linear vs MLP2x-GELU；
2. MiniCPM final vs SFT；
3. 256 vs 448 分辨率；
4. connector-only vs 解冻 Qwen merger；
5. caption-only vs caption+VQA/OCR；
6. 随机 connector 初始化 vs shared-token embedding ridge/Procrustes 初始化；
7. 无蒸馏 vs 原 Qwen3.5 VLM teacher distillation。

第 6、7 项属于研究增强，不是行业默认基线。应在基本 CE 路线稳定后再做。

## 8. 何时进入 L0.5/L1

满足以下任一情况就应进入 L0.5，而不是继续盲目加 L0 数据：

- train/heldout loss 已收敛，但图像打乱敏感性和 VQA 指标仍弱；
- Linear/MLP、分辨率、LR 都已排除，connector 容量仍成为瓶颈；
- 模型会描述显著物体，但不能建立细粒度关系、OCR 或指令遵循；
- TinyLLaVA 式部分解冻试验在 10k–50k 数据上已有一致增益。

L0.5 首选：解冻 Qwen merger + MiniCPM 顶部 4–8 层 LoRA，connector LR 1e-4–2e-4、backbone/LoRA LR 1e-5–2e-5，并加入 5–10% 文本 replay。L1 再扩大多模态指令数据和可训练层范围。
