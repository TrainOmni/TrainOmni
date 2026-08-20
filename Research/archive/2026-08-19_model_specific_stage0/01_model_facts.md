# 模型事实与接口审计（历史材料，已被主路线取代）

> 归档说明：本文绑定具体模型，并把 connector-only 对齐称为 L0。2026-08-20 起，项目 L0 改为“完整 SOTA 能力建设流程”；本文仅保留为后续具体模型接入时的参考。

> 标签约定：**本地事实**来自当前文件或 tensor header；**官方事实**来自官方模型卡/实现；**工程推断**是基于两者得到的设计判断。

## 1. 本地资产状态

| 项目 | Qwen | MiniCPM |
|---|---|---|
| 路径 | `D:\Models\VLM\Qwen3.5-0.8B` | `D:\Models\LLM\MiniCPM5-1B` |
| 精确身份 | `Qwen/Qwen3.5-0.8B` | `openbmb/MiniCPM5-1B` |
| 架构 | `Qwen3_5ForConditionalGeneration` | `LlamaForCausalLM` |
| 本地完整性 | 权重已存在 | **权重缺失，仅索引存在** |
| 许可证 | Apache-2.0 | Apache-2.0 |

2026-08-19 本地 SHA256：

| 文件 | SHA256 |
|---|---|
| Qwen `config.json` | `B90B86F35C8E6925EF74EE04D0E758F0A845C83A42089AD82BBAA948DE9B4204` |
| Qwen safetensors | `04B1C301231DD422B8860DB31311AB2721511346A32CB1E079C4C4E5F1FE4696` |
| MiniCPM `config.json` | `6A6509B646CB3169616C5FFC3196E7CCAF9D4D6BC17B266581D241A31C217714` |
| MiniCPM index | `162ADD042E75ABC3D571C4A8679523FA4F1FFC55D1FEA25FC6658A19D6E957EE` |

MiniCPM index 指定唯一分片为 `model-00000-of-00001.safetensors`，声明大小 2,161,265,664 bytes；该文件当前不存在。因此 tokenizer/config 研究有效，但任何“已成功加载 MiniCPM 权重”的结论都不成立。

## 2. Qwen3.5 视觉侧

### 2.1 它是什么

**官方事实：** Qwen3.5 是从文本、图像、视频混合数据训练的原生多模态模型；0.8B checkpoint 不是单独的视觉模型。Transformers 实现中的视觉模块位于 `model.visual.*`。

**本地事实：** 当前 checkpoint 的视觉配置为：

| 字段 | 值 |
|---|---:|
| blocks | 12 |
| hidden size | 768 |
| MLP intermediate | 3072 |
| attention heads | 12 |
| spatial patch | 16×16 |
| temporal patch | 2 |
| spatial merge | 2×2 |
| merger output | 1024 |

从 safetensors header 统计：

- `model.visual.*`：100,592,896 参数；
- 不含 merger 的视觉塔：88,004,352 参数；
- Qwen merger：12,588,544 参数；
- patch embed 卷积：`[768, 3, 2, 16, 16]`；
- merger 先把 2×2 个 768 维 patch 拼接为 3072 维，再经 `3072 -> 3072 -> 1024`。

### 2.2 视觉 token 数

**工程推断：** 对单张、无额外切块且边长能被 32 整除的正方形图片，patch 16 加 2×2 merge 后，进入 MiniCPM 的 token 数约为 `(R / 32)^2`：

| 输入分辨率 | 视觉 token（约） | 单样本 BF16 特征体积（1024 维） |
|---:|---:|---:|
| 256×256 | 64 | 128 KiB |
| 384×384 | 144 | 288 KiB |
| 448×448 | 196 | 392 KiB |
| 512×512 | 256 | 512 KiB |
| 768×768 | 576 | 1.125 MiB |

真实 token 数必须以 processor 输出的 grid 为准，不能只按原图尺寸猜测；动态 resize、长宽比和视频都会改变结果。

## 3. MiniCPM5-1B 语言侧

**官方/本地一致事实：**

| 字段 | 值 |
|---|---:|
| 架构 | LlamaForCausalLM |
| hidden size | 1536 |
| layers | 24 |
| intermediate | 4608 |
| query heads / KV heads | 16 / 2 |
| head dim | 128 |
| vocabulary | 130,560 |
| context | 131,072 |
| embedding tie | false |
| dtype | BF16 |
| 总参数 | 1,080,632,832 |
| 非 embedding 参数 | 679,552,512 |

官方模型卡把当前 final checkpoint 描述为 SFT 后继续经过 RL 和 online policy distillation（OPD）的模型，并同时发布 Base、SFT、final 三个阶段。官方当前要求 `transformers>=5.6`。

### 3.1 tokenizer 可利用空间

- `<|im_start|>`：130072；
- `<|im_end|>`：130073；
- 存在 478 个 `<unused_token_N>`，首个 id 为 130082。

**推荐：** 初版映射为三枚 unused token，例如 `<vision_start>`、`<image>`、`<vision_end>`；chat template 中保留一个 `<image>` 占位，forward 时把该位置扩展/替换为 N 个 projected visual embeddings。这样不必 resize embedding 和 LM head。

## 4. 必须新增的桥接层

Qwen merger 输出 1024，MiniCPM token hidden 是 1536，不能直接拼接。

| 方案 | 参数量 | 评价 |
|---|---:|---|
| Linear `1024 -> 1536` | 1,574,400 | 最小对照；表达力有限 |
| MLP2x-GELU `1024 -> 1536 -> 1536` | 3,935,232 | **L0 默认**；与 LLaVA/TinyLLaVA 基线一致 |
| 替换 Qwen merger 最后一层 `3072 -> 1536` | 4,720,128 | 去掉 1024 bottleneck，但丢失原 1024 输出映射 |
| 重训完整 merger/connector | 至少约 12.6M | 更强但不再是最保守 L0 |

默认 connector 前可加一个无仿射 LayerNorm/RMSNorm，或记录 source/target embedding norm 后决定是否使用。任何 normalization 变体都必须单独登记，避免“同名 MLP”实际不等价。

## 5. 位置编码边界

Qwen 视觉塔内部继续使用其原生空间/时间位置编码，这是视觉 encoder 自己的职责。视觉特征离开 merger 后，MiniCPM 是普通 Llama 架构，应把 N 个视觉 token 当成连续的 1D token positions。

因此：

- 保留 Qwen vision 内部 grid/rotary 逻辑；
- 不把 Qwen LLM 的多模态 RoPE index 复制到 MiniCPM；
- MiniCPM 序列形如 `BOS, user text, vision_start, v1...vN, vision_end, assistant text`；
- visual/user/padding labels 均设为 `-100`，只监督 assistant target。

这是一个需要 framework 用单元测试锁定的接口契约。

## 6. checkpoint 选择建议

| checkpoint | 用途 | 风险 |
|---|---|---|
| MiniCPM5-1B final | L0 冒烟、保持现成聊天能力、对照 | RL/OPD 后分布可能对多模态再训练更敏感 |
| MiniCPM5-1B-SFT | **主 L0/L1 推荐** | 需要新增下载与校验 |
| MiniCPM5-1B-Base | 大规模多模态预训练研究 | 数据和算力需求显著提高，不适合当前单卡首轮 |

最终应把三者至少做 `final vs SFT` 小规模对照，而不是只凭直觉决定。

## 7. 官方来源

- [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5 Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [MiniCPM5-1B model card](https://huggingface.co/openbmb/MiniCPM5-1B)
- [OpenBMB MiniCPM repository](https://github.com/OpenBMB/MiniCPM)
