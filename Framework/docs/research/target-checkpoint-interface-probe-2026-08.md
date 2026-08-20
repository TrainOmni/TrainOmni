# Qwen3.5 Vision + MiniCPM5-1B 接口探测

- 状态：静态 checkpoint probe 完成；尚未执行 Transformers forward
- 日期：2026-08-19
- 输入清单：`D:\Codex\TrainOmni\Downloads\checkpoint-assets-20260819.md`
- 方法：只读检查 config、processor/tokenizer、model card、safetensors index/header；未修改模型目录

## 1. 可复现资产

| 资产 | 本地路径 | Repo / revision | 权重 SHA-256 | 状态 |
|---|---|---|---|---|
| Qwen3.5-0.8B | `D:\Models\VLM\Qwen3.5-0.8B` | `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17` | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` | 488 tensors，完整 |
| MiniCPM5-1B | `D:\Models\LLM\MiniCPM5-1B` | `openbmb/MiniCPM5-1B` @ `87179e5c1f455ef22e6223592d2d61351b525bfc` | `7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d` | 219 tensors，11/11 文件完整 |

两者均没有仓库自定义 Python 文件，`config/tokenizer auto_map` 为 null；checkpoint 本身不要求 `trust_remote_code=True`。

## 2. Qwen3.5-0.8B 可复用视觉部分

### 2.1 Config

- 完整模型：`Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`。
- Vision depth 12，hidden 768，12 heads，MLP 3072。
- 3D patch embedding：`[out=768, in=3, temporal=2, height=16, width=16]`。
- `spatial_merge_size=2`，每 2×2 个 patch token 合并。
- merger 输出 `out_hidden_size=1024`，原本对齐 Qwen language hidden 1024。
- processor：`Qwen3VLProcessor` + `Qwen2VLImageProcessorFast`。
- image pixel policy：shortest edge 65,536 pixels，longest edge 16,777,216 pixels。

### 2.2 Safetensors shape 证据

```text
model.visual.patch_embed.proj.weight       BF16 [768, 3, 2, 16, 16]
model.visual.pos_embed.weight              BF16 [2304, 768]
model.visual.merger.norm.weight            BF16 [768]
model.visual.merger.linear_fc1.weight      BF16 [3072, 3072]
model.visual.merger.linear_fc2.weight      BF16 [1024, 3072]
```

解释：2×2 spatial merge 将 4×768 拼成 3072，merger 最终产生 1024 维 visual tokens。因此最稳妥的首版不是拆掉 merger，而是复用完整 `model.visual`，再增加 1024→1536 的 connector。

权重命名前缀：

- `model.visual.*`：153 tensors，可作为独立 vision component 抽取。
- `model.language_model.*`：320 tensors，不进入目标 composite。
- `mtp.*`：13 tensors，不进入目标 composite。

## 3. MiniCPM5-1B 语言模型接口

- 标准 `LlamaForCausalLM`，`model_type=llama`。
- hidden 1536，24 layers，16 query heads / 2 KV heads，head_dim 128。
- vocab 130,560；input embedding 与 lm_head 不共享。
- native context 131,072，BF16，1D Llama RoPE，`rope_theta=5,000,000`。
- model card 报告总参数 1,080,632,832；非 embedding 参数 679,552,512。

Safetensors：

```text
model.embed_tokens.weight                 BF16 [130560, 1536]
lm_head.weight                            BF16 [130560, 1536]
model.layers.0.self_attn.q_proj.weight    BF16 [2048, 1536]
model.layers.0.self_attn.k_proj.weight    BF16 [256, 1536]
model.layers.0.self_attn.v_proj.weight    BF16 [256, 1536]
model.layers.0.self_attn.o_proj.weight    BF16 [1536, 2048]
```

Tokenizer 已有 `<|im_start|>` / `<|im_end|>`、thinking 和 tool tokens；从 ID 130082 开始有大量 `<unused_token_N>`。原 chat template 只把字符串 content 当作正文，不处理 image/video content block。

## 4. Composite 必须解决的接口差异

### 4.1 Visual connector

确定的 shape：

```text
Qwen vision + merger: [batch_visual_tokens, 1024]
MiniCPM token embeddings: [batch_text_tokens, 1536]
required connector: 1024 -> 1536
```

L0 候选：

1. `LayerNorm(1024) + Linear(1024, 1536)`：最小、适合接口和数据 sanity check。
2. `Linear(1024, 1536) + GELU + Linear(1536, 1536)`：更接近常见 LLaVA projector，容量更强。
3. 重新训练 Qwen merger 直接输出 1536：参数更多，破坏已训练视觉输出空间，不建议作为第一条基线。

Framework 不在没有对照实验时写死 projector；Recipe 选择实现，component catalog 统一命名为 `connector`。

### 4.2 Tokenizer 与 image placeholder

Qwen 的 `<|vision_start|><|image_pad|><|vision_end|>` IDs 位于 248k 词表，不能直接用于 130,560 词表的 MiniCPM。

两种方案：

- 新增明确的 vision special tokens，并 resize MiniCPM input embedding 与 untied lm_head。语义清楚、导出标准，但两个矩阵都要扩展并定义初始化。
- 显式保留一个 `<unused_token_N>` 作为 image placeholder，processor 将其扩展为视觉 token 数；对应 embedding 在 forward 时由 visual embeddings 替换。无需 resize，适合 L0 smoke，但导出必须携带 alias/profile，不能静默改变 tokenizer 语义。

建议：M1/M2 smoke 先支持“reserved token、无 resize”路径；正式可发布 checkpoint 默认新增命名 special token，并对 tokenizer surgery 做版本化 manifest。

无论哪种方案，placeholder、vision start/end 和 padding positions 都必须在 labels 中 mask 掉。

### 4.3 Chat template 与 formatter

- Qwen template 支持 list-form multimodal content，但使用 Qwen tokenizer tokens。
- MiniCPM template 保留 MiniCPM 对话、thinking 与 tool 语义，但只消费字符串 content。

Composite formatter 应以 MiniCPM chat protocol 为基础，在 user content 中插入框架自有 placeholder；不能整套复制 Qwen template，也不能把 `<image>` 写回 canonical dataset。

### 4.4 Position IDs

Qwen language model使用 multimodal RoPE sections；MiniCPM 是标准 1D Llama RoPE。复用 `model.visual` 后，视觉 encoder 内部已经处理 patch 空间/时间位置；进入 MiniCPM 的 visual tokens 第一版可按普通 1D sequence positions 排列。

这只是最小可行假设，必须通过以下对照验证：

- image token 顺序与 merger grid 顺序一致；
- attention mask 允许文本看见全部前置视觉 token；
- 多图之间有明确边界或 segment metadata；
- 同一图不同分辨率下 position/length 不越界；
- grounding 任务是否需要额外 2D/mRoPE 信息。

### 4.5 Transformers 版本

- Qwen config 记录 `4.57.0.dev0`。
- MiniCPM config 记录 `5.6.2`。
- Qwen 本地模型卡明确要求从 Transformers `main` 安装，并同时安装 torchvision/Pillow。
- MiniCPM 本地模型卡要求 `transformers>=5.6`。
- 当前 Codex bundled Python 没有 torch、transformers、safetensors 或 tokenizers。

这不影响静态 probe，但 forward smoke 需要单独、可锁定的训练环境。由于 Qwen 官方要求 `main`，安装时必须记录 Transformers 的精确 Git commit，不能长期依赖浮动分支。该 commit 还需满足 MiniCPM 的 `>=5.6` 行为，再一起锁定 torch、CUDA、torchvision、Pillow 和 attention backend；不能依赖当前桌面运行时。

## 5. 推荐的首个 L0 假设

用于建立可工作的最低基线，不替代 Research 的最终路线：

```text
frozen Qwen model.visual (including merger, output 1024)
  -> trainable connector 1024 -> 1536
  -> frozen MiniCPM5-1B Llama model
```

- 数据：高质量 caption + OCR/短 QA，先单图、短上下文。
- loss：assistant/caption next-token loss；vision placeholder 和 prompt mask。
- processor：复用 Qwen image processor，Composite formatter 使用 MiniCPM chat protocol。
- 训练检查：只允许 connector 参数更新；2-step synthetic overfit 后再上真实数据。
- 后续消融：unfreeze Qwen merger、MiniCPM embeddings/top layers、两层 MLP connector。

## 6. M1/M2 验收检查

1. 静态 probe 验证 pinned repo/revision、config、processor、tensor keys/shapes。
2. 无 `trust_remote_code` 加载 processor、vision model 与 Llama model。
3. 抽取 `model.visual` 后无 missing/unexpected vision keys。
4. processor 产生的 merged visual token count 与 placeholder expansion 完全一致。
5. connector 输出最后一维为 1536，dtype/device 与 LLM embeddings 一致。
6. trainable parameter audit 只包含声明的 connector/tokenizer surgery 参数。
7. 一个 batch forward/backward 的 labels、attention mask、position IDs 和 media trace 可视化。
8. 保存 composite config、processor/token profile 和权重后可离线重新加载。

## 7. 尚待 Research 决策

- L0 使用 linear 还是 2-layer MLP connector。
- 是否完全冻结 Qwen merger；是否训练 MiniCPM embeddings 或 top layers。
- reserved-token smoke 与正式 tokenizer resize 的切换时机。
- alignment 数据组成、token/pixel budget、训练步数和学习率。
- grounding/多图阶段是否引入额外 position encoding。
