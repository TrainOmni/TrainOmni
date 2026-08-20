# Canonical Multimodal Data Contract v0.1

> 历史协议说明：可执行 contract 以 `src/trainomni/data/`、当前 fixtures 和 [support matrix v1](support-matrix-v1.md) 为准。

- 状态：Draft protocol；M0 Python typed models、validator、hash 和 fixtures 已实现
- Schema：[`sample-v0.1.schema.json`](./sample-v0.1.schema.json)
- Runtime：[`../../src/trainomni/data/`](../../src/trainomni/data/)
- 核心原则：保存任务语义，不保存模型专属 token 或 processor tensor

## 1. 数据管线边界

```text
source record
  -> Reader / Importer
  -> Canonical Sample
  -> deterministic semantic transforms
  -> Model Family Formatter + Processor
  -> Encoded Sample (tokens, media tensors, labels, cost)
  -> stateful Packer / Batch Planner
  -> model-specific Collator
  -> Batch
```

各层职责：

- Reader 只解决外部列名、文件格式和 URI，不插入模型 special token。
- Canonical Sample 表达 conversation/document、media、annotation、loss intent 和 objective。
- Semantic Transform 做 crop/resize/frame sampling 时同步更新 bbox/point，并写 transform trace。
- Formatter/Processor 属于 Model Family Plugin，负责 chat template、media token、坐标序列化和 token loss mask。
- Packer 属于 framework data runtime，维护 pack plan、segment isolation 和可恢复状态。
- Collator 属于模型插件，生成该模型 forward 真正需要的 tensor 键。

## 2. 顶层结构

| 字段 | 含义 |
|---|---|
| `schema_version` | 固定为 `trainomni.sample.v0.1` |
| `id` | 数据集内稳定且非空的样本 ID |
| `objective` | `cpt`、`sft`、`preference` 或 `prompt_only` |
| `messages` | CPT document、对话 prompt 或完整 SFT 序列 |
| `assets` | 由稳定 `asset_id` 引用的 image/video/audio |
| `tools` | 可选工具 JSON schema 列表 |
| `preference` | preference objective 的 chosen/rejected continuations |
| `rollout` | prompt-only objective 的 verifier/environment 信息 |
| `metadata` | source、license、language、quality、tags 等 provenance |

`messages` 使用 content block 数组，而不是在字符串中嵌 `<image>`。同一 asset 可以在多轮、多候选或多个 annotation 中复用。

## 3. Content block

v0.1 定义：

- `text`：原始文本。
- `media`：用 `asset_id` 在当前位置插入 media。
- `bbox`：`xyxy` + `pixel` 或 `norm_0_1` 坐标系。
- `point`：`xy` + 坐标系。
- `json`：结构化模型输出或输入。
- `tool_call`：结构化工具名和 arguments。
- `tool_result`：工具结果，可通过后续 `media` block 返回图像。

message 和 block 都可带非负 `loss_weight`。继承规则从外到内：

1. block 显式值优先；
2. 否则继承 message 值；
3. 否则由 objective + role 的默认 loss policy 决定。

建议默认值：

- `cpt`：`document` 全部文本 target weight = 1，media placeholder 本身由 adapter 决定。
- `sft`：`assistant` target weight = 1，其他 role = 0。
- preference：candidate assistant completion = 1，共享 prompt = 0。
- prompt-only：训练 labels 不由静态样本产生。

## 4. 示例

### 4.1 Multi-image SFT

```json
{
  "schema_version": "trainomni.sample.v0.1",
  "id": "compare-000001",
  "objective": "sft",
  "assets": [
    {"id": "left", "modality": "image", "uri": "images/cat.jpg"},
    {"id": "right", "modality": "image", "uri": "images/dog.jpg"}
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "media", "asset_id": "left"},
        {"type": "media", "asset_id": "right"},
        {"type": "text", "text": "比较这两张图。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "左图是猫，右图是狗。"}
      ]
    }
  ],
  "metadata": {
    "source": "internal-demo",
    "split": "train",
    "language": "zh"
  }
}
```

### 4.2 Grounding SFT

```json
{
  "schema_version": "trainomni.sample.v0.1",
  "id": "ground-000001",
  "objective": "sft",
  "assets": [
    {
      "id": "scene",
      "modality": "image",
      "uri": "images/scene.jpg",
      "width": 1280,
      "height": 720
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "media", "asset_id": "scene"},
        {"type": "text", "text": "指出红色汽车的位置。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "红色汽车在"},
        {
          "type": "bbox",
          "asset_id": "scene",
          "xyxy": [244, 301, 612, 633],
          "coordinate_space": "pixel",
          "label": "red car"
        }
      ]
    }
  ]
}
```

Qwen、InternVL 或其他模型所需的 bbox special token 与归一化方式由 adapter 生成，源样本不改写。

### 4.3 Multimodal preference

```json
{
  "schema_version": "trainomni.sample.v0.1",
  "id": "pref-000001",
  "objective": "preference",
  "assets": [
    {"id": "photo", "modality": "image", "uri": "images/animal.jpg"}
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "media", "asset_id": "photo"},
        {"type": "text", "text": "图里是什么动物？"}
      ]
    }
  ],
  "preference": {
    "chosen": {
      "messages": [
        {"role": "assistant", "content": [{"type": "text", "text": "一只橘猫。"}]}
      ]
    },
    "rejected": {
      "messages": [
        {"role": "assistant", "content": [{"type": "text", "text": "一只小狗。"}]}
      ]
    },
    "judge": "human"
  }
}
```

若 chosen/rejected 使用不同 media，两者可引用同一顶层 `assets` 中的不同 `asset_id`；不需要复制完整 prompt。

### 4.4 Prompt-only RLVR

```json
{
  "schema_version": "trainomni.sample.v0.1",
  "id": "rlvr-000001",
  "objective": "prompt_only",
  "assets": [
    {"id": "chart", "modality": "image", "uri": "charts/000001.png"}
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "media", "asset_id": "chart"},
        {"type": "text", "text": "读图并计算 2024 相比 2023 的增长率。"}
      ]
    }
  ],
  "rollout": {
    "reference_answer": 0.125,
    "max_completion_tokens": 512,
    "verifiers": [
      {
        "type": "numeric_tolerance",
        "weight": 1.0,
        "spec": {"absolute_tolerance": 0.001}
      }
    ]
  }
}
```

## 5. JSON Schema 与语义校验的边界

JSON Schema 负责结构类型；Python semantic validator 还必须检查：

- `assets[].id` 唯一，所有 `asset_id` 都存在。
- media modality 与 model capability 相容。
- pixel bbox/point 需要已知 width/height，且坐标合法；`norm_0_1` 在范围内。
- `cpt` 至少有一个 `document` 或可训练 target；`sft` 至少有一个非零 loss target。
- preference chosen/rejected 非空、引用合法且不是完全相同的 canonical hash。
- prompt-only verifier 可在选定 reward backend 注册。
- 工具 call/result 的 `call_id` 配对。
- URI scheme 允许且路径解析后不越过配置的数据根目录。
- license、source、split 等字段是否满足当前 data policy。

## 6. URI 与 asset 解析

- 相对 URI 以 DatasetSpec 的 `media_root` 为基准，不以当前工作目录为基准。
- 允许的 scheme 由 reader 配置白名单控制，例如 `file`、`https`、`hf`、`s3`。
- manifest 中不建议内嵌大体积 base64；导入器可以接受，但 canonical cache 应外置 asset。
- `sha256` 若存在必须在首次读取或离线 audit 时验证，并将结果写入 asset cache metadata。

## 7. Encoded Sample（非持久协议）

Encoder 输出至少包含：

```text
sample_id
model_inputs: dict[str, Tensor | nested value]
labels / token_weights
text_token_count
visual_token_estimate
media_cost
segment metadata
trace: template + token spans + source block mapping
```

Encoded Sample 是 adapter/processor 版本相关的缓存对象，必须携带：canonical sample hash、adapter version、processor fingerprint、transform fingerprint 和 max length/pixel policy。它不能替代原始 canonical sample。

## 8. Packer/Batch Planner 状态

为了精确恢复，state dict 至少保存：

- input dataset/shard cursor；
- sampler/mixer RNG 与 draw counters；
- 当前未完成 pack 的 sample ids、length/cost 和 buffer order；
- epoch、worker/rank 派生 seed；
- batch/token/pixel budget；
- packing algorithm/version。

恢复时如 dataset fingerprint、world size 兼容策略或 packer version 不匹配，默认 hard fail；允许显式 `weights-only resume`，但不能伪装成 exact resume。

## 9. 版本策略

- additive optional field：保持 `v0.1`。
- 修改字段语义、默认 loss policy、坐标定义或 objective 结构：升级版本并提供 migration。
- Reader 接受历史版本，但内部训练只消费迁移后的当前版本。
- 每个训练 run 记录 schema version 与 migration chain。
