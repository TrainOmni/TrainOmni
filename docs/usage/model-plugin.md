# Model Plugin Guide

模型插件是 TrainOmni 唯一允许出现模型家族特有语义的地方。目标是接入一个新 VLM 时新增一个外部文件/包，核心零修改。

## Required surface

```python
class MyVLMPlugin:
    manifest = ModelPluginManifest(...)

    def capabilities(self): ...
    def build(self, context: ModelBuildContext) -> ModelBundle: ...
    def component_catalog(self, bundle: ModelBundle) -> ComponentCatalog: ...
    def validate_sample(self, sample, objective): ...
    def encode(self, sample, context) -> EncodedSample: ...
    def collate(self, samples, plan) -> ModelBatch: ...
    def export(self, bundle, checkpoint, target): ...
```

完整可运行示例是 `examples/plugins/tiny_llava.py`；最小真实 PyTorch conformance 示例是 `tests/plugins/torch_toy_vlm_plugin.py`。

## Manifest

Manifest 必须静态描述：

- modalities 和 canonical content blocks；
- sample objectives；
- media 数量、packing、padding-free、generation；
- attention backend、parallelism、engine backend、export formats；
- stable component IDs、checkpoint patterns、dependency constraints；
- 是否要求 remote code。

不要声明没有验证过的能力。Core 会在权重加载前把 recipe requirements 与 manifest 做集合协商。

## Build

`ModelBuildContext` 同时是只读 Mapping，旧插件仍可把它当 model config 使用。新增字段包括：

- `stage_id`；
- `output_dir`；
- `input_artifacts`；
- `mode`（train/evaluate/export）。

返回 `ModelBundle(model, processor, tokenizer, auxiliary_models, metadata)`。Teacher/reference/reward model 放在命名 `auxiliary_models`，不要藏在全局变量。

## Component catalog

每个 parameter name 必须恰好属于一个稳定 component；unclassified、ambiguous、required-empty 都会失败。Component ID 用语义名，例如：

```text
vision_encoder
connector
language_model
audio_encoder
```

Recipe 的 freeze、LR、weight decay、dtype、grad clip、activation checkpointing 和 PEFT 都依赖该边界。

## Encode and loss mask

`encode()` 负责 canonical blocks 到模型 processor ABI 的投影：

- chat template 和 media placeholder；
- image/video/audio decode；
- assistant-only、span-weighted 或 CPT labels；
- `CostVector`；
- `SourceSpan` 和 trace。

必须确保 `sample_id` 不变。`collate()` 只能按给定 `BatchPlan` 顺序形成精确 forward kwargs，返回 `ModelBatch`。

## Export

Core 先把 local checkpoint 或 DCP `model_only` 加载到 bundle，再调用 plugin `export()`。Plugin 负责：

- state-dict key mapping/merge；
- processor/tokenizer/config；
- HF safe serialization、adapter-only 或部署格式；
- 对不支持的格式明确报错。

## Conformance checklist

1. Manifest shape and negative capability checks.
2. Canonical text-only and multimodal fixtures.
3. Assistant/token loss mask inspection.
4. Parameter exact-cover.
5. Tiny real forward/backward.
6. Freeze and optimizer groups.
7. Save, exact resume, evaluation and export.
8. Every declared distributed feature has its own smoke.

加载方式：

```powershell
trainomni --plugin path/to/plugin.py:PLUGIN validate recipe.yaml
```

Recipe 不会自动导入代码，避免数据文件或 YAML 悄悄越过信任边界。
