# One task, one run, one command

Framework code is installed or placed on `PYTHONPATH`; the task directory and run
directory remain outside Framework. The following is the minimal monolithic VLM
shape. Replace paths and SHA-256 values with immutable local assets.

`task.yaml`:

```yaml
schema_version: 1
name: my-vlm-sft
data:
  source:
    module: data_source:trainomni/jsonl@1
    config:
      path: data/train.jsonl
      sha256: <64-lowercase-hex>
      repeat: true
  transforms:
    - module: sample_transform:trainomni/media@1
      config: {require_sha256: true}
    - module: sample_transform:trainomni/image@1
  model_io:
    module: model_io:trainomni/transformers@1
    config:
      processor_name_or_path: D:/Models/my-vlm
      local_files_only: true
      conversation_mode: required
      require_assistant_mask: true
  supervision:
    module: supervision:trainomni/causal_lm@1
  packer:
    module: packer:trainomni/none@1
  collator:
    module: collator:trainomni/multimodal@1
model:
  implementation:
    module: model:trainomni/monolithic_transformers@1
    config:
      model_name_or_path: D:/Models/my-vlm
      local_files_only: true
  components: {}
objective:
  module: objective:trainomni/causal_lm@1
parameters:
  module: parameter_policy:trainomni/full@1
evaluation:
  data:
    source:
      module: data_source:trainomni/jsonl@1
      config:
        path: data/heldout.jsonl
        sha256: <64-lowercase-hex>
        repeat: true
    transforms:
      - module: sample_transform:trainomni/media@1
        config: {require_sha256: true}
      - module: sample_transform:trainomni/image@1
    model_io:
      module: model_io:trainomni/transformers@1
      config:
        processor_name_or_path: D:/Models/my-vlm
        local_files_only: true
        conversation_mode: required
    supervision: {module: supervision:trainomni/causal_lm@1}
    packer: {module: packer:trainomni/none@1}
    collator: {module: collator:trainomni/multimodal@1}
  evaluators:
    - module: evaluator:trainomni/loss@1
      config: {term: token_ce, metric_name: eval_loss}
exporters:
  - module: exporter:trainomni/transformers@1
```

`run.yaml`:

```yaml
schema_version: 1
name: baseline
seed: 42
deterministic: false
device: cuda:0
precision: bf16_true
attention_kernel: sdpa
max_steps: 1000
per_device_batch_size: 1
gradient_accumulation_steps: 8
max_grad_norm: 1.0
optimizer:
  name: adamw
  learning_rate: 2.0e-5
  weight_decay: 0.01
  foreach: false
scheduler:
  name: cosine
  warmup_steps: 50
activation_checkpointing:
  enabled: true
  components: [model]
  use_reentrant: false
checkpoint:
  directory: outputs/checkpoints
  every_steps: 100
```

Windows PowerShell startup keeps interpreter selection outside the task/run
semantics:

```powershell
$env:TRAINOMNI_PYTHON = 'D:\path\to\cuda-env\Scripts\python.exe'
D:\path\to\Framework\launch\windows\trainomni.ps1 train --task D:\tasks\my-vlm\task.yaml --run D:\runs\baseline\run.yaml
```

Resume, evaluate and export use the same task/run identities:

```text
trainomni train --task task.yaml --run run.yaml --resume outputs/checkpoints/step-00000100
trainomni evaluate --task task.yaml --run run.yaml --checkpoint outputs/checkpoints/step-00001000 --batches 100
trainomni export --task task.yaml --run run.yaml --checkpoint outputs/checkpoints/step-00001000
```

For a composite ViT + connector + LLM, only the `model` section changes to the
composite implementation plus named encoder/connector/fusion/language modules.
Loss, attention policy, data source, ModelIO, parameter policy, evaluator and
exporter are independent extension points described in `../modules/extensions.md`.

For deterministic weighted multi-dataset training, replace `data.source` and add
named child sources. Child module references are part of the task identity, and all
child cursors plus mixture counts are checkpointed:

```yaml
data:
  sources:
    captions:
      module: data_source:trainomni/jsonl@1
      config: {path: data/captions.jsonl, sha256: <64-lowercase-hex>, repeat: true}
    ocr:
      module: data_source:trainomni/jsonl@1
      config: {path: data/ocr.jsonl, sha256: <64-lowercase-hex>, repeat: true}
  source:
    module: data_source:trainomni/mixture@1
    config:
      weights: {captions: 0.7, ocr: 0.3}
      seed: 17
  # transforms/model_io/supervision/packer/collator remain unchanged
```
