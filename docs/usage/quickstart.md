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
      asset_manifest_sha256: <64-lowercase-hex>
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
      asset_manifest_sha256: <64-lowercase-hex>
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
        asset_manifest_sha256: <64-lowercase-hex>
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
data_loader:
  num_workers: 4
  prefetch_factor: 2
  persistent_workers: true
  pin_memory: true
  snapshot_every_n_steps: 100
activation_checkpointing:
  enabled: true
  components: [model]
  use_reentrant: false
checkpoint:
  directory: outputs/checkpoints
  every_steps: 100
```

Finite streams retain EOF across loader checkpoint restoration; they do not
implicitly start a new epoch. Multiple workers can each emit a partial final
batch when `drop_last=false`. Loader state schema v2 records this terminal state;
legacy v1 loader payloads are accepted only when their pinned TorchData finished
marker is valid. This does not bypass checkpoint-level source/module identity
validation or make a checkpoint from changed Framework code resumable.

Execution is a RunSpec concern. The task is unchanged when moving between direct
PyTorch backends:

```yaml
execution:
  backend: torch_ddp       # single | torch_ddp | torch_fsdp2 | deepspeed
  expected_world_size: 8
  process_group_backend: nccl
  ddp:
    find_unused_parameters: false
    static_graph: true
```

For FSDP2, the model plugin must return valid `DistributionHints.fsdp_units` and
the backend is `torch_fsdp2`. DeepSpeed is an optional Linux-only execution probe;
its native ZeRO checkpoint bridge is not complete, so checkpoint-enabled
DeepSpeed runs fail closed. See `../architecture/distributed-execution.md`.

A bounded training-only diagnostic can explicitly disable all checkpoint writes:

```yaml
checkpoint:
  enabled: false
  directory: outputs/checkpoints   # physical output location; not run identity
  every_steps: 100
```

This mode cannot resume, evaluate, export or save explicitly. It is useful for
loss/update/resource gates that should not duplicate multi-gigabyte payloads.

Windows PowerShell startup keeps interpreter selection outside the task/run
semantics:

```powershell
$env:TRAINOMNI_PYTHON = 'D:\path\to\cuda-env\Scripts\python.exe'
D:\path\to\Framework\launch\windows\trainomni.ps1 train --task D:\tasks\my-vlm\task.yaml --run D:\runs\baseline\run.yaml
```

Full-state resume requires the same semantic TaskSpec and RunSpec. The physical
`checkpoint.directory` may move without changing RunSpec identity; all other run
fields remain exact-resume inputs. Model-only evaluate/export validates checkpoint
task/module/framework and file integrity, but may use a different execution
device, precision, batch size and output directory:

The asset-manifest digest is a producer-owned small manifest identity that binds
the local Transformers payload without re-hashing multi-gigabyte weights at every
launch. A remote model may instead use an immutable 40--64 character lowercase
commit `revision`. An unpinned Transformers or Parquet/Arrow asset is explicitly
non-reproducible: it may run only with `checkpoint.enabled=false` and cannot claim
exact resume.

Pre-fix checkpoints use the old Framework version/provenance and are deliberately
rejected by the corrected exact-resume path. Checkpoint relocation applies to
checkpoints written by the corrected implementation; changing any semantic run,
task, module, asset or dataset identity still fails.

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
