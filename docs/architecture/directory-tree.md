# Concrete directory tree

## Framework source

```text
Framework/
├── pyproject.toml
├── README.md
├── STATUS.md
├── launch/                       # shell/platform boundary only
│   ├── README.md                  # common semantic launch contract
│   ├── windows/
│   │   ├── trainomni.ps1         # explicit Python + exact CLI forwarding
│   │   └── distributed/torchrun.ps1 # certified one-rank Windows probe
│   └── linux/
│       ├── trainomni.sh          # POSIX exec + exact CLI forwarding
│       └── distributed/torchrun.sh # torch.distributed.run adapter
├── src/trainomni/
│   ├── api/                       # stable Python-facing operations
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── export.py
│   │   └── inspect.py
│   ├── core/                      # module kernel; imports no concrete module
│   │   ├── module.py
│   │   ├── registry.py
│   │   ├── resolver.py
│   │   ├── capability.py
│   │   ├── context.py
│   │   └── errors.py
│   ├── specs/                     # Framework/Task/Run configuration boundary
│   │   ├── task.py
│   │   ├── run.py
│   │   ├── loading.py
│   │   └── digest.py
│   ├── contracts/                 # stable cross-module value objects only
│   │   ├── sample.py
│   │   ├── data.py                 # storage-neutral physical row contract
│   │   ├── features.py
│   │   ├── batch.py
│   │   ├── forward.py
│   │   ├── loss.py
│   │   ├── distribution.py       # model-declared physical topology hints
│   │   ├── state.py
│   │   └── artifact.py
│   ├── catalog/                   # builtin module descriptors, no construction
│   │   └── builtin.py
│   ├── assembly/                  # resolves specs into one executable assembly
│   │   ├── task_builder.py
│   │   ├── data_builder.py
│   │   ├── model_builder.py
│   │   └── preflight.py
│   ├── modules/
│   │   ├── data/
│   │   │   ├── adapters/
│   │   │   │   ├── protocol.py
│   │   │   │   ├── binding.py     # source -> semantic adapter composition
│   │   │   │   └── msswift/{config.py,module.py}
│   │   │   ├── sources/
│   │   │   │   ├── protocol.py
│   │   │   │   ├── _columnar.py   # physical-fragment planning and cursor
│   │   │   │   ├── memory/{config.py,module.py}
│   │   │   │   ├── jsonl/{config.py,module.py}
│   │   │   │   ├── parquet/{config.py,module.py}
│   │   │   │   ├── arrow/{config.py,module.py}
│   │   │   │   └── mixture/{config.py,module.py}
│   │   │   ├── transforms/
│   │   │   │   ├── protocol.py
│   │   │   │   ├── media/{config.py,module.py}
│   │   │   │   ├── image/{config.py,module.py}
│   │   │   │   ├── video/{config.py,module.py}
│   │   │   │   └── tensor_cache/{config.py,module.py}
│   │   │   ├── model_io/
│   │   │   │   ├── protocol.py
│   │   │   │   └── transformers/{config.py,module.py}
│   │   │   ├── supervision/
│   │   │   │   ├── protocol.py
│   │   │   │   ├── causal_lm/{config.py,module.py}
│   │   │   │   ├── dense_kd/{config.py,module.py}
│   │   │   │   └── preference/{config.py,module.py}
│   │   │   ├── packing/
│   │   │   │   ├── protocol.py
│   │   │   │   ├── none/{config.py,module.py}
│   │   │   │   └── sequence/{config.py,module.py}
│   │   │   └── collation/
│   │   │       ├── protocol.py
│   │   │       └── multimodal/{config.py,module.py}
│   │   ├── model/
│   │   │   ├── encoders/
│   │   │   │   ├── protocol.py
│   │   │   │   ├── transformers_vision/{config.py,module.py}
│   │   │   │   └── transformers_video/{config.py,module.py}
│   │   │   ├── connectors/
│   │   │   │   ├── protocol.py
│   │   │   │   ├── linear/{config.py,module.py}
│   │   │   │   └── mlp/{config.py,module.py}
│   │   │   ├── fusions/
│   │   │   │   ├── protocol.py
│   │   │   │   ├── token_replace/{config.py,module.py}
│   │   │   │   ├── prefix/{config.py,module.py}
│   │   │   │   └── cross_attention/{config.py,module.py}
│   │   │   ├── language/
│   │   │   │   ├── protocol.py
│   │   │   │   └── transformers_causal_lm/{config.py,module.py}
│   │   │   ├── models/
│   │   │   │   ├── protocol.py
│   │   │   │   ├── monolithic/{config.py,module.py}
│   │   │   │   └── composite/{config.py,module.py}
│   │   │   └── attention/
│   │   │       ├── protocol.py
│   │   │       └── policies/{config.py,module.py}
│   │   ├── objectives/
│   │   │   ├── protocol.py
│   │   │   ├── causal_lm/{config.py,module.py}
│   │   │   ├── dense_kd/{config.py,module.py}
│   │   │   └── dpo/{config.py,module.py}
│   │   ├── parameters/
│   │   │   ├── protocol.py
│   │   │   ├── full/{config.py,module.py}
│   │   │   ├── component/{config.py,module.py}
│   │   │   ├── freeze/{config.py,module.py}
│   │   │   └── lora/{config.py,module.py}
│   │   ├── evaluation/
│   │   │   ├── protocol.py
│   │   │   ├── loss/{config.py,module.py}
│   │   │   └── task_metrics/{config.py,module.py}
│   │   └── export/
│   │       ├── protocol.py
│   │       ├── transformers/{config.py,module.py}
│   │       ├── lora_adapter/{config.py,module.py}
│   │       └── safetensors/{config.py,module.py}
│   ├── runtime/                   # RunSpec execution; no task semantics
│   │   ├── loop/{engine.py,step.py}
│   │   ├── device/context.py
│   │   ├── execution/
│   │   │   ├── {process.py,protocol.py,factory.py,data.py,selection.py}
│   │   │   ├── torch_backends.py # direct single/DDP/FSDP2
│   │   │   ├── fsdp_state.py     # upstream portable state bridge
│   │   │   └── deepspeed_backend.py # optional thin Linux adapter
│   │   ├── kernels/{activation_checkpointing.py,compilation.py}
│   │   ├── kernels/attention/selection.py
│   │   ├── optimization/{optimizer.py,scheduler.py,gradients.py}
│   │   ├── checkpoint/{manager.py,manifest.py,resume.py}
│   │   └── observability/{events.py,resources.py}
│   ├── artifacts/{manifest.py,lineage.py}
│   └── cli/{main.py,commands.py}
├── tests/
│   ├── contracts/
│   ├── unit/{core,specs,catalog,assembly,modules,runtime}/
│   ├── integration/{vertical_slice,objectives,distributed}/
│   ├── resume/
│   ├── distributed/
│   ├── provenance/
│   └── fixtures/{data,tasks,runs,models}/
└── docs/{architecture,contracts,modules,usage,verification,redesign,research}/
```

Every concrete module directory has exactly `config.py` and `module.py`. It may
import its category protocol and stable contracts, but cannot import a sibling
implementation or receive the global task/run configuration.

Large upstream source checkouts are intentionally outside this tree. Their pinned
commit ledger is `docs/research/upstream-sources.md`; Framework never imports or
executes those checkouts.

## Consumer task tree

Concrete tasks are not stored under Framework. Each test group has an independent
task directory, strongly named `YYYYMMDD_<specific_task>`. Existing tasks and their
results are retained when starting a new group or changing model architecture.
The tree below describes one task, not a singleton shared by all experiments.
See [task organization](../usage/task-organization.md) for a multi-task example.

```text
<task-root>/
├── task.yaml                    # what to learn; references registered modules
├── modules/                     # optional task-owned, hash-pinned code modules
│   └── objectives/
│       └── custom_loss/{__init__.py,config.py,module.py,module.toml}
├── data/
│   └── datasets.yaml            # immutable source identities and split semantics
├── assets/
│   └── models.yaml              # checkpoint/tokenizer/processor identities
├── eval/
│   └── metrics.yaml
└── tests/
    └── test_task_preflight.py
```

Reusable behavior belongs in a Framework builtin module. Task-specific behavior
may instead live under modules/, but it is loaded only when the task declares its
module ID, relative path and exact source-tree SHA-256 and the caller explicitly
allows local code. Loading uses an isolated synthetic package namespace and never
mutates sys.path. This is provenance isolation, not a security sandbox.

## Run and output tree

```text
<run-root>/
└── run.yaml                     # how one attempt executes

<output-root>/<run-id>/
├── resolved/
│   ├── task.resolved.yaml
│   ├── run.resolved.yaml
│   └── modules.lock.yaml
├── checkpoints/step-*/
├── metrics/events.jsonl
├── artifacts/
└── run-manifest.json
```

Framework source, task semantics, run controls, and generated state therefore have
four distinct responsibilities and non-overlapping write targets. Run definitions
and outputs may be separate subdirectories within their owning task directory
(for example `runs/` and `outputs/<run-id>/`), or live in external roots. One task
may have several runs; independent runs must not reuse an output root. A changed
task must not overwrite the configuration or evidence of an older experiment.

Platform startup is a fifth, thin boundary: it selects an already-created Python
environment and forwards CLI arguments. It cannot interpret or mutate any of the
four roots. Windows and Linux shell mechanics stay in separate directories while
the Python command and TaskSpec/RunSpec contracts remain identical.
