# Objective modules: one extension example

Loss semantics live in an Objective module. They do not live in the trainer,
model wrapper, collator, or run configuration.

## Internal boundary

~~~
src/trainomni/modules/objectives/
├── protocol.py                  # ObjectiveModule ABI
├── _ops/                        # stable causal shift / CE / KL / reduction helpers
├── causal_lm/
│   ├── config.py                # only causal-LM loss configuration
│   ├── module.py                # plan() and compute()
│   └── module.toml
├── dense_kd/{config.py,module.py,module.toml}
└── dpo/{config.py,module.py,module.toml}
~~~

An objective implements:

~~~python
requirements() -> ObjectiveRequirements
plan(batch, context) -> ForwardPlan
compute(batch, outputs, context) -> LossBundle
state_dict() -> Mapping
load_state_dict(state) -> None
~~~

plan declares one or more named forwards. The runtime executes those forwards
with device/precision/distributed rules and gives their outputs to compute.
compute owns alignment, masking, FP32 numerical operations, reduction,
normalization and named metrics. It returns a scalar LossBundle.total; the runtime
alone owns backward, accumulation, clipping and optimizer stepping.

Every `LossTerm` must expose its unnormalized `numerator` and its semantic
`denominator`. All named terms in one bundle must use the same denominator and
`LossBundle.total` must equal the weighted sum of term values. During gradient
accumulation the engine backpropagates local numerators, sums denominators across
all microbatches and data-parallel ranks, then applies exactly one global
normalization before clipping and stepping. This makes token-, sample- and
pair-mean objectives correct when local sequence lengths differ. A custom
objective must not hide a second batch-size normalization inside its numerator.

Every named metric must be an `ObjectiveMetric`; raw scalar metrics fail closed.
Counts use `ObjectiveMetric.sum(count)`. Means use
`ObjectiveMetric.weighted_mean(numerator, denominator)`. The engine sums those
states across microbatches and ranks before producing the final value, so a custom
objective must never return a local mean without its weight. `requirements()`
must also declare the fixed `(metric_name, aggregation)` schema in
`ObjectiveRequirements.metric_aggregations`; every computed bundle is checked
against it before backward. For example:

~~~python
from trainomni.contracts import ObjectiveMetric

metrics = {
    "supervised_tokens": ObjectiveMetric.sum(mask.sum()),
    "accuracy": ObjectiveMetric.weighted_mean(correct.sum(), mask.sum()),
}
~~~

Offline caches are not trusted merely because their shard SHA-256 matches. The
builtin KD/DPO objectives require per-sample bindings for expanded `input_ids`,
the full expanded `attention_mask`, absolute supervised positions, target token
IDs, the complete uncollated model-input mapping, branch identity and one producer
identity digest. The complete mapping includes media tensors and auxiliary inputs
such as `image_grid_thw`, `position_ids`, `token_type_ids` and `cache_position`.
Producers must calculate that field with the stable
`trainomni.modules.objectives.ops.model_inputs_digest` helper. Builtin supervision
recomputes the current mapping digest after ModelIO and before collation, so
variable/concatenated media batches remain composable; the Objective compares it
with the producer-bound cache identity before policy forward. Unsupported input
value types and reserved-current-field collisions fail closed. The producer digest
must itself bind the producing model, processor and tokenizer snapshots.
Tensor-cache schemas v2/v3 are rejected; producers must write schema v4.

Supervision modules create labels, target positions, masks or preference branches.
Objectives must still validate them before calculating loss. Loss weights and
normalization are task semantics and therefore belong in objective config inside
task.yaml; learning rate, precision and accumulation belong in run.yaml.

## One-off custom loss

A task can carry an isolated module without modifying or reinstalling Framework.
This uses the generic mechanism in extensions.md; Objective is not privileged:

~~~
<task-root>/modules/objectives/weighted_ce/
├── __init__.py
├── config.py
├── module.py
└── module.toml
~~~

module.toml:

~~~toml
[module]
id = "objective:my_task/weighted_ce@1"
entrypoint = "module:descriptor"
api_version = 1
~~~

config.py:

~~~python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class WeightedCEConfig:
    weight_field: str = "token_weights"
    ignore_index: int = -100
~~~

module.py supplies an ObjectiveModule and a descriptor() returning its typed
ModuleDescriptor. It can import stable contracts and
trainomni.modules.objectives.ops, but it does not receive RunSpec or the global
task dictionary.

The task pins and selects it:

~~~yaml
local_modules:
  - module: objective:my_task/weighted_ce@1
    path: modules/objectives/weighted_ce
    source_sha256: <64 lowercase hex characters>

objective:
  module: objective:my_task/weighted_ce@1
  config:
    weight_field: token_weights
~~~

The caller must separately opt into local code. Before import, Framework resolves
the path under the task root, rejects symlinks, hashes every source-directory file,
compares the declared digest, and validates manifest ID/API/entrypoint. It loads
the directory under a digest-derived namespace without changing sys.path.
Checkpoint identity records both the task digest and local source digest, so source
changes cannot silently resume an older run.

Local modules are trusted executable Python, not a security sandbox. A loss that
will be reused across tasks should graduate into a builtin Framework module with
focused numerical and gradient-routing tests.

A historical pre-fix task-local example is retained at
`D:/Codex/TrainOmni/FrameworkValidation/modules/position_weighted_ce`. It changes
causal CE weighting and label smoothing, then completes BF16 CUDA training,
checkpointing and held-out evaluation without modifying Framework. It predates the
explicit `ObjectiveMetric` contract and requires current-tree revalidation. Its compact
evidence is
`D:/Codex/TrainOmni/FrameworkValidation/extension-validation/custom-objective.json`.
