# TrainOmni Framework

TrainOmni is a module-oriented multimodal training framework. Framework source,
TaskSpec, RunSpec and generated output are separate roots. A model or training
task is composed from typed modules; core code does not branch on model names.

The replacement implementation currently has an executable first vertical slice:

- explicit descriptor/registry/resolver with capability preflight;
- separate, strict task and run schemas with independent SHA-256 identities;
- generic hash-pinned task-local extensions for every module kind;
- canonical sample to encoded/supervised/packed/batched data boundaries;
- deterministic weighted multi-dataset sampling with named child sources and
  exact-resume cursors/counts;
- native PyArrow Parquet row-group and Arrow IPC readers with pre-I/O DP-rank
  and worker partitioning, plus a separate ms-swift-compatible multimodal row
  adapter;
- composite encoder/connector/fusion/language model assembly;
- ordered multi-branch image/video routing through `ModalFeatureSet`, with
  optional branches and an additive path for a later audio encoder;
- model-default semantic attention policy and run-level attention-kernel boundary;
- causal-LM, offline dense-logit KD and offline-reference DPO Objectives with
  FP32 numerics, explicit masks/reductions and multi-forward planning;
- full/component/freeze and native Linear-LoRA parameter policies
  (`train_bias=false`; trained bias fails closed);
- AdamW, scheduler, accumulation, clipping, FP32/BF16/FP16 precision contracts;
- component-scoped activation checkpointing;
- optional `torch.compile` forward execution without compiled checkpoint keys;
- direct PyTorch single/DDP/FSDP2 execution selected by RunSpec, with rank-safe
  data, metrics and checkpoint state;
- optional Linux DeepSpeed ZeRO adapter with explicit fail-closed checkpoint and
  platform boundaries;
- atomic split model/optimizer/scheduler/objective/data/RNG checkpoints;
- explicit checkpoint-disabled diagnostic runs that retain identities/metrics;
- structured run identity, metrics and exact fresh-process resume;
- held-out evaluation with config-addressed receipts, generic/Transformers/LoRA
  export and strict artifact reload.

The automated vertical slice reads separate task/run files, loads five generic
local modules (ModelIO, encoder, connector, fusion and language), trains a tiny
composite VLM, checkpoints at step 2 and resumes to step 4.

Important documents:

- implementation sequence: docs/redesign/implementation-plan.md
- concrete source/task/run tree: docs/architecture/directory-tree.md
- generic extension contract: docs/modules/extensions.md
- Objective/loss example: docs/modules/custom-objective.md
- pinned upstream source ledger: docs/research/upstream-sources.md
- Windows/Linux launch boundary: launch/README.md
- modal branch/fusion ABI: docs/contracts/modal-features.md
- multimodal field collation policies: docs/contracts/collation.md
- canonical flat/chat samples and assistant-mask semantics: docs/contracts/samples.md
- Parquet/Arrow and ms-swift-compatible data path: docs/modules/columnar-data.md
- resumable sequence packing and attention isolation: docs/contracts/sequence-packing.md
- verified support and explicit non-claims: docs/verification/support-matrix.md
- current Windows CUDA development environment: docs/verification/windows-cuda-environment.md
- real VLM five-stage CUDA evidence: docs/verification/real-vlm-five-stage-20260821.md
- real VLM five-route exact-resume evidence: docs/verification/real-vlm-exact-resume-20260821.md
- real VLM custom-objective/attention/mixture/packing/video evidence:
  docs/verification/real-vlm-extension-routes-20260821.md
- distributed architecture, dense/MoE and B200/Ascend boundaries:
  docs/architecture/distributed-execution.md
- seven-route medium-data loss and update evidence:
  docs/verification/real-vlm-medium-data-20260822.md
- one task / one run / one command: docs/usage/quickstart.md

Basic command boundary:

~~~text
trainomni inspect --task <task.yaml> [--allow-local-code]
trainomni train --task <task.yaml> --run <run.yaml> [--allow-local-code]
trainomni train --task <task.yaml> --run <run.yaml> --resume <step-directory>
trainomni evaluate --task <task.yaml> --run <run.yaml> --checkpoint <step-directory> --batches N
trainomni export --task <task.yaml> --run <run.yaml> --checkpoint <step-directory>
trainomni module-digest <module-directory>
~~~

Platform scripts are intentionally separate from the Python CLI. They require
`TRAINOMNI_PYTHON` to be the absolute interpreter path and only forward arguments:

~~~text
launch/windows/trainomni.ps1 <trainomni arguments...>
launch/linux/trainomni.sh <trainomni arguments...>
~~~

They never choose or install PyTorch, activate an environment, or own distributed
topology. Separate distributed wrappers launch one-rank Windows probes or
`torch.distributed.run` on Linux; RunSpec still owns backend/world-size semantics.

Local modules are trusted Python and require explicit opt-in. They are loaded in a
digest-derived namespace without changing sys.path; this is provenance and module
isolation, not a security sandbox.

The current Windows development interpreter is `Framework/.venv/Scripts/python.exe`.
It contains CUDA Torch and is excluded from Git; launchers still require an explicit
`TRAINOMNI_PYTHON`, so a checkout never silently chooses an environment. Production
dependency locking remains separate release work. Real-checkpoint single-GPU
compatibility is now validated for the documented five-stage chain and its five
fresh-process exact-resume routes. Direct DDP and FSDP2 have real one-rank CUDA
execution/checkpoint/resume evidence; actual multi-rank Linux/NCCL and
Ascend/HCCL remain server gates.
The pre-redesign implementation remains archived and is not part of this source tree.
