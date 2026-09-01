# Distributed execution boundary

TrainOmni does not implement collective algorithms. It selects an upstream
execution backend through `RunSpec.execution`, supplies model-declared physical
boundaries, and keeps task/data/objective/artifact identity outside the backend.

## Backend matrix

| Backend | Upstream owner | TrainOmni adapter | Current checkpoint contract | Current evidence |
| --- | --- | --- | --- | --- |
| `single` | PyTorch | direct | atomic full training state | real VLM CUDA lifecycle |
| `torch_ddp` | PyTorch `DistributedDataParallel` | thin | atomic full model/optimizer plus per-rank runtime state | real process group, CUDA world size 1; real VLM train/evaluate and tiny exact resume |
| `torch_fsdp2` | PyTorch `fully_shard` + distributed state-dict APIs | thin | portable full state gathered through the upstream state-dict API | real process group, CUDA world size 1; real VLM train/evaluate and tiny exact resume |
| `deepspeed` | DeepSpeed engine/ZeRO | thin optional | **not bridged**; checkpoint-enabled runs fail before training | config mapping tested; native Windows rejected; Linux execution remains a server gate |

DDP/FSDP2/DeepSpeed are selected by RunSpec. Launcher topology is supplied by
the separate platform wrappers. There is no fallback between backends.

PyTorch FSDP2 turns parameters into DTensors and owns all-gather/reduce-scatter.
TrainOmni calls the upstream distributed state-dict API to obtain/load portable
full model and optimizer state; it does not reconstruct shards itself. DeepSpeed
ZeRO similarly owns partitioning, but its native rank-sharded checkpoint layout
has not yet been reconciled with TrainOmni's atomic identity and model-only-load
contract. A DeepSpeed run must therefore use `checkpoint.enabled=false` until the
Linux multi-rank checkpoint gate is implemented.

## Process, data and observability contract

- `RANK`, `LOCAL_RANK` and `WORLD_SIZE` are launcher facts. Partial or
  contradictory environments fail closed.
- The RunSpec can pin `expected_world_size` and process-group backend.
- CUDA multi-rank auto-selection requires NCCL. HCCL is an explicit NPU platform
  boundary. Native Windows is certified only for a one-rank Gloo probe.
- Every rank receives a deterministic disjoint source stream below transforms,
  packing and collation. Rank, world size, source cursor and local sample count
  are exact-resume state.
- Loss terms reduce numerator and denominator globally. Objective metrics declare
  either a global sum or weighted-mean numerator/denominator and reduce that state
  across microbatches/ranks. Unknown scalar semantics fail closed. Peak memory uses
  a maximum reduction. Data counters are
  retained per rank in `data_metrics_by_rank`; TrainOmni does not guess whether a
  task-local counter should be summed or averaged.
- Only rank zero writes shared run receipts. Pure rank-local checkpoint state is
  captured before an all-rank outcome exchange; any rank failure prevents gather
  and filesystem work. Every rank then participates in state collection. FSDP2
  state-adapter capture is an upstream collective boundary: failures after it
  returns are coordinated, while a failure inside a stuck upstream collective is
  bounded by the configured process-group timeout.

`checkpoint.enabled=false` is an explicit training-only diagnostic mode. It still
materializes immutable task/run/module/parameter identities and structured
metrics, but neither periodic nor final checkpoints are written, and an explicit
save call fails. It is not resumable and cannot be passed to evaluate/export.

## Dense and MoE models

Data parallelism and expert parallelism are different dimensions:

- DDP can train a model containing experts by replicating every expert on every
  rank. This is correct ordinary data parallelism but provides no expert-memory
  scaling.
- Generic FSDP2 rejects model hints containing experts or routers. Sharding a
  module tree does not construct expert process groups, token dispatch, capacity
  rules or router auxiliary losses.
- Generic DeepSpeed ZeRO also rejects expert/router hints. DeepSpeed MoE/AutoEP
  would require a dedicated backend adapter that owns expert groups and declares
  its checkpoint identity.
- Explicit replicated modules and cross-unit tied parameters currently fail the
  generic FSDP2 path. Silently sharding them would violate the model contract.

`DistributionHints` is therefore structural metadata, not a parallelism
strategy. It exposes FSDP units, experts, routers, replicated modules and tied
parameters so a backend can accept the topology or reject it before forward.

## NVIDIA B200 and Ascend

B200 does not require a separate TrainOmni model/objective path. The baseline
server gate is Linux + CUDA + NCCL using BF16 DDP/FSDP2. FP8/MXFP8/NVFP4 are not
current TrainOmni precisions; adding them should be an upstream Transformer Engine
adapter and a distinct precision capability, not a dtype string mapped to BF16.

Ascend does require a platform adapter: a compatible `torch_npu`/PyTorch pair,
NPU device placement, HCCL process groups, NPU memory metrics, supported attention
kernels and a Linux launcher image. None is silently inferred from CUDA. The
model/data/objective/module contracts remain unchanged, but DDP/FSDP2 parity and
checkpoint portability require a real multi-NPU gate.

## Server acceptance gates

The current workstation cannot satisfy these gates. On the target servers, run:

1. two-rank DDP uninterrupted versus resume, checking disjoint samples, global
   loss reductions, per-rank metrics and final model equality;
2. two-rank FSDP2 uninterrupted versus resume plus model-only evaluation from the
   portable full-state checkpoint;
3. one- and multi-node topology mismatch/corruption negative tests;
4. Linux DeepSpeed ZeRO-2 and ZeRO-3 backward/step, followed by implementation and
   validation of the native checkpoint bridge before enabling checkpoints;
5. B200 BF16 first; FP8 only after a Transformer Engine capability is explicit;
6. Ascend HCCL only after the NPU device/metrics/kernel adapter exists.

Primary upstream references: PyTorch
[`fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html),
[FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html),
[Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html),
and [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html);
DeepSpeed [ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html) and
[checkpointing](https://deepspeed.readthedocs.io/en/stable/model-checkpointing.html);
NVIDIA [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/).
