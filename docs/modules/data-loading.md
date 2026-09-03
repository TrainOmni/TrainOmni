# Basic data loading runtime

TrainOmni's default data runtime is deliberately small. TaskSpec owns the
semantic pipeline (source, adapter, transforms, ModelIO, supervision, packing and
collation). RunSpec owns machine execution settings. The default runtime is
`torchdata.stateful_dataloader.StatefulDataLoader`; Grain, Energon and DALI are
not dependencies and are not hidden fallback paths.

The supported first path is:

```text
local Parquet row groups or Arrow IPC record batches
  -> physical DP-rank/worker partition
  -> ms-swift row adapter (optional)
  -> transforms and ModelIO inside worker processes
  -> supervision, packing and collation inside workers
  -> prefetched OmniBatch
  -> optional pinned memory and non-blocking device transfer
```

Configure workers in the run, not the task:

```yaml
data_loader:
  num_workers: 4
  prefetch_factor: 2
  persistent_workers: true
  pin_memory: true
  snapshot_every_n_steps: 100
  multiprocessing_context: spawn
  timeout_seconds: 120
```

`num_workers: 0` is the safe default. `prefetch_factor` and
`persistent_workers` require workers. `in_order` is fixed to `true` in this
version because TorchData does not guarantee resumable state with out-of-order
delivery. Worker count and all loader settings are part of RunSpec identity.

Parquet must have at least `world_size * num_workers` non-empty row groups.
Arrow must have that many independently assignable files/record batches. This is
intentional: a worker is assigned physical fragments before opening and decoding
rows, rather than every rank reading the same remote bytes and discarding most of
them. A single Arrow stream is therefore a single physical fragment.

The worker boundary is Windows-spawn safe: builtin readers and immutable data
contracts are pickle-safe, and task-local modules used by worker processing must
also define classes/functions at module scope. Lambdas or locally defined worker
objects are rejected by Python multiprocessing itself.

`multiprocessing_context` defaults explicitly to `spawn`, including Linux; it is
not inherited from Python's platform default. `forkserver`/`fork` must exist on
the host, and `fork` is rejected after CUDA initialization. `timeout_seconds`
is a nonnegative host batch-wait timeout (zero disables it); nonzero requires
workers. These are execution settings, not remedies for every native crash.

The pipeline bootstrap reloads SHA-pinned task-local module definitions before
deserializing worker objects. Immutable nested module configuration is pickle
safe. Custom worker classes/functions still need module-level definitions; do
not retain GPU tensors, open unpicklable handles, lambdas or closures in them.

Worker read failures report rank, worker count, start method, timeout and the
original exception. Caught batch-preparation failures are coordinated before
the next distributed forward so healthy ranks do not enter its collectives.
This does not recover a killed parent rank, CUDA driver hang, or arbitrary NCCL
failure. Process-group timeout and server diagnostics remain necessary.

Each event includes UTC `timestamp` and process-local `monotonic_seconds`.
`data/loader/wait_seconds` remains a cumulative rank-local host wait for backward
compatibility; `wait_seconds_total` is its explicit alias and
`wait_seconds_step` is the current optimizer-step delta, including accumulation.
Neither measures GPU idle time. `host_step_seconds` is host step duration,
not CUDA-kernel profiling. `metric_scopes` states the per-rank data scope and
global DP sum/weighted-mean reduction for objective metrics.

Checkpoint state is collected per worker by StatefulDataLoader and then per rank
by TrainOmni's checkpoint manager. Tests rebuild the loader with fresh worker
processes for two-worker Parquet and Arrow pipelines, including a worker which
had not yet received work when the snapshot was taken.

This layer does not yet implement S3 download/cache coordination, Energon,
WebDataset/TAR, DALI, dynamic cost-based batching or a separate device-prefetch
stream. Those remain later performance work and are not required to use this
basic path.
