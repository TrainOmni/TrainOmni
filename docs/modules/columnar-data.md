# Parquet and Arrow data modules

TrainOmni keeps physical storage and semantic sample shape independent:

```text
Parquet/Arrow data_source -> DataRecord -> data_adapter -> OmniSample
```

The builtin modules are:

- `data_source:trainomni/parquet@1`;
- `data_source:trainomni/arrow@1`;
- `data_adapter:trainomni/msswift@1`.

`pyarrow` is the only added runtime dependency. TrainOmni does not import
ms-swift, Hugging Face Datasets, VeOmni, Energon, or Mosaic Streaming.

## Task configuration

```yaml
data:
  source:
    module: data_source:trainomni/parquet@1
    config:
      dataset_id: geometry-sft-v1
      paths:
        - data/train/*.parquet
      columns: [messages, images, videos, audios, id]
      batch_rows: 256
      repeat: true
      dataset_manifest_sha256: <64-lowercase-hex>
  adapter:
    module: data_adapter:trainomni/msswift@1
    config:
      sample_id_column: id
      metadata_columns: []
      media_without_placeholders: prepend
      decode_image_bytes: true
```

The Arrow source uses the same fields and accepts Arrow IPC files and streams.
IPC files are planned by record batch. IPC streams are sequential and use one
physical fragment per file.

## Supported ms-swift row shapes

The adapter accepts, in priority order:

1. `messages` plus optional `images`, `videos`, and `audios`;
2. `texts=[{user, assistant, source}, ...]` plus media;
3. `query`/`response`, optional `system` and `[query, response]` history;
4. flat `text` plus media.

`<image>`, `<video>`, and `<audio>` placeholders consume media in row order.
Partial placeholder/media alignment fails. If a modality has no placeholders,
`media_without_placeholders=prepend` inserts it before the first user text,
matching common ms-swift datasets. Set it to `error` for strict producers.
Hugging Face Image rows represented as `{bytes, path}` are supported; embedded
image bytes are decoded to RGB PIL images lazily during row adaptation.

## Distributed I/O contract

Parquet files are decomposed into row groups before sample reads. The physical
fragments are deterministically balanced by row count across the combined
`(DP rank, DataLoader worker)` topology, so one consumer does not read rows
assigned to another consumer. Arrow IPC files are decomposed into record
batches; IPC streams are assigned at file granularity. A worker-local source is
selected before the first read with:

```python
source.shard(
    rank=distributed_rank,
    world_size=distributed_world_size,
    worker_id=worker_id,
    num_workers=num_workers,
)
```

If physical fragments are fewer than ranks, construction fails and asks the
producer to write more row groups or Arrow shards. TrainOmni does not fall back
to having every rank read the same stream and discard samples.

The source cursor records the semantic dataset snapshot, topology, assigned
fragments, epoch, fragment position, row position, and emitted count. Exact
restore rejects a changed manifest, logical layout, assignment, or topology.
Physical file roots, modification times and sizes are not semantic identity, so
the same snapshot can be staged under a different node-local directory and resume
without weakening the snapshot gate.

`dataset_manifest_sha256` is the content-provenance boundary. The producer owns a
small immutable manifest that lists object versions or shard hashes; TrainOmni
stores its digest in TaskSpec, module lock and source cursor. The readers inspect
row counts/schema/fragment layout but do not repeatedly hash all Parquet/Arrow
payload bytes. Missing manifest identity is explicitly non-reproducible and can
only be used by a checkpoint-disabled diagnostic run. Supplying a stale or false
producer manifest is outside the reader's trust boundary.

For `repeat: false`, a single-process stream flushes the packer at EOF and either
returns the final partial batch (`drop_last: false`, the default) or records and
drops it (`drop_last: true`). Multi-rank training rejects finite or unknown-length
sources before reading because unequal rank exhaustion would otherwise deadlock or
silently change the effective batch. Use an explicitly repeating source until an
equal-step finite sampler is selected.

Current scope is local files. S3/object-store transport, node-local cache,
prefetch, and construction of PyTorch worker processes remain separate
storage/runtime modules. The source has deterministic worker partitioning, but
does not claim asynchronous loading merely because PyArrow can expose an S3
filesystem.

## Real-data validation

`scripts/validate_columnar_datasets.py` reads one real dataset directly as
Parquet, converts a second real dataset into temporary Arrow shards, then reads
both through the builtin sources and ms-swift adapter. It simulates multiple DP
ranks, verifies complete/disjoint sample coverage, decodes embedded images, and
checks user/assistant conversation structure. Temporary Arrow data is deleted at
process exit.
