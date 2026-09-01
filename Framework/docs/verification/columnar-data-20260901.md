# Real Parquet and Arrow data verification — 2026-09-01

## Scope

The replacement data path was exercised with two existing multimodal datasets,
not generated semantic rows:

1. InterGPS: 1,280 rows, 13 Parquet row groups, embedded PNG images and
   `{user, assistant, source}` text pairs;
2. Diagram Image-to-Text: 300 rows, 3 Parquet row groups, embedded PNG images
   and the same text-pair representation.

The first dataset was consumed directly as Parquet. The second was converted
row-group-by-row-group into temporary Arrow IPC shards and consumed through the
Arrow source. The temporary files were removed automatically.

## Command

```powershell
$env:PYTHONPATH = (Resolve-Path src)
& .\.venv\Scripts\python.exe scripts\validate_columnar_datasets.py `
  --parquet-dataset D:\Codex\TrainOmni\Downloads\datasets\vlm-minimal-v1\raw\intergps\train-00000-of-00001-d12182a583de4589.parquet `
  --arrow-source-parquet D:\Codex\TrainOmni\Downloads\datasets\vlm-minimal-v1\raw\diagram_image_to_text\train-00000-of-00001-37a8de19cc7bc987.parquet `
  --ranks 2
```

## Result

| Dataset | Read format | Rows | Embedded images decoded | Duplicate IDs | Simulated rank coverage |
| --- | --- | ---: | ---: | ---: | --- |
| InterGPS | Parquet | 1,280 | 1,280 | 0 | 680 + 600 |
| Diagram Image-to-Text | Arrow IPC | 300 | 300 | 0 | 164 + 136 |

Every emitted sample had ordered `user` and `assistant` messages and an image
block. The union of rank-local samples matched the physical dataset row count,
and rank-local sample IDs were disjoint.

Observed local smoke throughput, including embedded PNG decoding and Python
sample construction, was about 1,493 samples/s for InterGPS and 368 samples/s
for Diagram Image-to-Text. These numbers are machine- and image-dependent and
are evidence of functional progress only, not a performance claim.

## Automated regression

- targeted columnar/spec tests: 11 passed;
- complete Framework suite: 111 passed, 1 skipped;
- Ruff: clean.

The tests cover Parquet row-group rank/worker assignment, Arrow IPC file and
stream reading, bounded record batches, exact source restore, topology
rejection, embedded-image decoding, complete/disjoint rank and worker coverage,
fail-closed media placeholder alignment, and the assembled source -> adapter ->
model-I/O -> supervision -> packing -> collation path.

## Claim boundary

This verifies local Parquet/Arrow ingestion and deterministic source-level
DP/worker partitioning. It does not claim S3 transport, node-local caching,
asynchronous DataLoader execution, Linux multi-process execution, or Ascend
multi-node throughput. Those are separate storage and runtime gates.
