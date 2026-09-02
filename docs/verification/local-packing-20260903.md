# Local-data packing verification — 2026-09-03

## Result and scope

The current Framework source tree passed an actual single-GPU image/text packing
route, not just task inspection. This verifies engineering correctness for the
explicitly packing-aware Qwen3.5 Vision + MiniCPM5-1B task-local model adapter.
It is not a throughput benchmark or a model-quality result.

Tested builtin Python source SHA-256:
`8bd579a03114b771777dfd1da34cf857fe2c979464286c920e979c1578c7757a`.

The current tree includes the separately authorized TorchData loader work. It is
not identical to the correctness-only branch based on upstream v2; their test
counts and capability claims must not be interchanged.

## Real local data

- `Downloads/datasets/vlm-minimal-v1/raw/intergps/`: all **1,280** Parquet
  records passed the Framework reader and ms-swift adapter, including image
  decoding, message-role validation and duplicate-ID checks.
- `Downloads/datasets/vlm-minimal-v1/raw/diagram_image_to_text/`: all **300**
  records passed through temporary Arrow IPC shards and the same adapter.
- Two logical rank partitions covered all records without duplicate IDs:
  **680/600** for InterGPS and **164/136** for diagram-image-to-text. This was
  a sequential reader check, not a multi-GPU or network-storage benchmark.
- No original dataset or media file was modified. Temporary Arrow files were
  cleaned up by the existing validation script.

## GPU packing route

Read-only task: `FrameworkValidation/extension-packing.task.json`.
Read-only data: `FrameworkValidation/data/packing.jsonl`, two short conversations
with local geometry images. These are engineering fixtures, not the full
InterGPS training set.

| Check | Observed result |
| --- | --- |
| Original sequence lengths | 19 and 20 text tokens |
| Packing | One fixed-width 96-token tensor, 39 valid text tokens |
| Cross-segment attention entries | 0 in both directions |
| Second segment boundary label | -100, excluded from causal loss |
| Incompatible default attention | Rejected by capability preflight |
| Actual execution | CUDA device 0, true BF16, eager attention, two optimizer steps |
| Final training token CE | 8.221441650390625 |
| Model-only reload and same-fixture evaluation | Passed; token CE 8.241866302490234 |
| Actual connector update | 3 tensors changed; 4,433 sampled elements changed at step 2 |
| Peak CUDA allocated / reserved | 4,225,608,704 / 4,374,659,072 bytes |

The evaluation uses the same fixture as training. Although the reused validation
runner calls its output field `held_out_evaluation`, this is **not held-out
quality evidence**.

The route used the default loader with `num_workers=0`. Separate automated
Parquet and Arrow tests exercised two Windows-spawn workers, packing, complete
finite-stream coverage and restored continuation. Combining the real GPU task
with multi-worker loading was not tested by this route.

The existing validation runner was loaded from
`FrameworkValidation/verify_extension_routes.py`; its receipt destination and
temporary root were redirected into Framework. Receipt:
`Framework/.cache/local-packing-validation-20260903-final/sequence-packing.json`.
The large generated checkpoint was removed automatically after successful
model-only reload and evaluation; no previously accepted artifact was replaced.

## Automated checks

- `tests/unit/modules/data/test_sequence_packer.py`: **6 passed**.
- `tests/unit/runtime/test_kernel_policies.py` plus
  `tests/unit/modules/data/test_columnar_sources.py`: **17 passed**, with four
  upstream TorchData deprecation warnings.
- Final full suite plus the coordinator's nine independent reproductions:
  **307 passed, 1 expected Windows skip** (298 Framework cases plus 9 independent
  cases). Ruff and diff-check passed.
- Compileall and wheel-only import/CLI smoke passed. The current-tree wheel is
  `trainomni-0.1.2-py3-none-any.whl` (224,001 bytes), SHA-256
  `6fd9cf55d1b284016ad08e36031cc073dbd51596349daecf9c09bf9c3e68c845`,
  in the receipt directory's `dist/` subdirectory. It includes the new loader;
  the separately published correctness-only v2 branch does not.

## Corrections found during verification

- Worker-local partial batches no longer violate an incorrect global
  single-tail assumption. Delivered sample counts must remain between one and
  `batch_size` per delivered batch; invalid counter types/ranges still fail.
- The wrapper owns terminal EOF in state schema v2. Restoring and immediately
  re-snapshotting an exhausted finite stream never constructs an iterator or
  starts a new epoch. Legacy v1 snapshots require a valid TorchData finished
  marker; enclosing checkpoint/framework identity checks remain strict.
- The 26 loader regression cases cover 0/2 workers, persistent workers, full and
  partial batches, snapshots after the last batch and after observed EOF,
  invalid state, a finalized finite training checkpoint, and evaluation without
  replay/extra model forwards.
- Generic Transformers ModelIO now removes only declared batch axes, preserving
  single-image grids, patch axes and video axes. A real local Qwen processor
  read the two local images: single/double-image grids stayed `[1,3]`/`[2,3]`,
  the collated grid was `[3,3]`, patches `[3504,1536]`, and token IDs `[2,602]`.
  This processor-only check used no model weights. The task-local GPU adapter
  above is a separate, explicitly packing-aware path.

## Important limits

This is fixed-length sequential packing with explicit dense block-diagonal
attention, not FlashAttention variable-length/no-padding packing or a
best-fit/length-bucketed planner. The toy pack has 39/96 valid text slots; that
does not demonstrate improved utilization. Vision-token expansion is owned by
the model adapter and is not counted in the text packer's `max_length`.

Other model adapters must explicitly declare and implement packed attention,
position handling and multimodal alignment. A packer setting alone cannot make
an arbitrary VLM packing-compatible. Long-sequence throughput, remote storage,
multi-node/multi-GPU execution and GPU multi-worker loading remain outside this
verification.

## Delivery separation

The correctness-only branch `codex/framework-v2-data-correctness-20260903` is
commit `4dacc0d104c91ef88609d90200a5cc99c064e74c`, directly based on
`ced8cf83b9df044d71bcd7916de54b815ea34aed`. It contains no DataLoader runtime or
TorchData dependency and has its own **261 passed / 1 skipped** suite and real
packing verification. Its fixes include strict data/state/mask/identity checks
and processor batch-axis handling.

The local current-tree loader implementation, its EOF fixes and this report are
not part of that pushed branch. Their stable tested snapshot is identified by
the source digest and wheel above. No `main`, `v1` or `v2` ref was changed.
