# Framework v2 data correctness audit — 2026-09-03

## Scope

This branch is a correctness-only patch over `ced8cf83b9df044d71bcd7916de54b815ea34aed`.
It does not add a DataLoader runtime, asynchronous loading, a new storage format,
or a new training capability.

## Correctness gates closed

- Data source, transform, ModelIO, supervision, packer and collator configuration
  rejects coercible boolean/integer/sequence values instead of silently changing
  recipe semantics. Step, batch and checkpoint cardinalities follow the same rule.
- Canonical sample/record/example/batch identities require actual, trimmed,
  non-empty strings. JSONL and memory parsers use field presence for the strict
  `content` xor `messages` contract.
- Assistant and loss masks must be tensors aligned with tokens and containing
  only boolean or exact binary values. NaN, infinity, negative values and values
  greater than one fail before labels are produced.
- Collation checks tensor dtype and device before every stack, pad or concat, so
  neither promotion nor truncation can be silent. NaN padding configuration is
  rejected.
- Sequence packing checks integer token/label semantics, binary validity masks,
  dtype/device consistency and reachable buffered state. Packed attention
  independently revalidates the binary mask and device contract.
- Memory, JSONL, columnar, mixture, generic rank-sharded and pipeline restore
  paths reject unreachable or type-coerced state. JSONL verifies byte offsets
  against actual line boundaries and line counts.
- `data.sources` and model component names cannot collide through string
  coercion or whitespace normalization. ms-swift row IDs use a string-only
  policy; only a missing/`None` row ID falls back to the physical record ID.
- The ms-swift adapter rejects ambiguous media mappings and reserves the
  `trainomni.*` provenance namespace.
- Generic Transformers ModelIO removes only explicitly declared singleton batch
  axes. Single-image grids and patch/video axes are preserved, including through
  single-plus-double-image collation. Processor-specific overrides are documented
  as `batch_axis_fields`; shape-based inference is not used.
- Repeating physical columnar shards require equal assigned row totals. This is
  a deliberate fail-closed boundary until an explicit equal-step sampler with
  duplicate accounting exists; independent local shard rollover is not allowed
  to bias sample frequency.

## Verification evidence

- Framework tests: **261 passed, 1 expected skip**.
- Two fresh Python processes, each spawning two CPU/Gloo ranks: physical Parquet
  shards were disjoint and checkpoint continuation was exact on both ranks.
- Pinned local InterGPS Parquet: **1280/1280 rows**, **1280 image samples**, no
  duplicate sample IDs; finite two-rank partition was **680/600**. Repeating the
  same layout fails before reading because its rank totals are unequal.
- Pinned diagram-image-to-text data converted to Arrow IPC: **300/300 rows**,
  **300 image samples**, no duplicate sample IDs; finite two-rank partition was
  **164/136**.
- Real Qwen3.5 Vision + MiniCPM5-1B packing route on CUDA BF16: two source
  sequences of **19** and **20** tokens formed one 39-token pack, cross-segment
  attention entries were **0**, the second boundary label was **-100**, two
  optimizer steps changed three connector tensors, and same-fixture packed loss was
  finite.
- Ruff and compileall passed; `git diff --check` found no whitespace errors.
- A dependency-free wheel build with `pip wheel --no-deps --no-build-isolation`
  succeeded. Wheel-only import reported version `0.1.1`, the builtin catalog
  exposed 41 descriptors, and CLI help passed. Wheel SHA-256:
  `d2a59408dba63191bd3f30a991274df4281b323caaa49928b5ffa80c88586404`.

Generated wheel files, caches, environments, model/data payloads and the separate
uncommitted DataLoader work are excluded from the branch. The shared root
working tree and index are not switched, reset, cleaned or staged by this branch.

The real-data checks are engineering correctness evidence, not model-quality or
throughput claims.
