# Real VLM medium-data route validation — 2026-08-22

This gate checks varied-data training behavior after the small deterministic
lifecycle gates. It is an engineering observation, not a quality benchmark.

## Immutable inputs

- Model: Qwen3.5 vision + connector + MiniCPM5-1B, 1,182,802,176 parameters.
- Host: Windows, RTX 4060 Ti 16 GB, Torch 2.13.0+cu130, true BF16, SDPA.
- Dataset manifest SHA-256:
  `8eb92e52fa20f0d583bbff12e7c786cfcfd8ddb9ff8a43e2d43a72d8237fe8ae`.
- Diagram: 240 training / 30 validation samples.
- InterGPS: 1,024 training / 128 validation samples.
- KD and DPO bounded subsets: 64 training / 16 validation samples each.
- Dense-logit cache index SHA-256:
  `9f314b6a5e11fe3362f326420b25ce675bc772d316216b4baf75cd79d82637e3`.
- Reference-log-prob cache index SHA-256:
  `2249d2f12b86058c5d121d28b69ec0cac3ad9a570d988880afc573e6e841d54b`.
- Config manifest SHA-256:
  `3df7895de3897e9380cb26fa4eea402e68455780254b520b3203c428b74549f8`.

The preparation scripts verify the pinned Parquet digests and deterministic split
indices, extract canonical local images, write canonical JSONL, build hash-pinned
offline caches, and ensure every KD/DPO training and validation ID is present.

## Result

Seven independent routes completed 16 optimizer steps each: 112 total. Every loss
and gradient norm was finite. At steps 4/8/12/16 every required parameter group
proved actual changed tensors and changed sampled BF16 elements. Every event also
contained the world-size-one `data_metrics_by_rank` structure.

| Route | First-4 mean | Last-4 mean | Change | Linear slope/step | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| connector alignment | 5.2464 | 4.1487 | -20.9% | -0.1286 | 3.56 |
| multimodal CPT, full parameter | 6.3880 | 4.5237 | -29.2% | -0.2457 | 11.79 |
| assistant-only full SFT | 8.1995 | 0.9729 | -88.1% | -0.6108 | 11.78 |
| native Linear-LoRA SFT | 8.1261 | 1.5760 | -80.6% | -0.5408 | 2.91 |
| offline dense-logit KD | 5.7205 | 4.9206 | -14.0% | -0.0318 | 3.25 |
| offline-reference DPO | 0.6373 | 0.6605 | +3.6% | +0.0060 | 3.92 |
| native Linear-LoRA DPO | 0.6610 | 0.7365 | +11.4% | +0.0089 | 2.88 |

The DPO fixtures are 16 distinct cyclic-negative preference pairs, not a repeated
pair or a quality-tuned dataset. Their small positive loss slopes are reported,
not hidden and not treated as an engineering failure: forward/backward, cached
reference alignment, optimizer update and all fail-closed identities passed. A
claim about preference quality requires a real preference dataset and evaluation
protocol outside this gate.

The medium pass intentionally used `checkpoint.enabled=false`, because the earlier
real gates already cover checkpoint/evaluate/export/resume and duplicating seven
multi-gigabyte checkpoints would add no path coverage. The mode itself is strict:
no periodic/final checkpoint exists and explicit save is rejected.

Reproducible source and compact evidence live in the disposable consumer:

- `FrameworkValidation/prepare_medium_dataset.py`
- `FrameworkValidation/generate_offline_cache.py`
- `FrameworkValidation/prepare_medium_validation.py`
- `FrameworkValidation/summarize_medium_validation.py`
- `FrameworkValidation/medium-validation/receipt.json`

Receipt SHA-256:
`27b5e99eea6a38d33214ebbde8ec3601faeaef3b0b38bc8a4385f98a51bfda52`.
