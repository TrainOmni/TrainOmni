# VLMEvalKit Windows and real-model verification

Date: 2026-08-24

VLMEvalKit is usable as an external evaluation backend from the existing
TrainOmni CUDA environment. No additional Python environment or repository fork
was created.

## Identity and boundary

- upstream: `open-compass/VLMEvalKit`
- commit: `e8e78f05f3080fe28154f2130321f17951c3be94`
- package version: `0.2rc1`
- torch: `2.13.0+cu130`
- CUDA runtime: `13.0`, available
- integration: editable external checkout; not imported by TrainOmni Core

The Windows overlay and reproducible installer live under
`D:\Codex\TrainOmni\FrameworkValidation\evaluation`.

## Toolkit smoke

The deterministic smoke executed two image-backed MCQ samples through the real
VLMEvalKit runner. It covered model construction, prompt building, inference,
prediction TSV output, exact-match evaluation, score CSV output, status JSON,
and result publication.

- model calls: 2
- inference failures: 0/2
- judge failures: 0/2
- metric: `split=none|Overall = 1.0`
- upstream status: `done`, no error
- evidence:
  `D:\Codex\TrainOmni\FrameworkValidation\evaluation\runs\TrainOmniSmoke\T20260824-095556\trainomni-smoke-receipt.json`

This fixture proves the external evaluator plumbing only. It is not a model
quality result.

## Real checkpoint on AI2D_MINI

The real evaluation uses the exported `stage-05-final` composite checkpoint:

- vision encoder: `D:\Models\VLM\Qwen3.5-0.8B`
- language model: `D:\Models\LLM\MiniCPM5-1B`
- TrainOmni artifact SHA-256:
  `8f59b4c004058da9b6bb8b211c4785e5e58fb2ef18f2ad3c50961f2c0d98b3c2`
- benchmark: VLMEvalKit `AI2D_MINI`, 247 real diagram MCQs
- dataset SHA-256:
  `ecc687dee2a9d4272e5c478fcea676a54d0fe10fd951d7bf4bb12b5693a5ae8e`
- decoding: deterministic greedy generation, maximum 8 new tokens
- judge: local exact matching; no API or LLM judge
- device: CUDA BF16

The full path completed with 0/247 inference failures and 0/247 judge failures.
It produced prediction TSV, score CSV, status JSON, and a hash-pinned receipt.

- overall accuracy: 0.8097% (2/247)
- model calls: 247
- autoregressive forwards: 1,976
- wall time including model construction/evaluation: 105.82 seconds
- model-loaded CUDA allocation: 2,412,930,048 bytes
- peak inference CUDA allocation: 2,598,773,248 bytes
- evidence:
  `D:\Codex\TrainOmni\FrameworkValidation\evaluation\runs-real\TrainOmniStage05AI2D\T20260824-101229\trainomni-ai2d-real-receipt.json`
- repository receipt:
  `FrameworkValidation/evaluation/receipts/ai2d-mini-stage05-20260824.json`

The engineering integration passes, but the quality result does not. The model
usually emits empty/special-token-only output or geometry-caption fragments
instead of an option label. This is consistent with the tiny engineering
training fixture and its geometry-heavy supervision; it is not evidence of
AI2D capability. A separate constrained A-D diagnostic reached 23.48% but
collapsed to option A on 239/247 rows, so it is not reported as the primary
benchmark score.

## Regression evidence

After installing the evaluation dependencies into the existing environment:

- Framework tests: `105 passed, 1 skipped`
- Ruff: all checks passed

## Known boundary

`Polygon3` cannot currently be installed on this Windows/Python 3.12 host without
MSVC Build Tools. It is excluded from the evaluation installation, so any
benchmark that imports it is not claimed. This does not affect the validated
image-MCQ path. Linux deployment should use the unmodified upstream dependencies
and does not need the Windows compatibility overlay.
