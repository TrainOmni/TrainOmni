# Real VLM extension-route verification — 2026-08-21

This gate verifies that the replacement Framework is extensible at the points
that vary between VLM training tasks. It uses the local Qwen3.5 vision encoder,
MiniCPM5-1B language model, an RTX 4060 Ti, CUDA 13.0 and true BF16. The fixtures
are intentionally tiny: the result is an engineering compatibility claim, not a
quality or throughput claim.

The reusable verifier is
`D:/Codex/TrainOmni/FrameworkValidation/verify_extension_routes.py`. Large model
checkpoints are written under the system temporary directory and removed after
each route; compact receipts are retained under
`D:/Codex/TrainOmni/FrameworkValidation/extension-validation`.

## Results

| Route | Real execution | Boundary proved |
| --- | --- | --- |
| Task-local custom Objective | 2 BF16 CUDA optimizer steps, checkpoint, held-out evaluation | A SHA-pinned external module owns forward planning, position-weighted FP32 CE, label smoothing, reduction, named loss and metrics without modifying Framework or the model |
| Semantic attention + eager kernel | 2 BF16 CUDA optimizer steps, checkpoint, held-out evaluation | Task selects model-default attention semantics; Run selects eager; the real adapter exposes and records the explicit runtime kernel boundary |
| Semantic attention + SDPA kernel | 2 BF16 CUDA optimizer steps, checkpoint, held-out evaluation | The identical Task runs with SDPA selected only in RunSpec; no task/loss/data fork |
| Deterministic weighted mixture | 4 steps at batch size 2, checkpoint, held-out evaluation | Two independent JSONL sources are composed at weight 1:3 with namespaced IDs and structured source counts; the deterministic eight-sample draw was geometry=1, reasoning=7 |
| Multimodal sequence packing | 2 BF16 CUDA optimizer steps, checkpoint, held-out evaluation | Two samples of lengths 19 and 20 become one fixed-length pack; the second boundary label is `-100`, cross-segment mask entries are zero, visual prefixes receive the same expanded block-diagonal causal isolation, and the language model executes one packed forward |
| Video-shaped ordered frames | 2 BF16 CUDA optimizer steps, checkpoint, held-out evaluation | A canonical `video` block carries three ordered frame references; ModelIO resolves them without a video-container decoder and the vision processor emits three unequal grid rows |

Every route recorded a finite loss, nonzero connector gradient norm, a changed
connector tensor digest, at least three changed connector tensors, nonzero CUDA
allocated/reserved memory and a finite held-out loss. The custom Objective's
receipt additionally records `position_weighted_ce`, supervised-token count and
effective token weight.

## Fail-closed evidence

- Local Objective, model and ModelIO sources are pinned by full source-tree
  SHA-256 and imported only with explicit local-code opt-in.
- Explicit eager/SDPA selection is applied through
  `set_attn_implementation`; a model with no such boundary is already rejected by
  the focused kernel-policy test.
- A sequence pack paired with the model-default attention policy fails capability
  preflight because `model.attention.packed` is missing.
- Packed masks are re-derived from validity and segment IDs, and corrupted masks
  fail semantic-policy validation.
- Frame lists must be ordered, non-empty and resolve to existing local files.

## Durable evidence

- `custom-objective.json`
- `attention-eager.json`
- `attention-sdpa.json`
- `weighted-mixture.json`
- `sequence-packing.json`
- `predecoded-video.json`

All six receipts are in
`D:/Codex/TrainOmni/FrameworkValidation/extension-validation` and retain Task/Run
digests, final metrics, resource peaks, parameter-update evidence, model artifact
digest and held-out evaluation.

## Regression checks

- Focused attention/packing/local-module tests: **9 passed**.
- Full Framework suite: **90 passed, 1 skipped**. The skip is the expected POSIX
  launcher execution test on Windows.
- Ruff across Framework source/tests and the new validation modules/script, plus
  `git diff --check`: clean.

## Claim boundary

This gate does not claim model quality, large-corpus behavior, packing speedup,
video-container decoding, FlashAttention, distributed execution or performance
tuning. It proves that the task/module contracts reach real VLM forward,
backward, optimizer, checkpoint and evaluation paths without core changes per
task.
