# Local varlen / padding-free verification — 2026-09-03

## Result and boundary

The current Windows environment does **not** need another `flash-attn` install
to validate the variable-length/padding-free contract. Installed xFormers 0.0.35
executes the explicit CUTLASS route using PyTorch's
`aten::_efficient_attention_forward` / `aten::_efficient_attention_backward`.
This is not a FlashAttention verification or a speed comparison.

Environment: Framework `.venv`, Python 3.12.13, Torch 2.13.0+cu130,
Transformers 5.15.0, RTX 4060 Ti 16 GB. No packages were installed or replaced.
xFormers is inherited from the existing environment; `xformers.info` reports a
Torch 2.10.0+cu130 build. Only the exercised `cutlassF-pt/cutlassB-pt` routes backed
by the **current Torch's aten operators** are verified, not every bundled
xFormers binary extension. A deployment must pin/revalidate its own build.

`flash_attn` is absent. `torch.backends.cuda.is_flash_attention_available()` is
false; an actual public `torch.nn.attention.varlen.varlen_attn` CUDA call failed
with `USE_FLASH_ATTENTION was not enabled for build`. Existence of that Python
API does not imply this Windows wheel includes its kernel.

## Executed checks

Final full source suite: **317 passed, 1 skipped** in 74.06 s. The skip is the
POSIX launcher on Windows, not a CUDA/varlen skip. New coverage comprises 11
padding-free data tests and 8 actual CUDA integration tests. Ruff, CLI help,
43 builtin descriptors and `git diff --check` pass. These additions are included
in the Git `v3` release snapshot; the previously built 0.1.2 wheel predates them.

- FP16/BF16 CUDA variable-length forward and Q/K/V backward vs independently
  evaluated FP32 causal sequences; both MHA and GQA, lengths 1/7/13.
- Exactly unchanged outputs in an unaffected sample after another sample is
  mutated, and no future-token leakage within a sample.
- Profiler evidence of upstream efficient-attention forward/backward operators;
  no SDPA fallback in the isolated kernel check.
- Tiny Llama concatenated vs separate forwards and embedding gradients with
  frozen model weights. Missing layout, wrong token sum and CPU input fail.
- Padding-free pack/collate, EOF and buffered-state restore, shifted boundaries,
  wrong cumulative dtype/endpoint, stale positions/segments and invalid masks.
- Real pinned Qwen3.5 Vision + MiniCPM5-1B, two local geometry samples from
  `FrameworkValidation/data/packing.jsonl`, using a Framework-owned explicit
  visual-prefix probe. External fixtures and model assets are read-only.

Real VLM layout: text lengths **19/20**, visual lengths **242/256**, final
sequence lengths **261/276**, concatenated language input **537 tokens**,
**zero padding tokens**. A dispatch guard rejects any 537×537 language-side
tensor allocation during packed forward. Vision attention is independently
configured and is outside that no-quadratic-LM claim.

The numerical oracle runs each visual-prefix sample separately through SDPA,
then uses the same Framework causal objective and token weighting. BF16 packed
and separate matrix shapes need not be bit-identical. Observed relative-L2
logit error was ~1.47%; connector gradient errors ~1.33–1.39%; CE was
9.21714 (separate) versus 9.19947 (packed). Checks use preset bounds of 1.5%,
6% and 0.02 respectively; this is numerical engineering evidence, not exact
equivalence. The small FP32-reference kernel tests have tighter error gates.

Two packed optimizer steps changed actual connector parameters; vision and
language weights remained frozen while gradients passed through the language
model. The changed-element counts were 1,574,919 and 1,511,889; training CE was
9.19947 then 8.20667. Altering sample A left sample B logits bit-identical
(max absolute difference 0). Same-fixture inference CE afterward was 8.74650.
Peak allocated/reserved was 3,703,151,616 / 3,967,811,584 bytes for this probe,
including oracle work; these are not comparable throughput/memory benchmarks.
Loss values are diagnostic only;
this fixture cannot support a quality/generalization or speedup claim.

## Reproduction

From `D:\Codex\TrainOmni\Framework`:

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/modules/data/test_sequence_packer.py tests/unit/modules/data/test_padding_free.py tests/integration/test_varlen_cuda.py -q
.venv/Scripts/python.exe scripts/validate_padding_free_vlm.py
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m ruff check src tests scripts/validate_padding_free_vlm.py
git diff --check -- .
```

The real probe writes only `.cache/varlen-validation-20260903/real-vlm.json`,
including source/task/artifact identities, library versions, expanded lengths,
actual operator counts, oracle errors, alignment failure, per-step full
changed-element counts, gradients, loss and peak allocated/reserved memory.
No new model checkpoint or dataset payload is created. The script does not
modify the shared fixture project, register a consumer task, or publish Git refs.

The probe deliberately owns model-specific fusion outside generic runtime code;
its manual optimizer loop verifies the new attention/data contract, **not** a
new CLI train/checkpoint/resume acceptance. Existing lifecycle evidence is not
silently extended to this backend. [Integration contract](../contracts/padding-free.md).

Final probe source digest: `db93352966f07b87ad2bceba8853d035fe64afbc6f9322be44956002885465ef`
(lexicographic `src/**/*.py` relative POSIX paths followed by bytes).
Fixture task digest: `c0e175178f6f872d95c2fd72d39e8074329d460e20c8598e501a476de241c159`.
Initial artifact digest: `b76286a5eabdc8ec88e70d4a1267f458531d65b97635da211157abf1a4ac5a98`.
