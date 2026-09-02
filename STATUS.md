# Status

TrainOmni Framework replacement v0.1.2 is the corrected current implementation
for the documented multimodal-understanding engineering scope. The rejected
pre-redesign code is archive-only and is not a second active implementation.

## Completed scope

- Framework, task, run, output and platform-launch boundaries are separate.
- Typed module registry/resolver, strict TaskSpec/RunSpec, capability preflight and
  SHA-pinned task-local extensions are executable.
- Flat/chat multimodal samples, media identity, image/video transforms,
  safetensors sidecar cache, Transformers ModelIO, supervision, collation and
  resumable sequence packing are implemented.
- Opt-in padding-free packer/collator and an upstream xFormers CUTLASS Llama
  integration are implemented. CUDA forward/backward and an explicit real-VLM
  visual-prefix probe pass; this is not FlashAttention or a generic model toggle.
  See [the contract](docs/contracts/padding-free.md).
- Offline-cache schema v4 binds every per-sample uncollated model-input mapping,
  including media and auxiliary tensors; Framework computes the current digest
  before collation and rejects stale KD/DPO caches before model forward.
- Named child data sources and deterministic weighted mixture sampling preserve
  every child cursor, selection cursor and per-source count across exact resume.
- The default TorchData StatefulDataLoader runtime executes Parquet/Arrow
  processing, packing and collation inside Windows-spawn-safe workers, supports
  bounded prefetch/persistent workers/pinned batches, and restores per-worker
  state before Framework gathers rank-local checkpoint state.
- Monolithic and ordered multi-branch composite models support vision/video
  encoders, connectors, prefix/token-replace/cross-attention fusion and independent
  semantic/runtime attention selection.
- Causal LM, offline dense-logit KD and offline-reference DPO use one Objective ABI
  and one training engine.
- Full/component/freeze/native Linear-LoRA policies, AdamW groups, schedulers,
  accumulation, clipping, precision, activation checkpointing and optional
  `torch.compile` are implemented.
- RunSpec selects direct PyTorch single/DDP/FSDP2 or an optional thin Linux
  DeepSpeed adapter. Rank/process/device negotiation, deterministic disjoint data,
  global reductions, per-rank metrics and rank-safe runtime state are explicit.
- Model-declared FSDP units are executable. Expert/router, replicated-module and
  tied-parameter topologies fail closed unless a topology-aware backend owns them.
- Split atomic checkpointing records model/optimizer/runtime identities and state;
  exact resume, model-only load, held-out evaluation and structured resource/update
  evidence are implemented.
- Framework builtin provenance, remote/local Transformers asset identity and
  Parquet/Arrow snapshot identity are explicit. Unpinned external assets cannot
  claim checkpointed exact resume.
- Gradient accumulation uses one globally reduced semantic denominator across all
  microbatches/ranks. A two-rank Gloo oracle covers unequal local token counts.
- Objective metrics declare either global-sum or weighted-mean numerator/denominator
  semantics. Unequal microbatch/rank token and pair counts use exact global
  aggregation rather than averaging local scalars.
- Finite sources flush packer state and expose partial/drop-last behavior;
  multi-rank finite or unknown exhaustion fails before reading until an equal-step
  sampler exists.
- Rank-local scheduler/objective/stream/scaler/RNG capture failures are gathered
  before checkpoint collectives or filesystem work, and rank-zero filesystem or
  materialization failures are broadcast. Rank-invariant distributed objective
  state is portable to single-process model-only evaluation; rank-dependent state
  fails closed.
- FSDP2 uses upstream distributed state-dict APIs for portable full training state.
  Checkpoint-disabled diagnostic runs are explicit and non-resumable. DeepSpeed
  checkpointing is not claimed and fails before training.
- Generic safetensors, Transformers `save_pretrained` and strict native LoRA
  adapter export/load paths are implemented.
- Windows/Linux single and distributed launch scripts are isolated from Python
  task/run semantics and require an explicit interpreter.

## Verification evidence

- Current source including padding-free additions: **317 passed, 1 skipped**
  (74.06 s); CUDA tests executed, only the POSIX launcher is skipped on Windows.
  Ruff, CLI help/43 descriptors and `git diff --check` pass. The new modules are
  included in the Git `v3` release snapshot, not in the earlier wheel.
- Prior data-correctness baseline: **298 passed, 1 skipped**; together with nine coordinator
  reproductions: **307 passed, 1 skipped**. The only skip is the POSIX launcher
  execution test on the current Windows host. See the
  [2026-09-03 local-data/packing report](docs/verification/local-packing-20260903.md).
- Two-worker Parquet and Arrow IPC tests pass under Windows `spawn`, cover
  disjoint physical-fragment assignment, complete sample coverage and exact
  continuation from a snapshot containing both active and not-yet-active workers.
- Ruff: clean across `src/trainomni` and `tests`.
- Python compileall: passed with bytecode directed outside Framework.
- Prior data-correctness wheel (before the new padding-free modules):
  `trainomni-0.1.2-py3-none-any.whl` built successfully; SHA-256
  `6fd9cf55d1b284016ad08e36031cc073dbd51596349daecf9c09bf9c3e68c845`.
- Prior isolated wheel import/CLI: version `0.1.2` loaded from the isolated target;
  that wheel has 41 builtin descriptors. The current source has 43 after adding
  padding-free pack/collate; the earlier wheel does not contain this new route.
- Current explicit real-VLM padding-free probe: 537 post-fusion tokens, zero
  padding/no dense language mask, two BF16 connector updates, per-sample numerical
  oracle, exact cross-sample isolation and same-fixture inference pass. No
  additional package install. [Evidence](docs/verification/padding-free-20260903.md).
- Project-local Windows interpreter: `Framework/.venv/Scripts/python.exe`, Python
  3.12.13, Torch 2.13.0+cu130. It resolves CUDA 13.0 on an RTX 4060 Ti (compute 8.9),
  and a true-BF16 forward/backward plus AdamW update passed on `cuda:0`.
- Historical pre-fix evidence: tiny composite true-BF16 CUDA train/checkpoint/
  fresh-process exact resume/evaluate/export and tiny monolithic CUDA lifecycle
  passed. These routes require current-tree CUDA revalidation.
- Historical pre-fix evidence: a real Qwen3.5 vision + MiniCPM5-1B chain ran
  connector alignment → multimodal
  pretraining → full-parameter SFT → offline dense-logit KD → offline-reference
  DPO on the RTX 4060 Ti. Every stage completed train/checkpoint/evaluate/export,
  component update evidence is nonzero, the final artifact strictly reloads and a
  fresh post-reload multimodal forward is finite. See
  `docs/verification/real-vlm-five-stage-20260821.md`.
- Historical pre-fix evidence: native Linear-LoRA SFT and offline-reference DPO
  train/evaluate/export
  passed across 219 vision/connector/LLM targets. Strict adapter reload is
  bit-identical to checkpoint logits. A real batch-size-2 run also passed unequal
  text lengths, one/two images per sample, unequal image grids and two-way
  gradient accumulation.
- Historical pre-fix evidence: fresh-process exact resume passed independently for
  full-parameter SFT,
  multimodal pretraining, connector alignment, offline dense-logit KD and
  offline-reference DPO. Named model tensors, logical AdamW state, runtime state,
  final metrics and update evidence match the corresponding uninterrupted run.
- Historical pre-fix evidence: extension-route gates passed for a SHA-pinned
  task-local Objective,
  independent eager/SDPA runtime attention selection, two-source weighted data
  mixture, multimodal block-diagonal sequence packing and ordered video-frame
  input. All six routes completed BF16 CUDA train/checkpoint/evaluate with actual
  connector updates. See
  `docs/verification/real-vlm-extension-routes-20260821.md`.
- Historical pre-fix evidence: direct DDP and FSDP2 each executed a real process
  group on CUDA, performed
  forward/backward/update, saved portable state, resumed exactly and re-evaluated.
  The target Qwen3.5-vision/MiniCPM5 model additionally completed two-step DDP and
  FSDP2 training plus held-out evaluation on the RTX 4060 Ti. These are honest
  world-size-one backend gates; multi-rank remains server work.
- The current tree passes a two-rank CPU/Gloo unequal-denominator accumulated DDP
  oracle, explicit objective-metric sum/weighted-mean oracle, rank-local runtime
  capture failure, and rank-zero checkpoint/run-identity failure without deadlock.
- Historical pre-fix evidence: a deterministic medium fixture was built from 300
  diagram and 1,280 InterGPS
  rows. Connector alignment, full multimodal CPT, full SFT, LoRA SFT, cached dense
  KD, cached-reference DPO and LoRA DPO each completed 16 distinct-data optimizer
  steps: **7 routes / 112 steps**. All gradients and required component updates
  were finite/nonzero. See
  `docs/verification/real-vlm-medium-data-20260822.md`.
- Current-tree real-VLM/CUDA revalidation is pending. Historical runs are not
  presented as validation of the corrected cache, identity, metric or checkpoint
  paths.

## Explicitly outside v0.1.2

- Performance and model-quality characterization beyond the completed engineering
  gates. The medium-data loss curves are observations, not quality claims.
- Remote S3 cache coordination, WebDataset/Energon, Grain, DALI, dynamic
  multimodal cost batching and a dedicated device-prefetch stream.
- Real multi-rank/multi-host Linux NCCL execution and topology-change acceptance.
- DeepSpeed Linux backward/step and native ZeRO checkpoint bridging.
- Ascend/`torch_npu` device metrics, HCCL execution and multi-NPU checkpoint gates.
- QLoRA/quantized optimizers, tensor/pipeline/context parallelism.
- MoE expert parallelism; DDP may only replicate experts as ordinary data parallel.
- NVIDIA FP8/MXFP8/NVFP4; B200 baseline is BF16 CUDA/NCCL until an explicit
  Transformer Engine adapter exists.
- Audio understanding encoder and diffusion/generative-media training.

The detailed, authoritative claim boundary is
`docs/verification/support-matrix.md`. Upstream source clones live outside
Framework. Reference-only sources are never imported; optional DeepSpeed is used
only through its explicit backend and public API.
