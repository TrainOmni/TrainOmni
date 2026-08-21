# Status

TrainOmni Framework replacement v0.1.0 is basically complete for the
single-process multimodal-understanding scope. The rejected pre-redesign code is
archive-only and is not a second active implementation.

## Completed scope

- Framework, task, run, output and platform-launch boundaries are separate.
- Typed module registry/resolver, strict TaskSpec/RunSpec, capability preflight and
  SHA-pinned task-local extensions are executable.
- Flat/chat multimodal samples, media identity, image/video transforms,
  safetensors sidecar cache, Transformers ModelIO, supervision, collation and
  resumable sequence packing are implemented.
- Named child data sources and deterministic weighted mixture sampling preserve
  every child cursor, selection cursor and per-source count across exact resume.
- Monolithic and ordered multi-branch composite models support vision/video
  encoders, connectors, prefix/token-replace/cross-attention fusion and independent
  semantic/runtime attention selection.
- Causal LM, offline dense-logit KD and offline-reference DPO use one Objective ABI
  and one training engine.
- Full/component/freeze/native Linear-LoRA policies, AdamW groups, schedulers,
  accumulation, clipping, precision, activation checkpointing and optional
  `torch.compile` are implemented.
- Split atomic checkpointing records model/optimizer/runtime identities and state;
  exact resume, model-only load, held-out evaluation and structured resource/update
  evidence are implemented.
- Generic safetensors, Transformers `save_pretrained` and strict native LoRA
  adapter export/load paths are implemented.
- Windows and Linux single-process launch scripts are isolated from Python task/run
  semantics and require an explicit interpreter.

## Verification evidence

- Full source suite: **90 passed, 1 skipped**. The only skip is the POSIX launcher
  execution test on the current Windows host.
- Ruff: clean across `src/trainomni` and `tests`.
- Python compileall: passed with bytecode directed outside Framework.
- Wheel: `trainomni-0.1.0-py3-none-any.whl` built successfully; SHA-256
  `1e54bd8179000300e1f73b2efaba918e081f66969993c7a13f37e9055da7e168`.
- Isolated wheel import/CLI: version `0.1.0`; the current source catalog has 38
  builtin descriptors and CLI help passes.
- Installed-wheel lifecycle subset: **8 passed**, covering composite exact resume,
  monolithic train/eval/export, KD/DPO shared engine and fresh-process Transformers
  artifact reload.
- Project-local Windows interpreter: `Framework/.venv/Scripts/python.exe`, Python
  3.12.13, Torch 2.13.0+cu130. It resolves CUDA 13.0 on an RTX 4060 Ti (compute 8.9),
  and a true-BF16 forward/backward plus AdamW update passed on `cuda:0`.
- The full source suite was rerun from that interpreter with user site packages
  disabled: **88 passed, 1 skipped**; Ruff and `pip check` are clean.
- Tiny composite true-BF16 CUDA train/checkpoint/fresh-process exact resume/evaluate/
  export passed with weighted multi-source data. Tiny monolithic CUDA train/resume/
  evaluate/export also passed. Both record nonzero allocated/reserved GPU memory.
- A real Qwen3.5 vision + MiniCPM5-1B chain ran connector alignment → multimodal
  pretraining → full-parameter SFT → offline dense-logit KD → offline-reference
  DPO on the RTX 4060 Ti. Every stage completed train/checkpoint/evaluate/export,
  component update evidence is nonzero, the final artifact strictly reloads and a
  fresh post-reload multimodal forward is finite. See
  `docs/verification/real-vlm-five-stage-20260821.md`.
- Real native Linear-LoRA SFT and offline-reference DPO train/evaluate/export
  passed across 219 vision/connector/LLM targets. Strict adapter reload is
  bit-identical to checkpoint logits. A real batch-size-2 run also passed unequal
  text lengths, one/two images per sample, unequal image grids and two-way
  gradient accumulation.
- Real fresh-process exact resume passed independently for full-parameter SFT,
  multimodal pretraining, connector alignment, offline dense-logit KD and
  offline-reference DPO. Named model tensors, logical AdamW state, runtime state,
  final metrics and update evidence match the corresponding uninterrupted run.
- Real extension-route gates passed for a SHA-pinned task-local Objective,
  independent eager/SDPA runtime attention selection, two-source weighted data
  mixture, multimodal block-diagonal sequence packing and ordered video-frame
  input. All six routes completed BF16 CUDA train/checkpoint/evaluate with actual
  connector updates. See
  `docs/verification/real-vlm-extension-routes-20260821.md`.

## Explicitly outside v0.1.0

- Performance and quality characterization beyond the completed engineering gates.
- DDP, FSDP2, distributed launch/checkpoint, Ascend/HCCL and multi-host execution.
- QLoRA/quantized optimizers, tensor/pipeline/context parallelism.
- Audio understanding encoder and diffusion/generative-media training.

The detailed, authoritative claim boundary is
`docs/verification/support-matrix.md`. Upstream source clones live outside
Framework and are never imported or executed.
