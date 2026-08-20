# Framework Task Status

- Role: `framework`
- Owned path: `D:\Codex\TrainOmni\Framework`
- Updated: 2026-08-20
- State: TrainOmni v1 implemented and verified; VLM-first, with audio understanding and generation intentionally later

## Completion summary

| Area | State | Evidence |
|---|---|---|
| SOTA/open-source research | Complete | `docs/research/` |
| Lifecycle, requirements, architecture | Complete v1 | `docs/design/` |
| Canonical multimodal protocol | Implemented | `src/trainomni/data/model.py`, schema and fixtures |
| Reader/importer extensibility | Implemented | JSON/JSONL/Parquet/TAR + explicit data plugins |
| Stateful mixture/batching | Implemented | exact cursor/RNG/look-ahead and distributed grouping |
| Model registration | Implemented | external plugin, manifest, capabilities, exact component cover |
| Objective layer | Implemented | masked causal LM + delegated DPO/KD/GRPO/PPO contracts |
| Native execution | Implemented | PyTorch single/DDP/FSDP2 loop |
| PEFT/precision/optimization | Implemented | LoRA/QLoRA, AMP/TF32, component policies, schedulers |
| Checkpoint/resume | Implemented | local atomic + DCP; exact runtime state; model-only reshard/export |
| Pipeline/gates/lineage | Implemented | durable DAG executor and artifact URIs |
| Delegated backends | Implemented | pinned VeOmni bridge + secure TRL/NeMo/veRL/custom command adapters |
| Evaluation/export | Implemented | normalized loss, external evaluator, HF/plugin export |
| CLI/provenance/logging | Implemented | validate/inspect/plan/train/run/evaluate/export |
| Public real VLM smoke | Passed | `examples/plugins/tiny_llava.py` |
| Automated tests | 57/57 passed | optional-runtime suite; torch/pyarrow tests skip cleanly without their extras |
| Distributed verification | Passed | 2-process CPU DDP and FSDP2+DCP exact resume |
| Ascend multi-node | Deferred by explicit user decision | no implementation claim |

## Accepted decisions

1. TrainOmni selectively self-builds stable semantics/control-plane boundaries and reuses open-source kernels/runtimes.
2. PyTorch + Transformers is the native CPT/SFT execution ABI; FSDP2 + DCP is the sharded scale path.
3. RL/agentic orchestration is a delegated Stage, not forced into a fake batch loss loop.
4. New models and data formats require explicit external plugins and normally zero core edits.
5. Exact resume is a protocol property and includes data/mixture/batch/RNG state, not only model weights.
6. Local exact checkpoints contain pickle state and therefore require explicit trust flags; DCP model-only loading does not deserialize the rank-local runtime pickle.
7. TrainOmni is VLM-first: audio understanding is the next modality priority, while diffusion/generative training is deferred.
8. VeOmni is the preferred future scale/Ascend engine; the native torch engine remains the local correctness oracle.

## Verified boundaries

- Public tiny LLaVA: real processor, image encode, assistant-only loss mask, forward/backward, two checkpoints, exact resume tensor equality, loss evaluation, HF safe export.
- Single-process torch toy: automated train/resume/eval/export test.
- DDP: deterministic global batch grouping, per-rank exact checkpoint, two-process uninterrupted/resume tensor equality on both ranks.
- FSDP2: `fully_shard`, DCP model/optimizer, rank-local runtime sidecars, exact resume, single-process model-only DCP reshard/export equality.
- Delegated GRPO: explicit command authorization, redacted request manifest, metrics/output collection, no core model build.
- VeOmni bridge: immutable backend revision, versioned request/result contract, VLM-only capability matrix, and explicit refusal to claim exact resume before conformance.
- Two-stage torch Pipeline: physical checkpoint URI loading, lineage reconstruction and idempotent same-executor resume.
- Plugin security: YAML never auto-imports Python; model/data code requires explicit CLI trust.

## Known limitations (not hidden completion gaps)

- Native torch engine intentionally covers standard CPT/SFT-style loop ownership. TP/PP/CP, large-scale distillation, DPO and online/agentic RL execute through delegated backend adapters.
- FSDP2 exact runtime resume currently requires the same world size because rank-local data/RNG state is topology-specific. Model-only DCP can reshard to a different topology.
- Built-in TAR reader consumes JSON members; media-in-tar extraction policies belong in a data plugin.
- Real video/audio decode and model chat/processor semantics belong in model/data plugins; canonical contracts already represent them.
- LoRA/QLoRA code paths require optional PEFT and, for QLoRA, a plugin-loaded quantized model.
- Ascend/昇腾 has not been implemented in this version, per user direction.
- Audio is representable in the canonical contract, but no audio-encoder conformance plugin is claimed yet.
- Diffusion/continuous generative objectives are intentionally outside the current VLM-first implementation.

## Handoff anchors

- `README.md`: runnable entry point.
- `docs/implementation/framework-v1-2026-08.md`: what is implemented and why.
- `docs/design/support-matrix-v1.md`: exact native/delegated/plugin ownership.
- `docs/verification/verification-2026-08-20.md`: commands and observed evidence.
- `docs/usage/model-plugin.md`: target model integration boundary.
- `docs/research/open-source-foundation-decision-2026-08.md`: open-source foundation decision.

Next project work is the target VLM Model Plugin and target recipes/datasets. Audio understanding follows after the VLM path is stable; diffusion/generative training remains deferred. When scale/Ascend is re-prioritized, VeOmni is the first backend to validate.
