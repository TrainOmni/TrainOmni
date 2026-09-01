# Real VLM five-stage CUDA validation — 2026-08-21

This is an engineering-path validation, not a quality experiment. It used the
real local Qwen3.5 vision tower and MiniCPM5-1B language model on one RTX 4060 Ti
16GB with true BF16. Every accepted stage ran uninterrupted for two optimizer
steps through the same public `TaskSpec` / `RunSpec` / Objective / engine /
checkpoint / evaluation / export boundaries.

The consumer fixture is outside Framework at
`D:\Codex\TrainOmni\FrameworkValidation`. Its model and ModelIO adapters are
SHA-pinned task-local modules; Framework contains no Qwen/MiniCPM name branch.

## Artifact chain

| Stage | Trainable scope / objective | Task digest | Run digest | Final train evidence | Held-out evaluation | Export digest |
| --- | --- | --- | --- | --- | --- | --- |
| 1. connector alignment | connector; causal LM | `6332f875560f0838f44cf90b5f04f37dde4e645c3f64b1112db9a2d631e2f21f` | `9c336b6a1631c78f178f88547e49bc630842a0c675106c19b69885b5bb5a106e` | loss `4.03409`; 3 connector tensors changed | `alignment_loss=3.86653` | `549e2dcb52b890483ae522c7fce9c9306a3d5aed15f7bd23211253fbb5c3d464` |
| 2. multimodal pretraining | vision + connector + LLM; causal LM | `7937680a304436a444107cce99a9719ec4906d2732d6e7d375842568625c765c` | `48a8b16fec13cca906913fec62c11f15340b7b24b924164ed7156c2c47abb327` | vision 127, connector 3, LLM 173 tensors changed | `pretraining_loss=4.88417` | `9ff76111e1604a2fd9aca1291c2ee99b18800c63c93664c864caa1f98908341a` |
| 3. full-parameter SFT | full model; assistant-only causal LM | `da2e93ca39cf70dfa175012b8fcea691139470ceb8b55c5ad9500380e0b9066b` | `75caa06bea6c42fa6112e3d435bc0a737774ed8fdc6b8b6aa642922318a0140b` | vision 127, connector 3, LLM 173 tensors changed | `sft_loss=8.24050` | `7591646b878b5059ce9641d480f4c8069b6aa5f0ccf812ba17d5f4c792fcf6dd` |
| 4. offline dense-logit KD | connector; `0.5 CE + 0.5 T² KL`, `T=2` | `0b32bdbc8f86d52ad1c9aa0e7f4d5eb45b45819167c349b804866360f1df97ee` | `8b9d63e3774cf31c2f69f90e83f7e0e10e8cfb24319145eb1b6caa977670c942` | CE `9.73684`, KL `0.44933`; 3 connector tensors changed | `kd_ce=9.52933`, `kd_kl=0.41392` | `b76286a5eabdc8ec88e70d4a1267f458531d65b97635da211157abf1a4ac5a98` |
| 5. offline-reference DPO | connector; sigmoid DPO, beta `0.1` | `eb9f3b4cec200784eba725f7e421eb91d304f4437d109254ed228f9875b3ce04` | `299b4c330f3129529db4047688e110dc6b8a7211e0fb2fbb5d2839fa95a2838a` | loss `0.51449`, margin `0.39632`, both policy forwards, 3 connector tensors changed | `dpo_loss=0.04298` | `8f59b4c004058da9b6bb8b211c4785e5e58fb2ef18f2ad3c50961f2c0d98b3c2` |

Stage outputs were consumed in order: alignment → pretraining → SFT → KD →
DPO. The KD cache was generated from the SFT artifact and contains complete
`[17, 130560]` BF16 logits behind a SHA-pinned safetensors index. The DPO cache
was generated from the KD artifact and contains shifted FP32 chosen/rejected
per-token reference log-probs behind a SHA-pinned index.

The final export has 376 tensors, its manifest and tensor digest agree, strict
artifact reload succeeded in a fresh process, and a post-reload real multimodal
forward produced finite `[17, 130560]` BF16 logits.

## Resource evidence

| Stage | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| Alignment | 3,090,323,456 B | 3,118,465,024 B |
| Multimodal pretraining | 11,261,701,120 B | 12,236,881,920 B |
| Full-parameter SFT | 11,269,361,152 B | 12,224,299,008 B |
| Offline KD | 3,088,645,120 B | 3,179,282,432 B |
| Offline DPO | 3,678,979,584 B | 3,781,165,056 B |

## Framework defects found and closed

1. Large CUDA parameters exposed out-of-bounds sampling in update evidence.
   CUDA integer `linspace` was replaced with exact CPU integer arithmetic; the
   regression covers a 200,540,160-element logical tensor.
2. True-BF16 batch placement incorrectly cast identity-bearing FP32 supervision,
   breaking offline DPO. Supervision now preserves declared dtype, while the
   generic `ForwardRequest` boundary applies model-input precision casting.

Targeted regressions and the complete Framework suite pass: **89 passed, 1
Windows-expected POSIX launcher skip**; Ruff and `git diff --check` are clean.

## LoRA and variable-batch follow-up gate

After the uninterrupted five-stage chain, the real VLM also passed the following
required framework paths:

| Route | Evidence |
| --- | --- |
| Native Linear-LoRA SFT | 219 vision/connector/LLM Linear targets, rank 4, 3,442,688 trainable parameters; 438 LoRA tensors changed; held-out loss `6.00911` |
| LoRA adapter export/reload | 6,938,072-byte adapter, SHA-256 `4531d0c10b4d296059e9494f2ce7602ee2e3b8ee91f6521f05d46570afc9f415`; strict reload reproduced checkpoint logits bit-for-bit (`max_abs_difference=0`) |
| Native Linear-LoRA offline-reference DPO | two real policy branches, 438 LoRA tensors changed, reward margin `0.24596`; held-out DPO loss `0.24614`; adapter SHA-256 `02502899c72fb3fe73d9c12a28b477840c35b9cb9a157d564a75b95712ed3fc4`; strict adapter-only reload reproduced the same finite DPO loss |
| Variable batch / multi-image | one real batch contained text lengths `[19, 54]`, image counts `[1, 2]`, three unequal image grids and padded text shape `[2, 56]` |
| Gradient accumulation | real `per_device_batch_size=2`, `gradient_accumulation_steps=2`; both optimizer records report two micro-batches, finite loss/gradients and connector updates |

LoRA exposed a lifecycle boundary that is now explicit: parameter policies are
materialized once during task assembly, before train/evaluate/export checkpoint
loading. This lets structural policies such as LoRA reproduce the same module
tree for model-only evaluation and adapter export; non-structural full/component
policies use the same path. The new integration regression executes native LoRA
train → checkpoint → evaluation → adapter export.

After these follow-ups the complete Framework suite passes: **90 passed, 1
Windows-expected POSIX launcher skip**. Framework and external consumer Ruff
checks plus `git diff --check` are clean.

## Retention and claim boundary

The complete final DPO artifact, final checkpoint, all metrics/evaluation
receipts, cache indexes, run identities and intermediate artifact manifests are
retained. Large intermediate model/optimizer payloads were pruned only after the
next stage consumed them because the validation disk had limited free space.
The compact LoRA SFT/DPO adapter artifacts are retained; their large full-model
checkpoint payloads were likewise pruned after evaluation/export/reload evidence
was complete.

This proves single-GPU uninterrupted training-path compatibility and artifact
handoff for the real composite. Fresh-process exact resume was subsequently
verified for all five real routes; see `real-vlm-exact-resume-20260821.md`. These
engineering gates do not prove model quality, distributed execution, Linux,
Ascend, audio, or generative-media training.
