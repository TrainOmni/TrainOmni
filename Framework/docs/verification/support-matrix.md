# Verified support matrix

This is the authoritative claim boundary for the replacement Framework. A source
directory, schema field, or reserved launcher is not counted as support. `verified`
means an automated test executes the stated path; hardware-specific claims say
which hardware was actually used.

## Training lifecycle

| Route | Status | Foundation | Evidence / boundary |
| --- | --- | --- | --- |
| Composite image/video encoder → connector → fusion → causal LM | verified on CPU and CUDA tiny fixtures | PyTorch + Transformers adapters | CUDA BF16 train, checkpoint, fresh-process exact resume, evaluation and export pass |
| Monolithic Transformers-compatible VLM | verified on CPU and CUDA tiny fixtures | Transformers model adapter + PyTorch runtime | CUDA BF16 train, exact resume, evaluation and safetensors export pass |
| Causal CPT / connector alignment / SFT | verified on real VLM CUDA chain | custom-required Objective ABI; PyTorch CE | connector alignment, full multimodal pretraining and full-parameter assistant-only SFT each train/checkpoint/evaluate/export |
| Offline dense-logit KD | verified on real VLM CUDA chain | custom-required Objective ABI; PyTorch FP32 CE/KL | SHA-pinned full-vocab BF16 teacher logits; CE/KL evidence, connector update, evaluation and export pass |
| Offline-reference sigmoid DPO | verified on real VLM CUDA chain | custom-required Objective ABI; PyTorch FP32 log-probs | two real policy forwards, SHA-pinned FP32 reference log-probs, connector update, evaluation and export pass |
| Live-reference or online-teacher objectives | unsupported | — | must be a new Objective plus explicit model/cache contract; no fallback |
| Diffusion / media generation | explicitly deferred | — | requires a different model/noise/scheduler Objective ABI |

## Data and model variation

| Capability | Status | Boundary |
| --- | --- | --- |
| Flat text/image/video/audio blocks and role-aware conversations | verified | audio is representable but no builtin decoder/encoder yet |
| Local media resolution and SHA-256 validation | verified | local files only |
| Pillow image decode | verified | RGB/RGBA/L and max-pixel gate |
| Video frame sampling | verified on real VLM CUDA for ordered frame lists | canonical `video` block → three unequal vision grids → train/checkpoint/evaluate passed; container-path decoding remains optional PyAV and is not hardware-tested |
| Hash-pinned safetensors sidecar cache | verified | index and shard digest failures occur before model forward |
| Transformers processor/chat template/assistant mask | verified with contract fixture | concrete processors remain model-task integration work |
| Variable multimodal tensor collation | verified on real VLM CUDA batch | stack/pad/concat/list are explicit; batch-size 2 passed unequal text lengths, image grids and one/two images per sample |
| Fixed-length multimodal sequence packing | verified on real VLM CUDA | two multimodal samples were packed into one language forward; boundary label masking, expanded visual-prefix block-diagonal causal isolation, finite backward/update and held-out evaluation passed |
| Multiple ordered modal branches | verified | independent encoder/connector per branch; audio can be added as another branch |
| Prefix, token replacement, cross-attention fusion | verified with fixtures | cross-attention requires a language module declaring `language.cross_attention` |
| Builtin multi-dataset weighted mixer | verified on real VLM CUDA | two named JSONL sources, 1:3 weights, namespaced IDs and structured counts reached a batch-size-2 train/checkpoint/evaluate route; child cursors/counts and immutable mixture identity also survive fresh-process exact resume |

## Parameter and execution policies

| Capability | Status | Evidence / boundary |
| --- | --- | --- |
| Full, selected-component and freeze policies | verified | strict component resolution and explicit optimizer groups |
| Native Linear-LoRA | verified on real VLM CUDA SFT and DPO | 219 explicit vision/connector/LLM Linear targets; train/evaluate/export pass and strict adapter reload is bit-identical to checkpoint logits |
| QLoRA / quantized base weights | unsupported | no quantization backend is silently inferred |
| AdamW, including `foreach=false` and per-group LR/weight decay | verified | optimizer type/version/state dtype recorded in checkpoint metadata |
| AdamW8bit | unsupported | no bitsandbytes dependency or silent downgrade |
| Constant/linear/cosine schedule | implemented; exact state resume verified | native PyTorch `LambdaLR` |
| Gradient accumulation and norm clipping | verified on real VLM CUDA | batch-size 2 × accumulation 2 records two micro-batches per optimizer step, finite gradients and actual connector updates |
| Component activation checkpointing (`use_reentrant=false` supported) | verified on real vision + LLM components | full-model pretraining/SFT apply both component hooks; isolated memory-benefit attribution is not claimed |
| `torch.compile` | verified with CPU `eager` backend | compiled callable is separate; checkpoint keys remain unwrapped |
| FP32 | verified on CPU | full lifecycle |
| true BF16 and BF16 autocast | true-BF16 CUDA real lifecycle verified | RTX 4060 Ti five-stage real VLM train/checkpoint/evaluate/export plus five-route fresh-process exact resume |
| FP16 autocast | implemented for CUDA only | no CUDA verification in the replacement environment yet |
| CUDA memory allocated/reserved metrics | real-VLM verified | all five stages record nonzero structured peaks; full-model maximum reserved was 12,236,881,920 bytes |

## Checkpoint, evaluation, export and extension

| Capability | Status | Boundary |
| --- | --- | --- |
| Atomic split checkpoint | verified | `model.safetensors`, `optimizer.pt`, `runtime.pt`, manifest SHA-256 |
| Exact resume | verified on real VLM CUDA routes | full SFT, pretraining, alignment, offline KD and offline DPO match uninterrupted logical model/AdamW/runtime state and final evidence after fresh-process resume |
| Model-only evaluation/export load | verified | does not allocate/load optimizer state; objective restore is optional |
| Held-out evaluation | verified | separate data stream, eval/inference/autocast/device semantics |
| Generic full-state safetensors export | verified | composite and monolithic fixtures |
| Transformers `save_pretrained` export | verified in a fresh process | reloaded logits are bit-equal for the FP32 tiny model |
| Native LoRA adapter export | verified | strict target/config/tensor/digest matching |
| Task-local module extensions | verified on real VLM CUDA | a SHA-pinned external Objective owns a distinct position-weighted, label-smoothed FP32 CE and completes train/checkpoint/evaluate; all module kinds share descriptor/config/capability/source-hash rules |
| Semantic attention policy | verified on real VLM CUDA | model-default and packed block-diagonal policies execute through the model boundary; incompatible packer/policy composition fails capability preflight |
| Runtime attention kernel | eager and SDPA verified on real VLM CUDA | identical semantic Task ran with eager and SDPA chosen only by RunSpec; applied model boundaries are recorded; FlashAttention is not claimed |
| Framework / Task / Run / Output root separation | verified | task and run digests are distinct; outputs are immutable receipts |

## Distributed and platform boundary

| Route | Status | Boundary |
| --- | --- | --- |
| Single process/device | verified on CPU | current executable runtime |
| CUDA single-device | real VLM five-stage lifecycle and exact resume verified | Windows, RTX 4060 Ti, Torch 2.13.0+cu130; uninterrupted train/checkpoint/evaluate/export, strict reload and five-route fresh-process resume pass |
| DDP | explicitly deferred | no source stub, RunSpec field, launcher, or support claim |
| FSDP2 | explicitly deferred | requires distributed checkpoint/topology contract and two-process tests |
| Tensor/pipeline/context parallel | unsupported | no advertised schema or fallback |
| Ascend/HCCL/NPU | explicitly deferred | later device/distributed adapter; no model/objective fork |
| Windows launcher | verified | explicit interpreter, pure CLI forwarding |
| Linux launcher | present, platform test skipped on Windows | Linux validation intentionally postponed |

TRL, PEFT, Accelerate, ms-swift, LLaMA-Factory, VeOmni, TorchTitan and NeMo are
not runtime dependencies or backends. Their pinned source trees are external
reading references only. The in-process runtime dependency set is declared in
`pyproject.toml`: PyTorch, Transformers, safetensors, PyYAML and Pillow; PyAV is
optional only for video-file decoding.
