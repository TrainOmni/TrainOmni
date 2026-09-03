# Verified support matrix

This is the authoritative claim boundary for the replacement Framework. A source
directory, schema field, or reserved launcher is not counted as support. Statuses
distinguish **current-tree automated/CPU-Gloo verified**, **historical pre-fix
real-VLM evidence**, and **current-tree real-VLM revalidation pending**. Historical
hardware runs demonstrate feasibility, not execution of the corrected current tree.

## Training lifecycle

| Route | Status | Foundation | Evidence / boundary |
| --- | --- | --- | --- |
| Composite image/video encoder → connector → fusion → causal LM | current-tree automated; historical pre-fix CUDA; real-VLM revalidation pending | PyTorch + Transformers adapters | current CPU vertical slice covers train/checkpoint/resume/evaluate/export; historical CUDA BF16 route is retained separately |
| Monolithic Transformers-compatible VLM | current-tree automated; historical pre-fix CUDA; revalidation pending | Transformers model adapter + PyTorch runtime | current CPU lifecycle tests; historical CUDA BF16 evidence is not a current-tree claim |
| Causal CPT / connector alignment / SFT | current-tree automated; historical pre-fix real VLM; revalidation pending | custom-required Objective ABI; PyTorch CE | corrected loss/metric/identity/checkpoint contracts are automated; historical real route remains feasibility evidence |
| Offline dense-logit KD | current-tree automated; historical pre-fix real VLM; revalidation pending | custom-required Objective ABI; PyTorch FP32 CE/KL | schema-v4 cache/alignment/numeric/gradient tests are current; historical full-vocab CUDA run predates the corrected schema |
| Offline-reference sigmoid DPO | current-tree automated; historical pre-fix real VLM; revalidation pending | custom-required Objective ABI; PyTorch FP32 log-probs | current paired-forward, cache, prompt-context, metric and gradient tests; historical CUDA run predates those corrections |
| Live-reference or online-teacher objectives | unsupported | — | must be a new Objective plus explicit model/cache contract; no fallback |
| Diffusion / media generation | explicitly deferred | — | requires a different model/noise/scheduler Objective ABI |

## Data and model variation

| Capability | Status | Boundary |
| --- | --- | --- |
| Flat text/image/video/audio blocks and role-aware conversations | verified | audio is representable but no builtin decoder/encoder yet |
| Local media resolution and SHA-256 validation | verified | local files only |
| Pillow image decode | verified | RGB/RGBA/L and max-pixel gate |
| Video frame sampling | current-tree automated; historical pre-fix real VLM; revalidation pending | ordered-frame semantics are tested; historical three-grid CUDA route predates the corrected engine tree; optional PyAV remains untested on hardware |
| Hash-pinned safetensors sidecar cache | current-tree automated | schema-v4 index/shard digests plus full expanded input IDs, full attention mask, absolute target positions/IDs, complete uncollated model-input/media digest, producer and branch bindings; builtin supervision computes current-input digests before collation, stale media/aux inputs and schemas v2/v3 fail before model forward |
| Transformers processor/chat template/assistant mask | verified with contract fixture | concrete processors remain model-task integration work |
| Variable multimodal tensor collation | current-tree automated; historical pre-fix real VLM; revalidation pending | stack/pad/concat/list are explicit; historical unequal batch route is retained as feasibility evidence |
| Fixed-length multimodal sequence packing | current-tree automated and single-GPU real VLM verified | [2026-09-04 feedback verification](feedback-v3-20260904.md): replacement raw-ViT/random-merger model, nested vision routing, B=2 packs, mixed one/two images, Parquet/Arrow and workers 0/1/2; FP32 per-sample oracle and BF16 isolation; no throughput claim |
| Padding-free data + variable-length causal attention | current-tree CUDA kernel and explicit real-VLM task verified | [2026-09-04 feedback verification](feedback-v3-20260904.md): optional xFormers CUTLASS, real task/CLI BF16 connector updates with workers 0/2, no dense LM mask and exact cross-sample isolation; sequence-preserving token replacement, one pack per batch; not FlashAttention, generic-VLM/distributed or speedup evidence |
| Multiple ordered modal branches | verified | independent encoder/connector per branch; audio can be added as another branch |
| Prefix, token replacement, cross-attention fusion | verified with fixtures | prefix regenerates ordinary expanded `position_ids` and rejects stale model-specific position/cache fields; token replacement supports masked unequal modal counts; cross-attention requires `language.cross_attention` |
| Builtin multi-dataset weighted mixer | current-tree automated; historical pre-fix real VLM; revalidation pending | current deterministic cursor/count/identity tests; historical 1:3 CUDA route is not a current-tree run |
| Stateful multi-worker Parquet/Arrow runtime | current-tree Windows-spawn + real VLM verified | TorchData StatefulDataLoader is the default runtime; raw-ViT/random-merger real BF16 tasks pass unpacked and packed with workers 0/1/2 and both formats. Explicit spawn/task-local bootstrap and bounded worker-failure tests pass; this does not claim the Linux/HAMI/NCCL SIGSEGV root cause is eliminated |
| Pinned batch plus non-blocking device transfer | implemented; tensor contract automated | `OmniBatch.pin_memory()` recursively pins tensors and device placement requests non-blocking copies; end-to-end throughput/overlap is not yet characterized |

## Parameter and execution policies

| Capability | Status | Evidence / boundary |
| --- | --- | --- |
| Full, selected-component and freeze policies | verified | strict component resolution and explicit optimizer groups |
| Native Linear-LoRA | current-tree automated; historical pre-fix real VLM; revalidation pending | target/config/export/reload contracts are current; 219-target CUDA SFT/DPO is historical evidence |
| LoRA trained bias | unsupported and fail-closed | `train_bias=true` is rejected during configuration; verified adapter identity covers LoRA tensors only |
| QLoRA / quantized base weights | unsupported | no quantization backend is silently inferred |
| AdamW, including `foreach=false` and per-group LR/weight decay | verified | optimizer type/version/state dtype recorded in checkpoint metadata |
| AdamW8bit | unsupported | no bitsandbytes dependency or silent downgrade |
| Constant/linear/cosine schedule | implemented; exact state resume verified | native PyTorch `LambdaLR` |
| Gradient accumulation and norm clipping | current-tree two-rank CPU/Gloo verified; historical pre-fix CUDA | local loss numerators/globally summed denominators match the unequal-count oracle; current-tree real-VLM revalidation pending |
| Component activation checkpointing (`use_reentrant=false` supported) | current-tree automated; historical pre-fix real components; revalidation pending | component hook placement is tested; full-model CUDA application is historical evidence |
| `torch.compile` | verified with CPU `eager` backend | compiled callable is separate; checkpoint keys remain unwrapped |
| FP32 | verified on CPU | full lifecycle |
| true BF16 and BF16 autocast | environment smoke current; historical pre-fix real lifecycle; revalidation pending | current host validates CUDA BF16 tensor/AdamW support; the full VLM route predates corrected Framework execution contracts |
| FP16 autocast | implemented for CUDA only | no CUDA verification in the replacement environment yet |
| CUDA memory allocated/reserved metrics | current-tree automated; historical pre-fix real VLM | collection/reduction tests are current; nonzero route peaks are historical and current-tree revalidation is pending |
| Training-only diagnostics without checkpoint payloads | current-tree automated; historical pre-fix real VLM | disabled mode preserves identities/metrics, writes no state and rejects explicit save; seven-route CUDA evidence is historical |

## Checkpoint, evaluation, export and extension

| Capability | Status | Boundary |
| --- | --- | --- |
| Atomic split checkpoint | verified | `model.safetensors`, `optimizer.pt`, `runtime.pt`, manifest SHA-256 |
| Exact resume | current-tree automated/CPU-Gloo; historical pre-fix real VLM; revalidation pending | current model/optimizer/runtime/RNG/identity and relocation tests pass; historical five-route CUDA resume cannot validate the corrected provenance/cache/metric/checkpoint tree |
| Distributed runtime-state identity | implemented; two-rank Gloo control-path verified | every rank contributes scheduler/objective/data/scaler/RNG state; topology changes fail; rank-invariant objective state supports single-process model-only evaluation while rank-dependent state fails closed |
| FSDP2 portable full-state checkpoint | current-tree automated world-size-one; historical pre-fix CUDA; real server gate | upstream state-dict bridge is tested; internal collective failures remain bounded by the process-group timeout; multi-rank exact resume is pending |
| Model-only evaluation/export load | verified | does not build the unused training source or allocate optimizer state; objective restore is optional; a validated checkpoint may be relocated and consumed with a different evaluation/export RunSpec while task/module/framework/file integrity remains strict |
| Held-out evaluation | verified | separate data stream, eval/inference/autocast/device semantics; config-addressed immutable receipts allow multiple batch/device/precision configs per checkpoint and are idempotent per config |
| Generic full-state safetensors export | verified | composite and monolithic fixtures |
| Transformers `save_pretrained` export | verified in a fresh process | reloaded logits are bit-equal for the FP32 tiny model |
| Native LoRA adapter export | verified | strict target/config/tensor/digest matching |
| Task-local module extensions | current-tree automated; historical pre-fix real VLM; revalidation pending | SHA-pinned extension loading/capability/identity is current; historical custom Objective CUDA route predates ObjectiveMetric |
| Semantic attention policy | current-tree automated; historical pre-fix real VLM; revalidation pending | model-default/packed compatibility is current; hardware route is historical |
| Runtime attention kernel | current-tree automated; historical pre-fix eager/SDPA real VLM; revalidation pending | RunSpec/model-boundary selection is current; FlashAttention is not claimed |
| Framework / Task / Run / Output root separation | verified | physical `checkpoint.directory` is excluded from RunSpec identity and stored in a separate location receipt; same-root directory relocation plus full resume passes |
| External Transformers asset identity | verified | immutable remote commit revision or producer-owned local asset-manifest digest enters task/module/checkpoint identity; unpinned assets are marked non-reproducible and checkpoint/exact-resume claims fail closed |
| Parquet/Arrow source identity | verified | producer-owned dataset-manifest digest plus logical fragment layout is semantic identity; physical roots may move; changed manifests fail exact restore; payloads are not repeatedly hashed by readers |
| Finite-source completion | verified single-process and two-worker columnar | packer tail flush plus explicit partial/drop-last behavior; multi-rank finite/unknown exhaustion fails before reading until an equal-step sampler exists |

## Distributed and platform boundary

| Route | Status | Boundary |
| --- | --- | --- |
| Single process/device | current-tree CPU verified; historical pre-fix real VLM CUDA; revalidation pending | default direct PyTorch runtime |
| CUDA single-device | historical pre-fix real VLM; current-tree revalidation pending | Windows RTX 4060 Ti five-stage/resume evidence is retained but predates corrected execution semantics |
| DDP | current-tree CPU two-rank Gloo verified; historical pre-fix CUDA world-size-one; Linux/NCCL pending | rank sharding, `no_sync`, exact loss and explicit metric oracles, rank-local capture and rank-zero filesystem failure coordination pass without deadlock |
| FSDP2 | current-tree automated world-size-one; historical pre-fix CUDA; multi-rank server gate | direct `fully_shard` and portable state APIs are tested; real current-tree CUDA/NCCL remains pending |
| DeepSpeed ZeRO 0/1/2/3 | thin optional Linux adapter; needs server execution | upstream owns backward/step/partitioning; Windows fails closed, no fallback; native ZeRO checkpoint bridge is absent so checkpoint-enabled runs fail closed |
| Deterministic distributed data and metrics | current-tree two-rank CPU/Gloo verified | disjoint rank streams/exact cursors; loss terms use global numerator/denominator; ObjectiveMetric declares sum or weighted mean; per-rank data metrics stay unguessed |
| Dense models under data/sharded parallel | implemented as above | DDP replicates; FSDP2 shards model-declared transformer units |
| MoE expert parallel | unsupported and fail-closed | DDP may replicate experts as ordinary data parallelism; generic FSDP2/DeepSpeed reject expert/router hints because no expert groups or dispatch exist |
| Tensor/pipeline/context parallel | unsupported | no advertised schema or fallback |
| Ascend/HCCL/NPU | explicitly deferred | later device/distributed adapter; no model/objective fork |
| NVIDIA B200 | Linux server gate | current BF16 CUDA/NCCL path should be used first; FP8/MXFP8/NVFP4 are unsupported until an explicit Transformer Engine adapter exists |
| Windows launchers | verified | explicit interpreter; single-process wrapper plus certified one-rank distributed probe, no native CUDA multi-process claim |
| Linux launchers | source-checked; executable platform test pending | single-process `exec` and `torch.distributed.run` wrappers are isolated from task/run semantics |

TRL, PEFT, Accelerate, ms-swift, LLaMA-Factory, VeOmni, TorchTitan and NeMo are
not runtime dependencies or backends. Their pinned source trees are external
reading references only. The in-process runtime dependency set is declared in
`pyproject.toml`: PyTorch, TorchData StatefulDataLoader, Transformers,
safetensors, PyYAML, Pillow and PyArrow; PyAV is optional only for video-file
decoding. DeepSpeed is an optional Linux dependency used only when its explicit
execution backend is selected.
The opt-in padding-free Llama helper additionally uses a compatible optional
xFormers installation; it is not imported by ordinary training/data paths.
