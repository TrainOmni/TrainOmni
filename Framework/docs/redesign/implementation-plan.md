# TrainOmni replacement framework implementation plan

Status: architecture execution plan, 2026-08-21.

This plan replaces the archived experimental architecture. It does not reopen or
extend that design. The archive remains the sole reference for previous work.

## 1. Accepted boundaries

- Framework code and concrete task/run content are separate.
- TrainOmni is one Python distribution. Module isolation does not mean one wheel,
  environment, or service per module.
- Runtime foundation is limited to small, deliberate base libraries. Full training
  frameworks listed in `docs/research/upstream-sources.md` are external source
  references only: no imports,
  subprocess execution, delegated backends, or runtime configuration coupling.
- A concrete model combination must be added by registering modules and writing a
  task specification. Core code must never branch on a model name.
- No module receives a flattened global configuration dictionary.
- Generated runs, caches, datasets, checkpoints, and task packages do not live in
  the Framework source tree.
- Image/video understanding is first priority. Audio understanding is a reserved
  modality extension. Diffusion/generative-media training has a separate future
  model/objective ABI and is explicitly deferred.

## 2. Framework, task, and run

The command boundary is:

```text
trainomni train --task <task.yaml> --run <run.yaml>
```

`task.yaml` answers **what is being learned**:

- model assembly or monolithic model module;
- modality encoders, connector, fusion, and language component;
- model I/O and special-token semantics;
- canonical datasets and semantic transforms;
- supervision construction and objective;
- parameter policy: frozen/full/adapter and component selection;
- evaluation semantics.

`run.yaml` answers **how this attempt executes**:

- seed, output directory, and restart policy;
- optimizer and scheduler values;
- batch size, accumulation, clipping, and step/epoch limit;
- precision, attention kernel, compilation, and activation checkpointing;
- single-process/DDP/FSDP2 topology;
- checkpoint frequency, logging, profiling, and resource limits.

The framework resolves them separately and records separate immutable digests.
A run may change execution settings without mutating the task identity. Exact
resume validates both identities and the explicitly resumable run fields.

### 2.1 Platform launch is not training configuration

Shell/process startup is isolated under `launch/windows/` and `launch/linux/`.
Both adapters require an explicit absolute `TRAINOMNI_PYTHON` and forward the
same `python -m trainomni` CLI. They do not activate or install environments and
cannot select a device, model, dataset, output path, or training behavior.

Distributed process creation is a separate platform adapter below
`launch/<platform>/distributed/`. Logical topology remains RunSpec semantics;
host facts such as scheduler allocation, node rank and rendezvous endpoint are
launcher inputs recorded in a launch receipt. Windows now exposes only a
certified one-rank probe; Linux exposes the upstream torchrun boundary, while real
multi-rank acceptance remains a server gate. This lets Windows and Linux differ
in shell and process mechanics without forking the framework API.

## 3. Module model

Every code-bearing variation point is a typed module. A module owns:

- one namespaced configuration type;
- one narrow protocol implementation;
- declared inputs, outputs, capabilities, and incompatibilities;
- state serialization rules when it owns state;
- focused contract tests;
- provenance metadata if derived from upstream source.

Modules are registered by `(kind, namespace, name, version)`. Resolution returns a
typed module descriptor, never a raw Python file and never mutates `sys.path`.
Modules cannot inspect unrelated task or run configuration.

A one-off task may carry a hash-pinned local module. It must use the same
descriptor/protocol contracts as a builtin module, declare a module.toml, stay
under the task root, and be explicitly enabled by the caller. The loader uses a
synthetic package namespace without changing sys.path; the source digest enters
task and checkpoint identity. This mechanism is not a sandbox for untrusted code.

The initial module kinds are:

| Kind | Variable behavior owned |
| --- | --- |
| `data_source` | storage/format access and resumable sample cursor |
| `sample_transform` | normalization, filtering, media loading and augmentation |
| `model_io` | processor/tokenizer/template/special tokens and tensor field mapping |
| `supervision` | labels, loss masks, target positions and preference branches |
| `packer` | length policy, multimodal-safe packing and sample boundaries |
| `collator` | padding and final typed batch construction |
| `encoder` | image/video, later audio, feature production |
| `connector` | feature dimension/sequence transformation |
| `fusion` | replacement, prefix, cross-attention, Q-Former or layered injection |
| `language` | embedding/decoder/output-head boundary |
| `model` | monolithic VLM or composite assembly and forward capability |
| `attention_policy` | semantic mask/pattern requirements |
| `objective` | forward plan and scalar loss computation |
| `parameter_policy` | freeze/full/adapter selection and optimizer groups |
| `evaluator` | held-out metric semantics |
| `exporter` | artifact conversion and deployment-facing output |

Optimizer, scheduler, precision and distributed topology are run configuration and
runtime services, not independently registered semantic modules.

## 4. Stable data contracts

The pipeline is explicit:

```text
DataSource
  -> OmniSample
  -> SampleTransform(s)
  -> ModelIO.encode
  -> Supervision.annotate
  -> Packer
  -> Collator
  -> OmniBatch
```

`OmniSample` contains ordered content blocks rather than model-specific fields:

- text;
- image;
- video;
- reserved audio;
- structured metadata and stable sample identity.

`ModelIO` is the only layer allowed to translate canonical content into a concrete
processor/template representation. `Supervision` is the only layer that constructs
labels and masks. Consequently, changing a template does not change the source,
and changing a loss mask does not require modifying a collator or model.

## 5. Model and attention contracts

Two model paths are first-class:

1. `MonolithicModelModule` wraps a complete Transformers-compatible VLM.
2. `CompositeModelModule` assembles encoder(s), connector, fusion, and language
   modules for custom ViT + LLM combinations.

An encoder returns a typed `ModalFeatures` object containing embeddings plus
spatial/temporal grid, validity mask, positions and modality metadata. A connector
accepts and returns `ModalFeatures`. Fusion owns where and how those features enter
the language computation.

Attention has four separate change points:

| Change | Owner |
| --- | --- |
| SDPA/Flash/eager/NPU kernel | run-level `attention_kernel` runtime service |
| causal/prefix/bidirectional/block semantic mask | task-level `attention_policy` |
| MHA/GQA/MQA/sliding-window/RoPE/QKV architecture | encoder/language/model module |
| cross-modal replacement/prefix/cross-attention/Q-Former | fusion module |

The model declares which semantic policies and runtime kernels it supports. The
resolver rejects unsupported combinations before loading data or allocating the
full model.

## 6. Objective and loss contract

Loss is not embedded in the trainer or data collator. An objective has two stages:

```text
Objective.plan(batch) -> ForwardPlan[ForwardRequest]
runtime executes model forwards
Objective.compute(batch, outputs, state) -> LossBundle
```

This represents one-forward causal LM, paired DPO forwards, online/offline teacher
distillation and auxiliary component losses without giving the objective control of
the optimizer or distributed lifecycle.

`LossBundle` contains total loss, named terms, weights, denominators and metrics.
All reductions and normalization rules are objective-owned and testable against a
small numerical oracle. Supervised positions originate in `supervision` and are
revalidated by the objective before computation.

Initial objectives, in implementation order:

1. causal language modeling for CPT/connector alignment/SFT;
2. dense-logit distillation;
3. sigmoid DPO with live or cached reference inputs represented explicitly;
4. additional objectives only after their data and numerical contracts are frozen.

## 7. Runtime kernel

TrainOmni owns a small PyTorch runtime rather than wrapping another Trainer:

- device and process-group initialization;
- deterministic seeding and state capture;
- model/optimizer/scheduler construction;
- gradient accumulation, AMP, clipping and optimizer step;
- objective forward-plan execution;
- metrics/events;
- checkpoint/save/load/exact resume;
- evaluation and export invocation.

Distributed execution uses PyTorch primitives directly. The order is single GPU,
DDP, then composable FSDP2. Tensor/pipeline/context parallelism is not claimed until
a model module and checkpoint path both pass real distributed verification.
Ascend support is deferred until the CUDA/PyTorch path is stable; it will enter via
a device/distributed runtime adapter, not model or objective branches.

## 8. Parameter adaptation

`ParameterPolicy` selects components and produces explicit parameter groups. The
first policies are:

- connector-only;
- selected-component full parameter;
- complete full parameter;
- native LoRA injection with explicit target resolution and adapter artifacts.

PEFT is a source reference, not a runtime dependency. Only the narrow LoRA behavior
we actually support will be implemented and tested. QLoRA is not silently equated
with LoRA: it requires an explicit quantization backend and remains unsupported
until that backend has a reproducible CUDA and checkpoint contract.

## 9. Checkpoint and artifact rules

Every checkpoint records:

- task digest and run digest;
- resolved module IDs, versions and configuration digests;
- model/base-weight identity;
- model, optimizer, scheduler, scaler and RNG state;
- data cursor, sampler/packer state and global counters;
- distributed topology and precision/kernel identity;
- framework version and source provenance.

Resume is fail-closed for semantic or state incompatibility. Export is separate from
checkpointing and must not erase training lineage.

## 10. Upstream extraction map

| Source | What to study or selectively adapt | What not to copy |
| --- | --- | --- |
| Transformers | model/processor contracts, attention dispatch | the whole Trainer lifecycle |
| PEFT | LoRA layers, injection and target resolution | full tuner matrix and compatibility surface |
| Accelerate | device placement, precision and distributed lifecycle edge cases | runtime dependency or state ownership |
| TRL | loss numerics, sequence log-prob utilities, VLM preference tests | Trainer classes and dataset ownership |
| ms-swift | VLM templates, loss masks, model metadata and adapter targets | CLI, global arguments, backend branches |
| LLaMA-Factory | task/data UX and configuration lessons | trainer stack and model-name conditionals |
| VeOmni | multimodal pipeline, operation registry and distributed recipes | framework runtime delegation |
| TorchTitan | typed component configs, FSDP2/activation-checkpoint/DCP patterns | complete platform or model zoo |
| NeMo AutoModel | VLM composition and large-scale recipe constraints | NVIDIA-specific framework lifecycle |

No code is copied during architecture work. Extraction requires an identified gap,
a minimal target module, license review, provenance entry and focused tests.

## 11. Delivery sequence and exit gates

Execution is vertical-slice-first. Phase 1 defines only the contracts needed by
Phase 2; Phase 2 must train and resume a tiny composite VLM before the framework
adds broader formats, objectives, tuning methods, or distributed modes. This keeps
contract design grounded in an executable path and prevents another wide but
weakly integrated framework.

### Phase 0: source and architecture lock

Deliverables:

- pinned local reference manifest;
- this implementation plan;
- final dependency policy and directory layout;
- deletion/replacement map for archived implementation files.

Exit: every old subsystem is explicitly retained, replaced or removed; no ambiguous
parallel architecture remains.

### Phase 1: module kernel and split configuration

Deliverables:

- typed module descriptor/registry/resolver;
- typed `TaskSpec` and `RunSpec` with separate digests;
- scoped configuration injection and capability negotiation;
- CLI resolve/inspect commands.

Exit: dummy modules compose without global configuration or model-name branches;
negative capability tests fail before model construction.

### Phase 2: first end-to-end vertical slice

Deliverables:

- the minimum `OmniSample` image-text path and in-memory fixture source;
- one composite tiny VLM: encoder, connector, fusion and language components;
- causal-LM supervision and objective;
- one single-device optimizer step loop;
- model/optimizer/RNG/data-cursor checkpoint and exact resume.

Exit: a tiny composite VLM completes load, forward, objective, backward, optimizer
step, checkpoint and fresh-process exact resume using separate task/run files. No
phase after this one may proceed while that vertical path is broken.

### Phase 3: canonical multimodal data expansion

Deliverables:

- complete source/transform/model-I/O/supervision/packer/collator contracts;
- JSONL and deterministic streaming source modules;
- text-only, single-image, multi-image and video-shaped samples;
- multimodal-safe packing and resumable sample/packer state.

Exit: all fixture forms produce stable batches and resume exactly across a process
restart without model-specific branches in the data source or collator.

### Phase 4: model composition and attention

Deliverables:

- monolithic and composite model contracts;
- encoder/connector/fusion/language interfaces;
- semantic attention policy and runtime-kernel negotiation;
- component identity and parameter grouping.

Exit: tiny VLM performs encode, fusion and logits forward; incompatible attention
policy/kernel combinations fail during preflight.

### Phase 5: objectives and supervision

Deliverables:

- forward plan, loss bundle and objective state contracts;
- causal LM objective and numerical oracle;
- dense-logit KD and DPO implemented only against their frozen contracts.

Exit: each objective passes numerical, masking, gradient-routing and corruption
tests without trainer/data special cases.

### Phase 6: parameter policies

Deliverables:

- connector/full/component policies;
- minimal native LoRA module and adapter artifact format;
- changed-parameter evidence by named component.

Exit: every required component has deterministic gradient and update evidence;
save/load preserves exact trainable state.

### Phase 7: training runtime hardening and exact resume

Deliverables:

- single-device PyTorch runtime;
- optimizer/scheduler/precision/accumulation/clipping;
- checkpoint manager and exact resume;
- structured metrics and resource evidence.

Exit: uninterrupted and resumed tiny-VLM runs are state-equivalent, and all required
states are present in checkpoints.

### Phase 8: evaluation and export

Deliverables:

- evaluator registry and held-out data path;
- inference-mode/device/precision-correct evaluation;
- Transformers/safetensors export with lineage receipt.

Exit: exported artifact reloads in a fresh process and matches reference logits
within the declared precision tolerance.

### Phase 9: distributed verification

Deliverables:

- DDP and FSDP2 runtime services;
- distributed checkpoint and topology-aware resume;
- activation checkpoint policies per model component.

Exit: two-process tiny-VLM forward/backward/save/resume passes; then a real VLM
smoke validates the same path. No unsupported parallel mode is advertised.

Current status: direct DDP and FSDP2 plus portable state are implemented and
real-VLM verified with a world-size-one CUDA process group. The required
two-process/multi-host Linux gate remains pending; see
`../architecture/distributed-execution.md`.

### Phase 10: consumer integration

Only after Phases 1-8 are stable is the concrete Qwen vision + MiniCPM language
combination registered outside Framework. Integration should consist of model/data
modules plus task/run files; framework changes indicate a missing generic contract
and require a focused design review.

## 12. Definition of basically ready

Framework is ready to reconnect to the consumer task when:

- Phases 1-8 pass automated tests;
- a tiny monolithic VLM and tiny composite VLM both complete train, checkpoint,
  exact resume, evaluation and export;
- causal LM, KD and DPO objectives use the same runtime without route-specific
  trainer classes;
- loss, attention semantics, attention kernel and fusion each have an independent
  documented change point;
- Framework contains no concrete run output, dataset, target checkpoint path or
  target-model name branch;
- the README gives one task file, one run file and one command for a complete smoke.
