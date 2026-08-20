# TrainOmni authoritative training-path coverage — 2026-08-21

This is the reconciled, authoritative inventory of training routes already
promised by Framework v1. It is intentionally finite: adding a row requires a
separate scope decision. The accepted target is the real composite
Qwen3.5-Vision + MiniCPM5-1B VLM; toy and public tiny-LLaVA runs prove Framework
mechanics but never upgrade a row to `accepted_real_vlm` by themselves.

The architecture is upstream-first. Standard numerical capabilities stay in
PyTorch, Transformers, PEFT, bitsandbytes, DCP, or an established delegated
trainer. TrainOmni owns the composite model/data contracts, capability
negotiation, immutable identity and lineage, exact-resume guarantees, and the
minimum glue needed to make those upstreams reproducible.

## State and integration vocabulary

- `accepted_real_vlm`: the real composite passed the applicable CLI, gradients,
  updates, precision/device, checkpoint/reload/resume, negative gates and a
  durable acceptance receipt.
- `implemented_needs_real_vlm`: Framework and the current target plugin expose a
  stable contract; the missing item is the bounded real-composite conformance run.
- `framework_gap`: Framework code or its local conformance proof is missing.
- `trainer_gap`: Framework has a stable boundary, but the target plugin, external
  backend, producer, dependency or real-run acceptance is not integrated yet.
- `explicitly_deferred`: deliberately outside the current execution program.
- `unsupported`: no native claim is made; validation must reject the combination.
- `direct_upstream`: configuration selects an upstream implementation directly.
- `thin_adapter`: TrainOmni validates identity/capability and passes through to an
  upstream without owning its numerical algorithm.
- `custom_required`: project-specific semantics are implemented locally because
  an upstream does not supply the required canonical multimodal identity,
  component evidence, or exact-resume contract. The precise gap is named per row.

## Frozen real-VLM routes

These routes are closed. Their acceptance artifacts are read-only evidence, not
an invitation to repeat or tune the fixed fixtures.

| Route ID | Promised route | Foundation | Integration | State | Real-VLM evidence and boundary |
|---|---|---|---|---|---|
| `vlm.align.connector` | Image/text modality alignment; connector trainable, vision and LLM frozen | PyTorch + Transformers | `custom_required` (frozen loop): canonical multimodal cursor, component policy and exact resume are not supplied together upstream | `accepted_real_vlm` | `VLMTraining/evidence/training-summary.json`, diagram alignment 240 steps; capability evidence only |
| `vlm.sft.connector` | Instruction SFT with connector only | PyTorch + Transformers | `custom_required` (frozen loop), same gap as alignment | `accepted_real_vlm` | `VLMTraining/evidence/training-summary.json`, InterGPS 64 steps; no quality claim |
| `vlm.sft.full` | True full-parameter SFT for vision, connector and LLM | PyTorch AdamW + Transformers activation checkpointing | `custom_required` (frozen): exact component-cover/update evidence and full data/RNG resume are project-specific | `accepted_real_vlm` | `VLMTraining/artifacts/p1-acceptance.json`; 1,182,802,176 trainable params, BF16, per-component gradients/updates, step-4 to step-8 exact resume |
| `vlm.kd.offline_dense` | Offline full-vocabulary cached-logit KD | PyTorch tensor kernels | `custom_required` (frozen): upstream trainers do not consume this immutable BF16 raw-logit cache with the required teacher/data/token/position binding | `accepted_real_vlm` | `VLMTraining/artifacts/kd-acceptance.json`; 16 steps, step-8 to step-16 exact resume, corruption gate; no teacher-quality claim |
| `vlm.dpo.offline_reference` | Offline-reference sigmoid DPO from FP32 per-token log-prob cache | PyTorch tensor kernels | `custom_required` (frozen): paired multimodal/cache identity and no-live-reference fail-closed contract are project-specific | `accepted_real_vlm` | `VLMTraining/artifacts/dpo-acceptance.json`; 16 steps, 13 negative cases, frozen backbones and exact resume; fixed synthetic pairs only |

## Lifecycle routes still in coverage

All native lifecycle rows reuse the already frozen masked-causal-LM behavior;
they do not define new loss methods.

| Route ID | Promised route | Foundation | Integration | State | Minimum real-VLM contract / owner |
|---|---|---|---|---|---|
| `vlm.vision_preparation` | Train the vision component through an image/text causal objective while connector/LLM policy is explicit | PyTorch + Transformers | `thin_adapter` over the frozen loop and target plugin | `implemented_needs_real_vlm` | Single CUDA, BF16, one deterministic sample, vision is the required updated component; checkpoint/reload and receipt. VLMTrainer runs it. |
| `vlm.cpt` | Multimodal continued pretraining with canonical `cpt` samples | PyTorch + Transformers | `thin_adapter` over frozen masked causal LM | `implemented_needs_real_vlm` | Single CUDA, BF16, explicit component policy, at least two optimizer steps and step-1 exact resume. VLMTrainer supplies a deterministic CPT fixture. |
| `vlm.curriculum` | A capability-curriculum stage and ordered multi-stage curriculum | PyTorch + Framework Pipeline | `custom_required`: artifact lineage and durable cross-stage input binding are TrainOmni-specific | `implemented_needs_real_vlm` | Two physical target-VLM stages with a checkpoint URI edge, stage-boundary reload, durable pipeline state and idempotent resume. |
| `vlm.distill.broader` | Live teacher or non-contract distillation recipes | TRL, NeMo or an authorized custom upstream trainer | `thin_adapter` delegated command/result contract | `trainer_gap` | Requires a pinned upstream launcher, formal backend recipe and produced acceptance receipt. Native fallback is forbidden. |
| `vlm.preference.broader` | Live-reference or non-contract preference optimization | TRL/Transformers trainers or another pinned upstream | `thin_adapter` delegated contract | `trainer_gap` | Requires backend-owned recipe/cache/model semantics. Native offline DPO must not silently substitute. |
| `vlm.reward_verifier` | Reward/verifier model training | TRL, NeMo, veRL or pinned custom upstream | `thin_adapter` delegated contract | `trainer_gap` | No target backend recipe/plugin is integrated. This row does not authorize inventing a reward method. |
| `vlm.online_rl` | GRPO/PPO/RLVR | TRL, veRL or pinned custom upstream | `thin_adapter` delegated contract | `trainer_gap` | Framework secure command conformance exists; a real-VLM upstream recipe, rollout boundary and result producer are missing. |
| `vlm.agentic_rl` | Agentic RL | veRL/AReaL or pinned custom upstream | `thin_adapter` delegated contract | `trainer_gap` | No real backend integration is present; do not map this to the native batch-loss loop. |

## Execution and optimization variants

Variants are orthogonal overlays, not a Cartesian-product promise. Each overlay
needs one representative real-VLM acceptance run; it inherits the frozen loss
semantics and must not be used to reopen those gates.

| Route ID | Capability | Foundation | Integration | State | Combination boundary and minimum contract |
|---|---|---|---|---|---|
| `exec.single.bf16` | Single-process CUDA BF16 | PyTorch autocast + Transformers | `direct_upstream` kernels, TrainOmni state glue | `accepted_real_vlm` | Covered by full SFT, KD and DPO receipts. |
| `exec.single.fp32` | Single-process FP32 | PyTorch | `direct_upstream` | `implemented_needs_real_vlm` | One connector-only target step, finite update, checkpoint/reload; no duplicate quality gate. |
| `exec.single.tf32` | CUDA TF32 matmul policy | PyTorch CUDA backend | `direct_upstream` | `implemented_needs_real_vlm` | CUDA only; receipt records resolved precision/device and a real parameter update. |
| `exec.single.fp16` | CUDA FP16 autocast + GradScaler | PyTorch AMP | `direct_upstream` | `implemented_needs_real_vlm` | CUDA only; scaler state must appear in checkpoint and compare under step-1 exact resume. |
| `exec.ddp` | Multi-process DDP and per-rank exact resume | `torch.distributed` DDP | `thin_adapter`: canonical distributed batch cursor and rank checkpoint glue | `trainer_gap` | Framework two-process toy proof exists, but target plugin declares only `single`. VLMTrainer must first conform and explicitly add `ddp`, then run 2 ranks, step 1→2 exact resume and cross-rank evidence. |
| `exec.fsdp2` | FSDP2 sharding, DCP checkpoint and same-world exact resume | PyTorch FSDP2 + DCP | `thin_adapter`: rank-local canonical runtime sidecars | `trainer_gap` | Framework two-process toy proof exists, but target plugin declares only `single`. After plugin conformance: 2 ranks, DCP model/optimizer, step 1→2 exact resume, single-process model-only reload. |
| `optim.component_policy` | Freeze/unfreeze, per-component LR/weight decay/dtype | PyTorch parameter groups | `custom_required`: composite component exact-cover is plugin-specific | `accepted_real_vlm` | Covered by connector-only and full-SFT receipts. |
| `optim.grad_clip` | Per-component gradient clipping | `torch.nn.utils.clip_grad_norm_` | `thin_adapter` | `accepted_real_vlm` | All accepted target recipes configured component clipping and completed finite updates. |
| `optim.grad_accumulation` | Accumulation greater than one with exact microstep resume | PyTorch autograd/DDP `no_sync` | `thin_adapter`: microstep and data cursor are checkpointed | `implemented_needs_real_vlm` | Framework conformance uses accumulation=2 and proves microstep 4 plus exact resume. Target contract: 4 microsteps/2 updates, checkpoint after update 1, full final equality. |
| `optim.activation_checkpointing` | Per-component non-reentrant activation checkpointing | Transformers component hooks | `thin_adapter`: plugin returns an exact typed receipt | `accepted_real_vlm` | Vision and LLM receipts plus exact resume are in P1 acceptance. No top-level silent fallback. |
| `optim.lora` | LoRA | PEFT | `thin_adapter` | `implemented_needs_real_vlm` | Framework uses real PEFT in a train/resume conformance. Target run must name exact modules, prove only adapter/modules-to-save params update, reload and exact resume. |
| `optim.qlora` | Plugin-loaded 4/8-bit base plus QLoRA | Transformers quantized load + PEFT | `thin_adapter` | `trainer_gap` | Framework validates quantized-model identity and calls PEFT k-bit preparation with no fallback. Target plugin currently has no quantized loader/capability; add it upstream-first before a real run. |
| `optim.torch_compile` | `torch.compile` single-process and DDP | PyTorch Dynamo/Inductor backend | `direct_upstream` with config pass-through | `implemented_needs_real_vlm` | Framework eager-backend compile conformance trains/checkpoints. Target single-CUDA route must pin backend/options, update params and reload. FSDP2 composition is not claimed. |
| `optim.adamw` | AdamW including explicit `foreach=false` | `torch.optim.AdamW` | `direct_upstream` plus immutable optimizer metadata | `accepted_real_vlm` | P1 records class, torch version, defaults, state dtypes and full-state exact resume. |
| `optim.adamw8bit` | Optional single-CUDA AdamW8bit | bitsandbytes `AdamW8bit`/`PagedAdamW8bit` | `thin_adapter` with explicit quantization config | `implemented_needs_real_vlm` | Install the pinned optional dependency, use existing target P1-shaped config, record package/version/quantization/state dtypes, prove step-1 resume. CPU or missing package fails without fallback. DDP/FSDP2 are not claimed. |
| `data.packing` | Token/media packing | PyTorch/Transformers model-specific collator | `thin_adapter` through `BatchPlan` and Model Plugin | `trainer_gap` | Target plugin declares `supports_packing=false`; plugin collator and real conformance are required before changing it. |
| `data.padding_free` | Padding-free batches | Upstream attention/model kernels + plugin collator | `thin_adapter` | `trainer_gap` | Target plugin declares `supports_padding_free=false`; no core emulation is allowed. |

## Pipeline, checkpoint, evaluation and export

| Route ID | Capability | Foundation | Integration | State | Evidence / remaining contract |
|---|---|---|---|---|---|
| `pipeline.physical_transfer` | Multi-stage physical checkpoint transfer, lineage and durable restart | PyTorch/Transformers stages | `custom_required`: artifact URI/parent/fingerprint DAG is project-specific | `implemented_needs_real_vlm` | Framework two-stage physical toy proof exists. Run the `vlm.curriculum` contract above on the target. |
| `checkpoint.local_exact` | Atomic single-process full-state exact resume | PyTorch state dict + trusted local pickle | `custom_required`: canonical reader/mixture/batch/RNG state is not in an upstream model checkpoint | `accepted_real_vlm` | Connector resume, P1, KD and DPO acceptance receipts. |
| `checkpoint.model_only_transfer` | Reset optimizer/RNG/data and load only model into the next stage | PyTorch state dict | `thin_adapter` plus artifact lineage | `accepted_real_vlm` | InterGPS SFT→P1→KD→DPO physical lineage. |
| `checkpoint.ddp_exact` | Same-world DDP exact resume | PyTorch DDP + rank-local local checkpoints | `thin_adapter` | `trainer_gap` | Blocked by target `ddp` plugin conformance; follows `exec.ddp`. |
| `checkpoint.fsdp2_exact` | Same-world FSDP2 exact resume | PyTorch DCP | `thin_adapter` plus rank-local runtime sidecars | `trainer_gap` | Blocked by target `fsdp2` conformance; follows `exec.fsdp2`. |
| `checkpoint.dcp_model_only_reshard` | FSDP2 DCP model-only load into one process/new topology | PyTorch DCP | `direct_upstream` with plugin export glue | `trainer_gap` | Framework toy proof exists; target FSDP2 producer is missing. |
| `evaluate.local` | Checkpointed target evaluation on configured device/precision | PyTorch inference/autocast + plugin objective | `thin_adapter` | `accepted_real_vlm` | `VLMTraining/evidence/evaluation-summary.json` records CUDA BF16 local execution. |
| `evaluate.delegated` | External benchmark harness | lmms-eval or another pinned harness | `thin_adapter` shell-free command boundary | `trainer_gap` | Framework authorization/result contract exists; target harness recipe and acceptance artifact are missing. |
| `export.plugin` | Safe deploy/export through model plugin | Transformers/safetensors or plugin-selected upstream serializer | `thin_adapter` | `implemented_needs_real_vlm` | Target plugin supports `torch`; export an accepted checkpoint, reload exported weights and compare a deterministic forward. HF/safetensors is not claimed by that plugin. |

## Delegated backend boundaries

| Route ID | Backend claim | Foundation | Integration | State | Boundary |
|---|---|---|---|---|---|
| `backend.generic` | Authorized external stage command | Any pinned upstream launcher | `thin_adapter` | `implemented_needs_real_vlm` | Framework validates argv/trust, redacts request, executes shell-free and collects physical artifacts. A backend-specific numerical claim belongs to another row. |
| `backend.trl` | TRL-owned standard post-training | TRL | `thin_adapter` | `trainer_gap` | Current adapter is only the generic command envelope; pinned launcher and recipe translator/result producer are missing. |
| `backend.nemo` | NeMo-owned scale/post-training | NVIDIA NeMo | `thin_adapter` | `trainer_gap` | Same boundary; no package execution or real target result is claimed. |
| `backend.verl` | veRL-owned RL/RLVR | veRL | `thin_adapter` | `trainer_gap` | Same boundary; rollout and result producer are missing. |
| `backend.veomni` | VeOmni scale backend | `veomni==0.1.11` | `thin_adapter` pinned bridge | `trainer_gap` | Versioned bridge conformance exists, actual package execution does not. Exact resume is rejected until data/RNG/topology conformance passes. |

## Explicitly deferred and unsupported boundaries

| Route ID | Boundary | Foundation | Integration | State | Fail-closed rule |
|---|---|---|---|---|---|
| `platform.ascend_multinode` | Ascend multi-node | Future VeOmni/torch-npu upstream | `thin_adapter` when re-prioritized | `explicitly_deferred` | No current implementation claim. |
| `modality.audio_understanding` | Audio encoder/projector understanding | Future Transformers/audio upstream + plugin | `thin_adapter` | `explicitly_deferred` | Canonical assets are representable, but no audio model plugin is claimed. |
| `generation.diffusion` | Image/video/audio diffusion or flow-matching generation | Future dedicated upstream | `thin_adapter` | `explicitly_deferred` | Outside v1 objective and engine families. |
| `native.fp8` | Native torch-loop FP8 | — | — | `unsupported` | FP8 is delegated only. |
| `native.tp_pp_cp_sp_ep` | Native TP/PP/CP/SP/EP | — | — | `unsupported` | Use a pinned delegated scale backend; torch native validation does not advertise these modes. |
| `native.live_kd_dpo` | Live teacher/reference KD/DPO in native loop | — | — | `unsupported` | The native objectives accept only their immutable offline caches; use delegated upstream routes. |
| `native.reward_rl_agentic` | Reward, online RL or agentic RL in native loop | — | — | `unsupported` | These stages are delegated; no silent masked-LM substitution. |
| `resume.fsdp2_changed_world_exact` | Exact FSDP2 resume with changed world size | — | — | `unsupported` | Runtime data/RNG state is topology-specific; model-only DCP reshard remains available. |
| `resume.veomni_exact` | VeOmni exact resume | — | — | `unsupported` | Bridge capability negotiation rejects `resume_level: exact`. |
| `compile.fsdp2` | `torch.compile` + FSDP2 composition | — | — | `unsupported` | Backend/version dependent and not claimed in v1. |
| `adamw8bit.distributed` | AdamW8bit under DDP/FSDP2 | — | — | `unsupported` | Only single-CUDA is claimed pending real conformance. |

There are no remaining `framework_gap` rows in this inventory. The former local
gaps for accumulation>1, LoRA, QLoRA adapter invocation and `torch.compile` now
have Framework conformance tests. That statement does not upgrade their target
states: real-composite execution or target-plugin work remains explicitly listed.

## Stable VLMTrainer acceptance contract

Every new real-route receipt should use a compact JSON object with at least:

```json
{
  "schema_version": "trainomni.training-route-acceptance.v1",
  "route_id": "optim.grad_accumulation",
  "accepted": true,
  "run_fingerprint": "sha256",
  "upstream": {
    "foundation": ["torch"],
    "versions": {"torch": "exact version"},
    "integration": "thin_adapter"
  },
  "lineage": {
    "model_plugin": {"id": "...", "version": "..."},
    "input_artifacts": [],
    "data_identity": "sha256"
  },
  "execution": {
    "device": "cuda:0",
    "precision": "bf16",
    "world_size": 1,
    "optimizer_steps": 2,
    "microsteps": 4
  },
  "evidence": {
    "finite_gradients": true,
    "actual_parameter_updates": true,
    "checkpoint_reload": true,
    "exact_resume": true
  },
  "artifacts": {},
  "claim_boundary": "engineering capability on a fixed deterministic fixture"
}
```

Additional route-specific fields are mandatory when stated in the tables (for
example scaler state for FP16, rank evidence for DDP, DCP metadata for FSDP2,
adapter-only changed parameters for LoRA, or quantization identity for
AdamW8bit). A receipt with `accepted=true` must point to existing physical
artifacts and must not claim quality or generalization.

## Composition rule

Do not run every combination. Use the smallest representative set that covers
each non-deferred overlay once. A valid order is:

1. lifecycle: vision preparation, CPT, then a two-stage curriculum Pipeline;
2. single-process overlays: FP32, TF32, FP16, accumulation=2, LoRA,
   `torch.compile`, AdamW8bit, export and delegated evaluation;
3. target-plugin additions: QLoRA/quantized load, packing/padding-free;
4. distributed target-plugin conformance: DDP, then FSDP2/DCP;
5. pinned delegated upstreams only after their existing method/recipe semantics
   are available; do not invent those semantics in Framework.

Frozen accepted routes remain inputs and correctness oracles; they are not rerun.
