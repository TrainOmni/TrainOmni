"""Optional native PyTorch training loop.

The module imports PyTorch lazily so validation, data inspection, pipeline
planning, and non-Torch adapters remain usable in a lightweight environment.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trainomni.checkpoint import (
    DCPCheckpointManager,
    LocalCheckpointManager,
    ObjectState,
    PythonRandomState,
    ScalarState,
    StateRegistry,
    TorchRandomState,
)
from trainomni.config import ResolvedRunSpec
from trainomni.contracts import ArtifactRef, ValidationIssue, ValidationReport
from trainomni.models import (
    ActivationCheckpointingReceipt,
    ActivationCheckpointingRequest,
    ModelBatch,
    ModelBundle,
)
from trainomni.objectives import ObjectiveBinding, ObjectiveSetup

from .peft import PeftError, apply_peft_if_requested
from .protocol import (
    EngineCapabilities,
    EngineKind,
    EngineManifest,
    PreparedStage,
    StageResult,
)


class TorchEngineError(RuntimeError):
    pass


_TRAINING_STAGES = frozenset(
    {
        "vision_preparation",
        "modality_alignment",
        "multimodal_pretraining",
        "capability_curriculum",
        "instruction_sft",
        "reasoning_distillation",
        "reward_verifier",
        "offline_preference",
    }
)


@dataclass(slots=True)
class TorchStageContext:
    resolved: ResolvedRunSpec
    plugin: Any
    bundle: ModelBundle
    objective: ObjectiveBinding
    batches: Iterable[ModelBatch]
    output_dir: Path
    input_artifacts: Mapping[str, ArtifactRef] = field(default_factory=dict)
    resume_from: str | None = None
    trusted_resume: bool = False
    callbacks: tuple[Any, ...] = ()


@dataclass(slots=True)
class _TorchPreparedState:
    context: TorchStageContext
    torch: Any
    model: Any
    optimizer: Any
    scheduler: Any | None
    scaler: Any | None
    device: Any
    registry: StateRegistry
    checkpoint_manager: LocalCheckpointManager
    component_assignments: Mapping[str, tuple[str, ...]]
    activation_checkpointing: Mapping[str, Mapping[str, Any]]
    trainable_numel_by_component: Mapping[str, int]
    objective_setup: ObjectiveSetup | None = None
    step: ScalarState = field(default_factory=ScalarState)
    microstep: ScalarState = field(default_factory=ScalarState)
    tokens: ScalarState = field(default_factory=ScalarState)
    last_checkpoint: ArtifactRef | None = None
    latest_evidence: Mapping[str, Any] = field(default_factory=dict)
    latest_objective_metrics: Mapping[str, float] = field(default_factory=dict)
    objective_counts: dict[str, ScalarState] = field(default_factory=dict)


@dataclass(slots=True)
class _UpdateProbe:
    component_id: str
    parameter_name: str
    parameter: Any
    before: Any


class TorchEngine:
    manifest = EngineManifest(
        engine_id="torch",
        engine_version="1.0.0",
        kind=EngineKind.LOOP,
        capabilities=EngineCapabilities(
            stage_types=_TRAINING_STAGES,
            objectives=frozenset(
                {
                    "masked-causal-lm",
                    "offline-dense-logit-kd",
                    "offline-reference-dpo",
                }
            ),
            parallelism=frozenset({"single", "ddp", "fsdp2"}),
            precisions=frozenset({"fp32", "tf32", "fp16", "bf16"}),
            resume_levels=frozenset(
                {"exact", "stage_boundary", "weights_only", "transfer"}
            ),
            supports_generation=False,
            supports_multiple_models=True,
        ),
        dependency_constraints=("torch>=2.4",),
    )

    def validate(self, stage: Any, model: Any) -> ValidationReport:
        issues: list[ValidationIssue] = []
        try:
            torch = import_torch()
        except TorchEngineError as exc:
            issues.append(
                ValidationIssue(
                    code="engine.dependency.torch",
                    message=str(exc),
                    path="stage.engine.backend",
                )
            )
            return ValidationReport(tuple(issues))
        if not isinstance(model, ModelBundle):
            issues.append(
                ValidationIssue(
                    code="engine.model_bundle",
                    message="torch engine requires ModelBundle",
                    path="model.plugin",
                )
            )
        if stage.engine.parallelism == "fsdp2" and not _has_fsdp2(torch):
            issues.append(
                ValidationIssue(
                    code="engine.fsdp2.unavailable",
                    message="installed PyTorch does not expose fully_shard/FSDP2",
                    path="stage.engine.parallelism",
                )
            )
        return ValidationReport(tuple(issues))

    def prepare(self, context: TorchStageContext) -> PreparedStage:
        if not isinstance(context, TorchStageContext):
            raise TorchEngineError("torch engine expects TorchStageContext")
        torch = import_torch()
        report = self.validate(context.resolved.run.stage, context.bundle)
        if not report.valid:
            details = "; ".join(item.message for item in report.issues)
            raise TorchEngineError(details)
        stage = context.resolved.run.stage
        device = select_torch_device(torch, stage.engine.config)
        original_model = context.bundle.model
        assignments = _component_assignments(context, original_model)
        _apply_component_policy(torch, context, original_model, assignments)
        activation_checkpointing = _configure_activation_checkpointing(context)
        try:
            train_model = apply_peft_if_requested(original_model, stage)
        except PeftError as exc:
            raise TorchEngineError(str(exc)) from exc
        uses_qlora = any(
            policy.peft is not None and policy.peft.method == "qlora"
            for policy in stage.component_policy.values()
        )
        if not uses_qlora:
            train_model.to(device)
        configure_torch_precision(
            torch, stage.engine.precision, device
        )
        execution_model = _wrap_parallel(
            torch, train_model, stage.engine.parallelism, device
        )
        compile_config = stage.engine.config.get("compile", False)
        if compile_config:
            if not hasattr(torch, "compile"):
                raise TorchEngineError("torch.compile was requested but is unavailable")
            if compile_config is True:
                compile_kwargs = {}
            elif isinstance(compile_config, Mapping):
                compile_kwargs = dict(compile_config)
            else:
                raise TorchEngineError("engine.config.compile must be bool or mapping")
            execution_model = torch.compile(execution_model, **compile_kwargs)
        optimizer = _build_optimizer(
            torch, context, train_model, assignments, device
        )
        scheduler = _build_scheduler(torch, optimizer, context.resolved)
        scaler = _build_scaler(torch, stage.engine.precision, device)
        checkpoint_root = context.output_dir / "checkpoints"
        if stage.engine.parallelism == "fsdp2":
            manager: Any = DCPCheckpointManager(checkpoint_root, torch)
        else:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("RANK", "0"))
            if world_size > 1:
                checkpoint_root = checkpoint_root / f"rank-{rank:05d}"
            manager = LocalCheckpointManager(checkpoint_root)
        state = _TorchPreparedState(
            context=context,
            torch=torch,
            model=execution_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            registry=StateRegistry(),
            checkpoint_manager=manager,
            component_assignments=assignments,
            activation_checkpointing=activation_checkpointing,
            trainable_numel_by_component=_optimizer_trainable_numel(optimizer),
        )
        _validate_diagnostics_contract(state)
        state.objective_setup = _setup_objective(state)
        if stage.engine.parallelism != "fsdp2":
            state.registry.register("model", ObjectState(train_model))
            state.registry.register("optimizer", ObjectState(optimizer))
        if scheduler is not None:
            state.registry.register("scheduler", ObjectState(scheduler))
        if scaler is not None:
            state.registry.register("grad_scaler", ObjectState(scaler))
        state.registry.register("step", state.step)
        state.registry.register("microstep", state.microstep)
        state.registry.register("tokens", state.tokens)
        if state.objective_setup is not None:
            for key in state.objective_setup.state_count_keys:
                counter = ScalarState()
                state.objective_counts[key] = counter
                state.registry.register(f"objective_count_{key}", counter)
        state.registry.register("python_rng", PythonRandomState())
        state.registry.register("torch_rng", TorchRandomState())
        state.registry.register("data", context.batches)  # type: ignore[arg-type]
        if context.resume_from:
            resume_manager, resume_name = _resume_location(
                context.resume_from,
                manager,
                torch,
                fsdp2=stage.engine.parallelism == "fsdp2",
            )
            if stage.engine.parallelism == "fsdp2":
                metadata = resume_manager.load(
                    resume_name,
                    model=execution_model,
                    optimizer=optimizer,
                    runtime=state.registry,
                    trusted=context.trusted_resume,
                )
            else:
                metadata = resume_manager.load(
                    resume_name,
                    state.registry,
                    trusted=context.trusted_resume,
                )
            expected = context.resolved.fingerprint
            if metadata.get("run_fingerprint") != expected:
                raise TorchEngineError("resume checkpoint run fingerprint mismatch")
            expected_optimizer = metadata.get("optimizer")
            if not isinstance(expected_optimizer, Mapping):
                raise TorchEngineError(
                    "exact resume checkpoint lacks optimizer contract metadata"
                )
            actual_optimizer = _optimizer_metadata(state)
            if dict(expected_optimizer) != actual_optimizer:
                raise TorchEngineError(
                    "resume optimizer identity/state-dtype contract mismatch"
                )
            actual_objective = _objective_metadata(state)
            if actual_objective is not None:
                expected_objective = metadata.get("objective")
                if not isinstance(expected_objective, Mapping):
                    raise TorchEngineError(
                        "exact resume checkpoint lacks objective identity metadata"
                    )
                if dict(expected_objective) != actual_objective:
                    raise TorchEngineError(
                        "resume objective/cache identity contract mismatch"
                    )
        _reset_peak_memory(state)
        return PreparedStage(stage.stage_id, state)

    def run(self, prepared: PreparedStage) -> StageResult:
        state = _prepared_state(prepared)
        torch = state.torch
        stage = state.context.resolved.run.stage
        max_steps = stage.optimization.max_steps
        max_tokens = stage.optimization.max_tokens
        accumulation = stage.optimization.gradient_accumulation_steps
        if max_steps is None and max_tokens is None:  # schema guards this
            raise TorchEngineError("training budget is missing")
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        latest_metrics: dict[str, float] = {}
        while True:
            at_update_boundary = int(state.microstep.value) % accumulation == 0
            if max_steps is not None and state.step.value >= max_steps:
                break
            if (
                max_tokens is not None
                and state.tokens.value >= max_tokens
                and at_update_boundary
            ):
                break
            try:
                batch = next(iter(state.context.batches))
            except StopIteration as exc:
                raise TorchEngineError(
                    "data stream exhausted before the configured training budget"
                ) from exc
            prepared_batch = state.context.objective.objective.prepare(batch, state)
            prepared_batch = move_model_batch(prepared_batch, state.device)
            autocast = torch_autocast_context(
                torch, stage.engine.precision, state.device
            )
            sync_now = (int(state.microstep.value) + 1) % accumulation == 0
            no_sync = getattr(state.model, "no_sync", None)
            sync_context = (
                nullcontext()
                if sync_now or not callable(no_sync)
                else no_sync()
            )
            with sync_context:
                with autocast:
                    loss_output = state.context.objective.objective.compute(
                        {"model": state.model, **state.context.bundle.auxiliary_models},
                        prepared_batch,
                    )
                    loss = loss_output.total / accumulation
                if state.scaler is not None:
                    state.scaler.scale(loss).backward()
                else:
                    loss.backward()
            state.microstep.value += 1
            state.tokens.value += int(loss_output.counts.get("loss_tokens", 0))
            _update_objective_counts(state, loss_output.counts)
            state.latest_objective_metrics = {
                key: float(value) for key, value in loss_output.metrics.items()
            }
            latest_metrics = dict(state.latest_objective_metrics)
            if not sync_now:
                continue
            if state.scaler is not None:
                state.scaler.unscale_(state.optimizer)
            evidence, update_probes = _capture_step_evidence(state)
            _clip_gradients(
                torch, state.context, state.model, state.component_assignments
            )
            if state.scaler is not None:
                state.scaler.step(state.optimizer)
                state.scaler.update()
            else:
                state.optimizer.step()
            evidence = _finish_step_evidence(state, evidence, update_probes)
            state.optimizer.zero_grad(set_to_none=True)
            if state.scheduler is not None:
                state.scheduler.step()
            state.step.value += 1
            latest_metrics["step"] = float(state.step.value)
            latest_metrics["loss_tokens"] = float(state.tokens.value)
            latest_metrics.update(_evidence_metrics(evidence))
            state.latest_evidence = evidence
            for callback in state.context.callbacks:
                callback("step_end", state, latest_metrics)
            every = stage.checkpoint.every_steps
            if every and int(state.step.value) % every == 0:
                self.checkpoint(prepared, "interval")
        if state.last_checkpoint is None or (
            stage.checkpoint.every_steps
            and int(state.step.value) % stage.checkpoint.every_steps != 0
        ):
            self.checkpoint(prepared, "stage_end")
        return StageResult(
            stage_id=prepared.stage_id,
            status="succeeded",
            outputs={"checkpoint": state.last_checkpoint},
            metrics=latest_metrics,
            metadata=_training_metadata(state),
        )

    def checkpoint(self, prepared: PreparedStage, reason: str) -> ArtifactRef:
        state = _prepared_state(prepared)
        name = f"step-{int(state.step.value):08d}"
        checkpoint_path: Path
        if name not in state.checkpoint_manager.list_complete():
            metadata = {
                "run_fingerprint": state.context.resolved.fingerprint,
                "stage_id": prepared.stage_id,
                "step": int(state.step.value),
                "tokens": int(state.tokens.value),
                "reason": reason,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "rank": int(os.environ.get("RANK", "0")),
                **_training_metadata(state),
            }
            if state.context.resolved.run.stage.engine.parallelism == "fsdp2":
                checkpoint_path = state.checkpoint_manager.save(
                    name,
                    model=state.model,
                    optimizer=state.optimizer,
                    runtime=state.registry,
                    metadata=metadata,
                )
            else:
                checkpoint_path = state.checkpoint_manager.save(
                    name,
                    state.registry,
                    metadata=metadata,
                )
        else:
            checkpoint_path = state.checkpoint_manager.root / name
        if state.torch.distributed.is_available() and state.torch.distributed.is_initialized():
            state.torch.distributed.barrier()
        reference = ArtifactRef(
            artifact_id=f"{state.context.resolved.run.name}/{name}",
            uri=str(checkpoint_path.resolve()),
        )
        state.last_checkpoint = reference
        return reference

    def collect(self, result: StageResult) -> StageResult:
        return result


def _prepared_state(prepared: PreparedStage) -> _TorchPreparedState:
    if not isinstance(prepared.state, _TorchPreparedState):
        raise TorchEngineError("prepared stage does not belong to torch engine")
    return prepared.state


def _resume_location(
    value: str,
    default_manager: Any,
    torch: Any,
    *,
    fsdp2: bool,
) -> tuple[Any, str]:
    path = Path(value)
    if path.is_dir() and (path / "manifest.json").is_file():
        manager = (
            DCPCheckpointManager(path.parent, torch)
            if fsdp2
            else LocalCheckpointManager(path.parent)
        )
        return manager, path.name
    return default_manager, value


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TorchEngineError(
            "PyTorch is not installed; install trainomni-framework[torch]"
        ) from exc
    return torch


def select_torch_device(torch: Any, config: Mapping[str, Any]) -> Any:
    requested = config.get("device", "auto")
    allowed = {"auto", "cpu", "cuda"}
    if requested not in allowed:
        raise TorchEngineError(f"unsupported torch device {requested!r}")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise TorchEngineError("CUDA device requested but unavailable")
    if requested == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def configure_torch_precision(torch: Any, precision: str, device: Any) -> None:
    if precision not in {"fp32", "tf32", "fp16", "bf16"}:
        raise TorchEngineError(
            f"unsupported native torch precision {precision!r}"
        )
    if precision == "tf32" and device.type == "cuda":
        matmul = torch.backends.cuda.matmul
        if hasattr(matmul, "fp32_precision"):
            matmul.fp32_precision = "tf32"
        else:  # pragma: no cover - older torch compatibility
            matmul.allow_tf32 = True


def _wrap_parallel(torch: Any, model: Any, parallelism: str, device: Any) -> Any:
    if parallelism == "single":
        return model
    if parallelism == "ddp":
        if not torch.distributed.is_available():
            raise TorchEngineError("torch.distributed is unavailable")
        if not torch.distributed.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            init_method = os.environ.get("TRAINOMNI_DIST_INIT_METHOD", "env://")
            rendezvous = {}
            if init_method != "env://":
                rendezvous = {
                    "rank": int(os.environ["RANK"]),
                    "world_size": int(os.environ["WORLD_SIZE"]),
                }
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                **rendezvous,
            )
        device_ids = [device.index] if device.type == "cuda" else None
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
    if parallelism == "fsdp2":
        fully_shard = _fully_shard()
        if not torch.distributed.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            init_method = os.environ.get("TRAINOMNI_DIST_INIT_METHOD", "env://")
            rendezvous = {}
            if init_method != "env://":
                rendezvous = {
                    "rank": int(os.environ["RANK"]),
                    "world_size": int(os.environ["WORLD_SIZE"]),
                }
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                **rendezvous,
            )
        fully_shard(model)
        return model
    raise TorchEngineError(f"unsupported parallelism {parallelism!r}")


def _has_fsdp2(torch: Any) -> bool:
    try:
        _fully_shard()
    except TorchEngineError:
        return False
    return True


def _fully_shard() -> Any:
    try:
        from torch.distributed.fsdp import fully_shard

        return fully_shard
    except ImportError:
        try:
            from torch.distributed._composable.fsdp import fully_shard

            return fully_shard
        except ImportError as exc:  # pragma: no cover - version-specific
            raise TorchEngineError("FSDP2 fully_shard is unavailable") from exc


def _component_assignments(context: TorchStageContext, model: Any) -> Mapping[str, tuple[str, ...]]:
    catalog = context.plugin.component_catalog(context.bundle)
    assignments, issues = catalog.classify(name for name, _ in model.named_parameters())
    if issues:
        details = "; ".join(f"{item.code}: {item.message}" for item in issues)
        raise TorchEngineError(f"component catalog is not exhaustive: {details}")
    return assignments


def _apply_component_policy(
    torch: Any,
    context: TorchStageContext,
    model: Any,
    assignments: Mapping[str, tuple[str, ...]],
) -> None:
    stage = context.resolved.run.stage
    by_name = dict(model.named_parameters())
    unknown_policy = set(stage.component_policy) - set(assignments)
    if unknown_policy:
        raise TorchEngineError(f"unknown component policy: {sorted(unknown_policy)}")
    for component, names in assignments.items():
        policy = stage.component_policy.get(component)
        trainable = policy.trainable if policy is not None else True
        for name in names:
            by_name[name].requires_grad_(trainable)
            if policy is not None and policy.dtype is not None:
                dtype_name = {
                    "fp32": "float32",
                    "fp16": "float16",
                    "bf16": "bfloat16",
                }.get(policy.dtype, policy.dtype)
                dtype = getattr(torch, dtype_name, None)
                if dtype is None:
                    raise TorchEngineError(
                        f"unknown dtype {policy.dtype!r} for component {component!r}"
                    )
                by_name[name].data = by_name[name].data.to(dtype=dtype)


def _configure_activation_checkpointing(
    context: TorchStageContext,
) -> Mapping[str, Mapping[str, Any]]:
    requests = {
        component_id: ActivationCheckpointingRequest(
            component_id=component_id,
            use_reentrant=configuration.use_reentrant,
            config=configuration.config,
        )
        for component_id, policy in context.resolved.run.stage.component_policy.items()
        if (configuration := policy.activation_checkpointing) is not None
    }
    if not requests:
        return {}
    configure = getattr(
        context.plugin, "configure_activation_checkpointing", None
    )
    if not callable(configure):
        raise TorchEngineError(
            "per-component activation checkpointing requires model plugin method "
            "configure_activation_checkpointing(bundle, requests); the core does not "
            "fall back to a top-level gradient_checkpointing_enable() hook"
        )
    try:
        receipts = configure(context.bundle, requests)
    except Exception as exc:
        raise TorchEngineError(
            "model plugin activation-checkpoint configuration failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(receipts, Mapping):
        raise TorchEngineError(
            "configure_activation_checkpointing() must return a receipt mapping"
        )
    missing = set(requests) - set(receipts)
    unexpected = set(receipts) - set(requests)
    if missing or unexpected:
        raise TorchEngineError(
            "activation-checkpoint receipts must exactly cover requests: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for component_id in sorted(requests):
        request = requests[component_id]
        receipt = receipts[component_id]
        if not isinstance(receipt, ActivationCheckpointingReceipt):
            raise TorchEngineError(
                f"activation-checkpoint receipt for {component_id!r} has invalid type"
            )
        if (
            receipt.component_id != component_id
            or not receipt.enabled
            or receipt.use_reentrant != request.use_reentrant
        ):
            raise TorchEngineError(
                f"activation-checkpoint receipt for {component_id!r} does not match "
                "the requested enabled/use_reentrant contract"
            )
        result[component_id] = {
            "enabled": True,
            "implementation": receipt.implementation,
            "use_reentrant": receipt.use_reentrant,
            "config": dict(request.config),
            "metadata": dict(receipt.metadata),
        }
    return result


def _build_optimizer(
    torch: Any,
    context: TorchStageContext,
    model: Any,
    assignments: Mapping[str, tuple[str, ...]],
    device: Any,
) -> Any:
    stage = context.resolved.run.stage
    by_name = dict(model.named_parameters())
    classified = _classify_runtime_parameters(by_name, assignments)
    groups = []
    for component, parameters in classified.items():
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        if not parameters:
            continue
        policy = stage.component_policy.get(component)
        groups.append(
            {
                "params": parameters,
                "lr": (
                    policy.learning_rate
                    if policy and policy.learning_rate is not None
                    else stage.optimization.learning_rate
                ),
                "weight_decay": (
                    policy.weight_decay
                    if policy and policy.weight_decay is not None
                    else stage.optimization.weight_decay
                ),
                "component_id": component,
            }
        )
    if not groups:
        raise TorchEngineError("no trainable parameters remain after component policy")
    name = stage.optimization.optimizer.lower()
    implementation, config, quantization = _resolved_optimizer_config(stage)
    if implementation == "bitsandbytes":
        if device.type != "cuda":
            raise TorchEngineError(
                "bitsandbytes AdamW8bit requires a CUDA device; no fallback to "
                "torch AdamW was performed"
            )
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise TorchEngineError(
                "bitsandbytes optimizer was explicitly requested but the package is "
                "not installed; install trainomni-framework[bitsandbytes]. No "
                "fallback to torch AdamW was performed"
            ) from exc
        assert quantization is not None
        optimizer_class = (
            bnb.optim.PagedAdamW8bit
            if quantization["paged"]
            else bnb.optim.AdamW8bit
        )
        bnb_config = {
            **config,
            "min_8bit_size": quantization["min_8bit_size"],
            "percentile_clipping": quantization["percentile_clipping"],
            "block_wise": quantization["block_wise"],
        }
        try:
            return optimizer_class(groups, **bnb_config)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TorchEngineError(
                f"cannot construct explicitly requested bitsandbytes optimizer: {exc}"
            ) from exc
    optimizer_classes = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    optimizer_class = optimizer_classes.get(name)
    if optimizer_class is None:
        raise TorchEngineError(
            f"unsupported optimizer {stage.optimization.optimizer!r}"
        )
    try:
        return optimizer_class(groups, **config)
    except (TypeError, ValueError) as exc:
        raise TorchEngineError(
            f"invalid torch {name} optimizer configuration: {exc}"
        ) from exc


def _resolved_optimizer_config(
    stage: Any,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    specification = stage.optimization.optimizer_config
    legacy = stage.optimization.config.get("optimizer")
    if legacy is not None:
        return "torch", dict(legacy), None
    config = dict(specification.kwargs)
    if specification.foreach is not None:
        config["foreach"] = specification.foreach
    quantization = (
        specification.quantization.model_dump(mode="json")
        if specification.quantization is not None
        else None
    )
    return specification.implementation, config, quantization


def _optimizer_trainable_numel(optimizer: Any) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for group in optimizer.param_groups:
        component_id = str(group.get("component_id", "__unclassified__"))
        counts[component_id] = counts.get(component_id, 0) + sum(
            int(parameter.numel())
            for parameter in group["params"]
            if parameter.requires_grad
        )
    return {key: counts[key] for key in sorted(counts)}


def _validate_diagnostics_contract(state: _TorchPreparedState) -> None:
    diagnostics = state.context.resolved.run.stage.optimization.diagnostics
    total = sum(state.trainable_numel_by_component.values())
    if (
        diagnostics.expected_trainable_numel is not None
        and total != diagnostics.expected_trainable_numel
    ):
        raise TorchEngineError(
            "trainable_numel contract mismatch: "
            f"expected {diagnostics.expected_trainable_numel}, observed {total}"
        )
    required = set(diagnostics.required_components)
    missing = required - set(state.trainable_numel_by_component)
    empty = {
        component
        for component in required
        if state.trainable_numel_by_component.get(component, 0) <= 0
    }
    if missing or empty:
        raise TorchEngineError(
            "required diagnostic components are absent from the optimizer: "
            f"missing={sorted(missing)}, empty={sorted(empty)}"
        )
    if diagnostics.max_reserved_bytes is not None and state.device.type != "cuda":
        raise TorchEngineError(
            "diagnostics.max_reserved_bytes requires a CUDA execution device"
        )


def _reset_peak_memory(state: _TorchPreparedState) -> None:
    diagnostics = state.context.resolved.run.stage.optimization.diagnostics
    if diagnostics.record_gpu_memory and state.device.type == "cuda":
        state.torch.cuda.reset_peak_memory_stats(state.device)


def _capture_step_evidence(
    state: _TorchPreparedState,
) -> tuple[dict[str, Any], tuple[_UpdateProbe, ...]]:
    diagnostics = state.context.resolved.run.stage.optimization.diagnostics
    evidence: dict[str, Any] = {
        "schema_version": "trainomni.training-evidence.v2",
        "step": int(state.step.value) + 1,
        "trainable_numel": sum(state.trainable_numel_by_component.values()),
        "components": {
            component_id: {"trainable_numel": count}
            for component_id, count in state.trainable_numel_by_component.items()
        },
    }
    if not (
        diagnostics.component_grad_norms
        or diagnostics.component_update_probes
    ):
        return evidence, ()

    parameter_names = {
        id(parameter): name for name, parameter in state.model.named_parameters()
    }
    parameters_by_component: dict[str, list[Any]] = {}
    for group in state.optimizer.param_groups:
        component_id = str(group.get("component_id", "__unclassified__"))
        parameters_by_component.setdefault(component_id, []).extend(
            parameter
            for parameter in group["params"]
            if parameter.requires_grad
        )
    probes: list[_UpdateProbe] = []
    targets = set(diagnostics.required_components) or set(parameters_by_component)
    for component_id in sorted(parameters_by_component):
        parameters = parameters_by_component[component_id]
        with_grad = [
            (parameter, _dense_gradient(parameter.grad))
            for parameter in parameters
            if parameter.grad is not None
        ]
        norms = [
            state.torch.linalg.vector_norm(
                gradient, ord=2, dtype=state.torch.float32
            )
            for _, gradient in with_grad
        ]
        if norms:
            norm_stack = state.torch.stack(norms)
            grad_norm = float(
                state.torch.linalg.vector_norm(
                    norm_stack, ord=2, dtype=state.torch.float32
                ).item()
            )
        else:
            norm_stack = None
            grad_norm = 0.0
        finite = math.isfinite(grad_norm)
        component = evidence["components"][component_id]
        component.update(
            {
                "grad_norm": grad_norm,
                "grad_finite": finite,
                "gradient_tensors": len(with_grad),
            }
        )
        if (
            diagnostics.require_finite_nonzero_gradients
            and component_id in targets
            and (not finite or grad_norm <= 0)
        ):
            raise TorchEngineError(
                f"component {component_id!r} has non-finite or zero gradient norm"
            )
        if diagnostics.component_update_probes:
            named_parameters = sorted(
                (
                    parameter_names.get(
                        id(parameter),
                        f"<{component_id}-optimizer-parameter-{index}>",
                    ),
                    parameter,
                )
                for index, parameter in enumerate(parameters)
            )
            component_probes: list[_UpdateProbe] = []
            for parameter_name, parameter in named_parameters:
                local_parameter = _local_parameter_tensor(parameter)
                if local_parameter.numel() == 0:
                    continue
                before = local_parameter.to(
                    device="cpu", copy=True
                ).contiguous()
                component_probes.append(
                    _UpdateProbe(
                        component_id=component_id,
                        parameter_name=parameter_name,
                        parameter=parameter,
                        before=before,
                    )
                )
            probes.extend(component_probes)
            probed_elements = sum(
                int(probe.before.numel()) for probe in component_probes
            )
            update_probe: dict[str, Any] = {
                "schema_version": "trainomni.component-update-evidence.v2",
                "method": "exact_full_parameter_bitwise_scan",
                "exact": True,
                "selection": (
                    "all optimizer-held component parameters in canonical "
                    "parameter-name order and ascending flat-index order"
                ),
                "count_semantics": (
                    "exact bitwise-changed optimizer elements; distributed "
                    "counts sum local shards or replicas"
                ),
                "chunk_elements": diagnostics.update_probe_chunk_elements,
                "snapshot_tensors": len(component_probes),
                "local_probed_elements": probed_elements,
            }
            if component_probes:
                representative = component_probes[0]
                before_value = _scalar_value(
                    representative.before.reshape(-1)[0]
                )
                update_probe.update(
                    {
                        "parameter": representative.parameter_name,
                        "flat_index": 0,
                        "before": before_value,
                        "representative": {
                            "parameter": representative.parameter_name,
                            "flat_index": 0,
                            "before": before_value,
                            "dtype": str(representative.before.dtype),
                            "shape": list(representative.before.shape),
                        },
                    }
                )
            component["update_probe"] = update_probe
    return evidence, tuple(probes)


def _dense_gradient(gradient: Any) -> Any:
    detached = gradient.detach()
    if getattr(detached, "is_sparse", False):
        return detached.coalesce().values()
    return detached


def _local_parameter_tensor(parameter: Any) -> Any:
    detached = parameter.detach()
    to_local = getattr(detached, "to_local", None)
    if callable(to_local):
        detached = to_local()
    return detached


def _scalar_value(value: Any) -> Any:
    scalar = value.item()
    if isinstance(scalar, complex):
        return {"real": float(scalar.real), "imag": float(scalar.imag)}
    if isinstance(scalar, (bool, int, float)):
        return scalar
    return str(scalar)


def _bitwise_changed_mask(torch: Any, before: Any, after: Any) -> Any:
    if before.dtype != after.dtype:
        raise TorchEngineError(
            "parameter dtype changed during optimizer step: "
            f"before={before.dtype}, after={after.dtype}"
        )
    element_size = before.element_size()
    before_bytes = before.contiguous().view(torch.uint8).reshape(-1, element_size)
    after_bytes = after.contiguous().view(torch.uint8).reshape(-1, element_size)
    return torch.any(before_bytes != after_bytes, dim=1)


def _absolute_delta(torch: Any, before: Any, after: Any) -> Any:
    dtype = torch.complex128 if before.is_complex() else torch.float64
    return (after.to(dtype=dtype) - before.to(dtype=dtype)).abs()


def _distributed_update_totals(
    state: _TorchPreparedState,
    *,
    probed_elements: int,
    changed_elements: int,
    snapshot_tensors: int,
    changed_tensors: int,
    abs_update_l1: float,
    max_abs_update: float,
) -> dict[str, Any]:
    torch = state.torch
    distributed = getattr(torch, "distributed", None)
    initialized = bool(
        distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
    )
    if not initialized:
        return {
            "world_size": 1,
            "probed_elements": probed_elements,
            "changed_elements": changed_elements,
            "snapshot_tensors": snapshot_tensors,
            "changed_tensors": changed_tensors,
            "abs_update_l1": abs_update_l1,
            "max_abs_update": max_abs_update,
        }
    counts = torch.tensor(
        [
            probed_elements,
            changed_elements,
            snapshot_tensors,
            changed_tensors,
        ],
        dtype=torch.int64,
        device=state.device,
    )
    distributed.all_reduce(counts, op=distributed.ReduceOp.SUM)
    l1 = torch.tensor(abs_update_l1, dtype=torch.float64, device=state.device)
    maximum = torch.tensor(
        max_abs_update, dtype=torch.float64, device=state.device
    )
    distributed.all_reduce(l1, op=distributed.ReduceOp.SUM)
    distributed.all_reduce(maximum, op=distributed.ReduceOp.MAX)
    return {
        "world_size": int(distributed.get_world_size()),
        "probed_elements": int(counts[0].item()),
        "changed_elements": int(counts[1].item()),
        "snapshot_tensors": int(counts[2].item()),
        "changed_tensors": int(counts[3].item()),
        "abs_update_l1": float(l1.item()),
        "max_abs_update": float(maximum.item()),
    }


def _finish_step_evidence(
    state: _TorchPreparedState,
    evidence: dict[str, Any],
    probes: tuple[_UpdateProbe, ...],
) -> dict[str, Any]:
    diagnostics = state.context.resolved.run.stage.optimization.diagnostics
    components = evidence["components"]
    probes_by_component: dict[str, list[_UpdateProbe]] = {}
    for probe in probes:
        probes_by_component.setdefault(probe.component_id, []).append(probe)
    for component_id in sorted(components):
        update = components[component_id].get("update_probe")
        if update is None:
            continue
        local_probed_elements = 0
        local_changed_elements = 0
        local_changed_tensors = 0
        local_abs_update_l1 = 0.0
        local_max_abs_update = 0.0
        first_changed: dict[str, Any] | None = None
        maximum_changed: dict[str, Any] | None = None
        representative_after: Any | None = None
        representative_bitwise_changed: bool | None = None
        component_probes = probes_by_component.get(component_id, [])
        for probe_index, probe in enumerate(component_probes):
            before_flat = probe.before.reshape(-1)
            after_local = _local_parameter_tensor(probe.parameter)
            if tuple(after_local.shape) != tuple(probe.before.shape):
                raise TorchEngineError(
                    "parameter shape changed during optimizer update scan: "
                    f"component={component_id!r}, parameter={probe.parameter_name!r}, "
                    f"before_shape={tuple(probe.before.shape)}, "
                    f"after_shape={tuple(after_local.shape)}"
                )
            after_flat = after_local.reshape(-1)
            tensor_changed = 0
            for start in range(
                0,
                int(before_flat.numel()),
                diagnostics.update_probe_chunk_elements,
            ):
                stop = min(
                    start + diagnostics.update_probe_chunk_elements,
                    int(before_flat.numel()),
                )
                before_chunk = before_flat[start:stop]
                after_chunk = after_flat[start:stop].to(
                    device="cpu", copy=True
                ).contiguous()
                if probe_index == 0 and start == 0:
                    representative_after = _scalar_value(after_chunk[0])
                changed_mask = _bitwise_changed_mask(
                    state.torch, before_chunk, after_chunk
                )
                if probe_index == 0 and start == 0:
                    representative_bitwise_changed = bool(
                        changed_mask[0].item()
                    )
                changed_count = int(changed_mask.sum().item())
                local_probed_elements += stop - start
                if changed_count == 0:
                    continue
                tensor_changed += changed_count
                local_changed_elements += changed_count
                changed_indices = state.torch.nonzero(
                    changed_mask, as_tuple=False
                ).reshape(-1)
                delta = _absolute_delta(
                    state.torch, before_chunk, after_chunk
                )
                changed_delta = delta[changed_mask]
                if not bool(state.torch.isfinite(changed_delta).all().item()):
                    raise TorchEngineError(
                        "component update scan found a non-finite parameter delta: "
                        f"component={component_id!r}, "
                        f"parameter={probe.parameter_name!r}"
                    )
                local_abs_update_l1 += float(
                    changed_delta.sum(dtype=state.torch.float64).item()
                )
                first_in_chunk = int(changed_indices[0].item())
                if first_changed is None:
                    first_changed = {
                        "parameter": probe.parameter_name,
                        "flat_index": start + first_in_chunk,
                        "before": _scalar_value(before_chunk[first_in_chunk]),
                        "after": _scalar_value(after_chunk[first_in_chunk]),
                        "abs_update": float(delta[first_in_chunk].item()),
                    }
                maximum_changed_offset = int(
                    state.torch.argmax(changed_delta).item()
                )
                maximum_in_chunk = int(
                    changed_indices[maximum_changed_offset].item()
                )
                maximum_value = float(delta[maximum_in_chunk].item())
                if maximum_changed is None or maximum_value > local_max_abs_update:
                    local_max_abs_update = maximum_value
                    maximum_changed = {
                        "parameter": probe.parameter_name,
                        "flat_index": start + maximum_in_chunk,
                        "before": _scalar_value(before_chunk[maximum_in_chunk]),
                        "after": _scalar_value(after_chunk[maximum_in_chunk]),
                        "abs_update": maximum_value,
                    }
            if tensor_changed > 0:
                local_changed_tensors += 1
        totals = _distributed_update_totals(
            state,
            probed_elements=local_probed_elements,
            changed_elements=local_changed_elements,
            snapshot_tensors=len(component_probes),
            changed_tensors=local_changed_tensors,
            abs_update_l1=local_abs_update_l1,
            max_abs_update=local_max_abs_update,
        )
        if representative_after is not None:
            update["after"] = representative_after
            update["representative"]["after"] = representative_after
            update["representative"]["bitwise_changed"] = (
                representative_bitwise_changed
            )
        update.update(
            {
                "world_size": totals["world_size"],
                "local_probed_elements": local_probed_elements,
                "local_changed_elements": local_changed_elements,
                "local_snapshot_tensors": len(component_probes),
                "local_changed_tensors": local_changed_tensors,
                "local_abs_update_l1": local_abs_update_l1,
                "local_max_abs_update": local_max_abs_update,
                "probed_elements": totals["probed_elements"],
                "changed_elements": totals["changed_elements"],
                "snapshot_tensors": totals["snapshot_tensors"],
                "changed_tensors": totals["changed_tensors"],
                "abs_update": totals["max_abs_update"],
                "max_abs_update": totals["max_abs_update"],
                "abs_update_l1": totals["abs_update_l1"],
                "first_changed_element": first_changed,
                "max_abs_update_element": maximum_changed,
            }
        )
    targets = set(diagnostics.required_components) or set(components)
    if diagnostics.require_parameter_updates:
        unchanged: dict[str, dict[str, Any]] = {}
        for component_id in sorted(targets):
            update = components.get(component_id, {}).get("update_probe", {})
            if (
                update.get("changed_elements", 0) <= 0
                or update.get("max_abs_update", 0.0) <= 0.0
            ):
                unchanged[component_id] = {
                    "probed_elements": update.get("probed_elements", 0),
                    "changed_elements": update.get("changed_elements", 0),
                    "max_abs_update": update.get("max_abs_update", 0.0),
                    "method": update.get("method"),
                }
        if unchanged:
            raise TorchEngineError(
                "required components have no exact numerical parameter-update "
                f"evidence after optimizer step: {unchanged}"
            )
    memory = _gpu_memory_evidence(state)
    evidence["gpu_memory"] = memory
    if (
        diagnostics.max_reserved_bytes is not None
        and memory["max_reserved_bytes"] > diagnostics.max_reserved_bytes
    ):
        raise TorchEngineError(
            "GPU peak reserved memory exceeded configured safety limit: "
            f"observed={memory['max_reserved_bytes']}, "
            f"limit={diagnostics.max_reserved_bytes}"
        )
    return evidence


def _gpu_memory_evidence(state: _TorchPreparedState) -> dict[str, Any]:
    diagnostics = state.context.resolved.run.stage.optimization.diagnostics
    if not diagnostics.record_gpu_memory or state.device.type != "cuda":
        return {
            "available": False,
            "device": str(state.device),
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
        }
    cuda = state.torch.cuda
    return {
        "available": True,
        "device": str(state.device),
        "allocated_bytes": int(cuda.memory_allocated(state.device)),
        "reserved_bytes": int(cuda.memory_reserved(state.device)),
        "max_allocated_bytes": int(cuda.max_memory_allocated(state.device)),
        "max_reserved_bytes": int(cuda.max_memory_reserved(state.device)),
    }


def _evidence_metrics(evidence: Mapping[str, Any]) -> dict[str, float]:
    metrics = {"trainable_numel": float(evidence["trainable_numel"])}
    for component_id, component in evidence["components"].items():
        prefix = f"components/{component_id}"
        for name in ("trainable_numel", "grad_norm", "gradient_tensors"):
            if name in component:
                metrics[f"{prefix}/{name}"] = float(component[name])
        update = component.get("update_probe", {})
        for name in (
            "probed_elements",
            "changed_elements",
            "snapshot_tensors",
            "changed_tensors",
            "abs_update",
            "max_abs_update",
            "abs_update_l1",
            "local_probed_elements",
            "local_changed_elements",
            "local_snapshot_tensors",
            "local_changed_tensors",
            "local_max_abs_update",
            "local_abs_update_l1",
        ):
            if name in update:
                metrics[f"{prefix}/{name}"] = float(update[name])
    memory = evidence["gpu_memory"]
    for name in (
        "allocated_bytes",
        "reserved_bytes",
        "max_allocated_bytes",
        "max_reserved_bytes",
    ):
        metrics[f"gpu_memory/{name}"] = float(memory[name])
    return metrics


def _training_metadata(state: _TorchPreparedState) -> dict[str, Any]:
    metadata = {
        "optimizer": _optimizer_metadata(state),
        "activation_checkpointing": {
            key: dict(value)
            for key, value in state.activation_checkpointing.items()
        },
        "trainable_numel": sum(state.trainable_numel_by_component.values()),
        "trainable_numel_by_component": dict(
            state.trainable_numel_by_component
        ),
        "training_evidence": dict(state.latest_evidence),
    }
    objective = _objective_metadata(state)
    if objective is not None:
        metadata["objective"] = objective
    if state.latest_objective_metrics:
        metadata["objective_evidence"] = dict(state.latest_objective_metrics)
    if state.objective_counts:
        metadata["objective_counts"] = {
            key: counter.value for key, counter in state.objective_counts.items()
        }
    return metadata


def _update_objective_counts(
    state: _TorchPreparedState, counts: Mapping[str, int]
) -> None:
    for key, counter in state.objective_counts.items():
        value = counts.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TorchEngineError(
                f"objective count {key!r} must be a non-negative integer"
            )
        counter.value += value


def _setup_objective(state: _TorchPreparedState) -> ObjectiveSetup | None:
    objective = state.context.objective.objective
    setup = getattr(objective, "setup", None)
    if not callable(setup):
        return None
    try:
        result = setup(state)
    except Exception as exc:
        raise TorchEngineError(f"objective preflight failed: {exc}") from exc
    if not isinstance(result, ObjectiveSetup):
        raise TorchEngineError(
            "objective setup() must return ObjectiveSetup"
        )
    return result


def _objective_metadata(state: _TorchPreparedState) -> dict[str, Any] | None:
    setup = state.objective_setup
    if setup is None:
        return None
    manifest = state.context.objective.objective.manifest
    metadata = {
        "objective_id": manifest.objective_id,
        "objective_version": manifest.objective_version,
        "identity": _metadata_value(setup.metadata),
        "exact_resume": "immutable_external_identity",
    }
    if setup.state_count_keys:
        metadata["state_count_keys"] = list(setup.state_count_keys)
    return metadata


def _optimizer_metadata(state: _TorchPreparedState) -> dict[str, Any]:
    stage = state.context.resolved.run.stage
    implementation, configured_kwargs, quantization = _resolved_optimizer_config(
        stage
    )
    if implementation == "bitsandbytes":
        package_name = "bitsandbytes"
        try:
            package_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - build guards
            package_version = None
    else:
        package_name = "torch"
        package_version = str(state.torch.__version__)
    optimizer_type = type(state.optimizer)
    return {
        "name": stage.optimization.optimizer.lower(),
        "implementation": implementation,
        "class": f"{optimizer_type.__module__}.{optimizer_type.__qualname__}",
        "package": {"name": package_name, "version": package_version},
        "configured_kwargs": _metadata_value(configured_kwargs),
        "actual_defaults": _metadata_value(state.optimizer.defaults),
        "quantization": _metadata_value(quantization),
        "state": _optimizer_state_summary(state.optimizer),
        "exact_resume": "full_state_dict",
    }


def _optimizer_state_summary(optimizer: Any) -> dict[str, Any]:
    tensor_dtypes: dict[str, dict[str, int]] = {}
    non_tensor_types: dict[str, int] = {}

    def visit(value: Any) -> None:
        if hasattr(value, "dtype") and hasattr(value, "numel"):
            name = str(value.dtype)
            item = tensor_dtypes.setdefault(name, {"tensors": 0, "numel": 0})
            item["tensors"] += 1
            item["numel"] += int(value.numel())
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            name = type(value).__name__
            non_tensor_types[name] = non_tensor_types.get(name, 0) + 1

    for item in optimizer.state.values():
        visit(item)
    return {
        "entries": len(optimizer.state),
        "state_dtypes": sorted(tensor_dtypes),
        "tensor_dtypes": {
            key: tensor_dtypes[key] for key in sorted(tensor_dtypes)
        },
        "non_tensor_types": {
            key: non_tensor_types[key] for key in sorted(non_tensor_types)
        },
    }


def _metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    return {"type": type(value).__name__}


def _build_scheduler(torch: Any, optimizer: Any, resolved: ResolvedRunSpec) -> Any | None:
    config = resolved.run.stage.optimization.config
    name = config.get("scheduler", "constant")
    if name == "constant":
        return None
    max_steps = resolved.run.stage.optimization.max_steps
    if max_steps is None:
        raise TorchEngineError("non-constant scheduler requires max_steps")
    warmup = int(config.get("warmup_steps", 0))
    if not 0 <= warmup < max_steps:
        raise TorchEngineError("warmup_steps must be in [0, max_steps)")

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        if name == "linear":
            return max(0.0, 1.0 - progress)
        if name == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        raise TorchEngineError(f"unsupported scheduler {name!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _build_scaler(torch: Any, precision: str, device: Any) -> Any | None:
    if precision != "fp16" or device.type != "cuda":
        return None
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        return amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def torch_autocast_context(torch: Any, precision: str, device: Any) -> Any:
    configure_torch_precision(torch, precision, device)
    if precision in {"fp32", "tf32"} or device.type == "cpu" and precision == "fp16":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def move_model_batch(batch: ModelBatch, device: Any) -> ModelBatch:
    def move(value: Any) -> Any:
        if hasattr(value, "to") and callable(value.to):
            return value.to(device, non_blocking=True)
        if isinstance(value, Mapping):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    return ModelBatch(
        sample_ids=batch.sample_ids,
        model_inputs=move(dict(batch.model_inputs)),
        plan=batch.plan,
        trace=batch.trace,
    )


def prepare_models_for_evaluation(bundle: ModelBundle, device: Any) -> None:
    """Move every objective-visible model to one device and enter eval mode."""

    seen: set[int] = set()
    for name, model in bundle.models().items():
        if id(model) in seen:
            continue
        seen.add(id(model))
        move = getattr(model, "to", None)
        if not callable(move):
            raise TorchEngineError(
                f"torch evaluation model {name!r} does not expose to(device)"
            )
        move(device)
        evaluate = getattr(model, "eval", None)
        if not callable(evaluate):
            raise TorchEngineError(
                f"torch evaluation model {name!r} does not expose eval()"
            )
        evaluate()


def _clip_gradients(
    torch: Any,
    context: TorchStageContext,
    model: Any,
    original: Mapping[str, tuple[str, ...]],
) -> None:
    stage = context.resolved.run.stage
    by_name = dict(model.named_parameters())
    assignments = _classify_runtime_parameters(by_name, original)
    for component, component_parameters in assignments.items():
        policy = stage.component_policy.get(component)
        if policy is None or policy.gradient_clip is None:
            continue
        parameters = [
            parameter
            for parameter in component_parameters
            if parameter.requires_grad and parameter.grad is not None
        ]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, policy.gradient_clip)


def _classify_runtime_parameters(
    parameters: Mapping[str, Any],
    original_assignments: Mapping[str, tuple[str, ...]],
) -> Mapping[str, list[Any]]:
    result: dict[str, list[Any]] = {
        component: [] for component in original_assignments
    }
    result["__adapter__"] = []
    exact = {
        name: component
        for component, names in original_assignments.items()
        for name in names
    }
    module_paths = {
        component: tuple(
            sorted(
                {name.rsplit(".", 1)[0] for name in names if "." in name},
                key=len,
                reverse=True,
            )
        )
        for component, names in original_assignments.items()
    }
    for name, parameter in parameters.items():
        component = exact.get(name)
        if component is None:
            matches = [
                item
                for item, paths in module_paths.items()
                if any(path in name for path in paths)
            ]
            component = matches[0] if len(matches) == 1 else "__adapter__"
        result[component].append(parameter)
    return result
