"""Optional native PyTorch training loop.

The module imports PyTorch lazily so validation, data inspection, pipeline
planning, and non-Torch adapters remain usable in a lightweight environment.
"""

from __future__ import annotations

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
from trainomni.models import ModelBatch, ModelBundle
from trainomni.objectives import ObjectiveBinding

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
    step: ScalarState = field(default_factory=ScalarState)
    microstep: ScalarState = field(default_factory=ScalarState)
    tokens: ScalarState = field(default_factory=ScalarState)
    last_checkpoint: ArtifactRef | None = None


class TorchEngine:
    manifest = EngineManifest(
        engine_id="torch",
        engine_version="1.0.0",
        kind=EngineKind.LOOP,
        capabilities=EngineCapabilities(
            stage_types=_TRAINING_STAGES,
            objectives=frozenset({"masked-causal-lm"}),
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
            torch = _import_torch()
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
        torch = _import_torch()
        report = self.validate(context.resolved.run.stage, context.bundle)
        if not report.valid:
            details = "; ".join(item.message for item in report.issues)
            raise TorchEngineError(details)
        stage = context.resolved.run.stage
        device = _select_device(torch, stage.engine.config)
        original_model = context.bundle.model
        assignments = _component_assignments(context, original_model)
        _apply_component_policy(torch, context, original_model, assignments)
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
        if stage.engine.precision == "tf32" and device.type == "cuda":
            matmul = torch.backends.cuda.matmul
            if hasattr(matmul, "fp32_precision"):
                matmul.fp32_precision = "tf32"
            else:  # pragma: no cover - older torch compatibility
                matmul.allow_tf32 = True
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
            torch, context, train_model, assignments
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
        )
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
            batch = _move_batch(batch, state.device)
            prepared_batch = state.context.objective.objective.prepare(batch, state)
            autocast = _autocast_context(
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
            latest_metrics = {
                key: float(value) for key, value in loss_output.metrics.items()
            }
            if not sync_now:
                continue
            if state.scaler is not None:
                state.scaler.unscale_(state.optimizer)
            _clip_gradients(
                torch, state.context, state.model, state.component_assignments
            )
            if state.scaler is not None:
                state.scaler.step(state.optimizer)
                state.scaler.update()
            else:
                state.optimizer.step()
            state.optimizer.zero_grad(set_to_none=True)
            if state.scheduler is not None:
                state.scheduler.step()
            state.step.value += 1
            latest_metrics["step"] = float(state.step.value)
            latest_metrics["loss_tokens"] = float(state.tokens.value)
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


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TorchEngineError(
            "PyTorch is not installed; install trainomni-framework[torch]"
        ) from exc
    return torch


def _select_device(torch: Any, config: Mapping[str, Any]) -> Any:
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
    if any(
        policy.activation_checkpointing is True
        for policy in stage.component_policy.values()
    ):
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise TorchEngineError(
                "activation checkpointing was requested but model does not expose "
                "gradient_checkpointing_enable()"
            )
        enable()


def _build_optimizer(
    torch: Any,
    context: TorchStageContext,
    model: Any,
    assignments: Mapping[str, tuple[str, ...]],
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
    config = dict(stage.optimization.config.get("optimizer", {}))
    if name == "adamw":
        return torch.optim.AdamW(groups, **config)
    if name == "adam":
        return torch.optim.Adam(groups, **config)
    if name == "sgd":
        return torch.optim.SGD(groups, **config)
    raise TorchEngineError(f"unsupported optimizer {stage.optimization.optimizer!r}")


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


def _autocast_context(torch: Any, precision: str, device: Any) -> Any:
    if precision in {"fp32", "tf32"} or device.type == "cpu" and precision == "fp16":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _move_batch(batch: ModelBatch, device: Any) -> ModelBatch:
    def move(value: Any) -> Any:
        if hasattr(value, "to") and callable(value.to):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
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
