"""Execution run specification: how one attempt executes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from trainomni.core.errors import SpecError

from .digest import identity_digest


def _strict_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpecError(f"{field} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class OptimizerGroupOverride:
    name: str
    learning_rate: float | None = None
    weight_decay: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("optimizer group name must not be empty")
        if self.learning_rate is not None and (
            not math.isfinite(self.learning_rate) or self.learning_rate <= 0
        ):
            raise ValueError("optimizer group learning_rate must be positive")
        if self.weight_decay is not None and (
            not math.isfinite(self.weight_decay) or self.weight_decay < 0
        ):
            raise ValueError("optimizer group weight_decay must be non-negative")


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    foreach: bool | None = None
    group_overrides: tuple[OptimizerGroupOverride, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OptimizerSpec:
        allowed = {
            "name",
            "learning_rate",
            "weight_decay",
            "betas",
            "eps",
            "foreach",
            "groups",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"optimizer contains unknown keys: {', '.join(unknown)}")
        name = str(value.get("name", "adamw"))
        learning_rate = float(value.get("learning_rate", 1e-3))
        weight_decay = float(value.get("weight_decay", 0.0))
        raw_betas = value.get("betas", (0.9, 0.999))
        if not isinstance(raw_betas, (list, tuple)) or len(raw_betas) != 2:
            raise SpecError("optimizer.betas must contain exactly two values")
        betas = (float(raw_betas[0]), float(raw_betas[1]))
        eps = float(value.get("eps", 1e-8))
        raw_foreach = value.get("foreach")
        if raw_foreach is not None and not isinstance(raw_foreach, bool):
            raise SpecError("optimizer.foreach must be true, false, or null")
        raw_groups = value.get("groups", {})
        if not isinstance(raw_groups, Mapping):
            raise SpecError("optimizer.groups must be a mapping")
        group_overrides = []
        for group_name, raw_group in sorted(
            raw_groups.items(), key=lambda item: str(item[0])
        ):
            if not isinstance(group_name, str) or not group_name:
                raise SpecError("optimizer group names must be non-empty strings")
            if not isinstance(raw_group, Mapping):
                raise SpecError(f"optimizer.groups.{group_name} must be a mapping")
            group_unknown = sorted(
                set(raw_group) - {"learning_rate", "weight_decay"}
            )
            if group_unknown:
                raise SpecError(
                    f"optimizer.groups.{group_name} contains unknown keys: "
                    f"{', '.join(group_unknown)}"
                )
            try:
                group_overrides.append(
                    OptimizerGroupOverride(
                        name=group_name,
                        learning_rate=(
                            None
                            if raw_group.get("learning_rate") is None
                            else float(raw_group["learning_rate"])
                        ),
                        weight_decay=(
                            None
                            if raw_group.get("weight_decay") is None
                            else float(raw_group["weight_decay"])
                        ),
                    )
                )
            except ValueError as exc:
                raise SpecError(f"invalid optimizer group {group_name!r}: {exc}") from exc
        if name != "adamw":
            raise SpecError(f"unsupported optimizer: {name}")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise SpecError("optimizer.learning_rate must be positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise SpecError("optimizer.weight_decay must be non-negative")
        if not all(math.isfinite(value) for value in betas) or (
            not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1
        ):
            raise SpecError("optimizer.betas values must be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise SpecError("optimizer.eps must be positive")
        return cls(
            name,
            learning_rate,
            weight_decay,
            betas,
            eps,
            raw_foreach,
            tuple(group_overrides),
        )


@dataclass(frozen=True, slots=True)
class SchedulerSpec:
    name: str = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SchedulerSpec:
        allowed = {"name", "warmup_steps", "min_lr_ratio"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"scheduler contains unknown keys: {', '.join(unknown)}")
        name = str(value.get("name", "constant"))
        warmup_steps = _strict_int(
            value.get("warmup_steps", 0), field="scheduler.warmup_steps"
        )
        min_lr_ratio = float(value.get("min_lr_ratio", 0.0))
        if name not in {"constant", "linear", "cosine"}:
            raise SpecError("scheduler.name must be constant, linear, or cosine")
        if warmup_steps < 0:
            raise SpecError("scheduler.warmup_steps must be non-negative")
        if not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1:
            raise SpecError("scheduler.min_lr_ratio must be in [0, 1]")
        return cls(name, warmup_steps, min_lr_ratio)


@dataclass(frozen=True, slots=True)
class ActivationCheckpointSpec:
    enabled: bool = False
    components: tuple[str, ...] = ()
    use_reentrant: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ActivationCheckpointSpec:
        allowed = {"enabled", "components", "use_reentrant"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(
                f"activation_checkpointing contains unknown keys: {', '.join(unknown)}"
            )
        enabled = value.get("enabled", False)
        use_reentrant = value.get("use_reentrant", False)
        components = value.get("components", ())
        if not isinstance(enabled, bool) or not isinstance(use_reentrant, bool):
            raise SpecError(
                "activation_checkpointing enabled/use_reentrant must be booleans"
            )
        if not isinstance(components, (tuple, list)) or any(
            not isinstance(item, str) or not item for item in components
        ):
            raise SpecError("activation_checkpointing.components must be names")
        if enabled and not components:
            raise SpecError(
                "activation_checkpointing.components is required when enabled"
            )
        return cls(enabled, tuple(components), use_reentrant)


@dataclass(frozen=True, slots=True)
class UpdateEvidenceSpec:
    enabled: bool = False
    every_steps: int = 1
    required_groups: tuple[str, ...] = ()
    sample_elements_per_group: int = 2048

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UpdateEvidenceSpec:
        allowed = {
            "enabled",
            "every_steps",
            "required_groups",
            "sample_elements_per_group",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(
                f"update_evidence contains unknown keys: {', '.join(unknown)}"
            )
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise SpecError("update_evidence.enabled must be a boolean")
        every_steps = _strict_int(
            value.get("every_steps", 1), field="update_evidence.every_steps"
        )
        sample_elements = _strict_int(
            value.get("sample_elements_per_group", 2048),
            field="update_evidence.sample_elements_per_group",
        )
        raw_groups = value.get("required_groups", ())
        if not isinstance(raw_groups, (tuple, list)) or any(
            not isinstance(group, str) or not group for group in raw_groups
        ):
            raise SpecError("update_evidence.required_groups must contain names")
        groups = tuple(raw_groups)
        if len(groups) != len(set(groups)):
            raise SpecError("update_evidence.required_groups contains duplicates")
        if every_steps <= 0:
            raise SpecError("update_evidence.every_steps must be positive")
        if sample_elements <= 0:
            raise SpecError(
                "update_evidence.sample_elements_per_group must be positive"
            )
        if groups and not enabled:
            raise SpecError(
                "update_evidence must be enabled when required_groups is non-empty"
            )
        return cls(enabled, every_steps, groups, sample_elements)


@dataclass(frozen=True, slots=True)
class CompileSpec:
    enabled: bool = False
    backend: str | None = None
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CompileSpec:
        allowed = {"enabled", "backend", "mode", "fullgraph", "dynamic"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"compile contains unknown keys: {', '.join(unknown)}")
        enabled = value.get("enabled", False)
        fullgraph = value.get("fullgraph", False)
        dynamic = value.get("dynamic")
        backend = value.get("backend")
        mode = value.get("mode")
        if not isinstance(enabled, bool) or not isinstance(fullgraph, bool):
            raise SpecError("compile enabled/fullgraph must be booleans")
        if dynamic is not None and not isinstance(dynamic, bool):
            raise SpecError("compile.dynamic must be true, false, or null")
        if backend is not None and (not isinstance(backend, str) or not backend):
            raise SpecError("compile.backend must be null or a non-empty string")
        if mode is not None and (not isinstance(mode, str) or not mode):
            raise SpecError("compile.mode must be null or a non-empty string")
        if not enabled and any(item is not None for item in (backend, mode, dynamic)):
            raise SpecError("compile options require compile.enabled=true")
        if not enabled and fullgraph:
            raise SpecError("compile.fullgraph requires compile.enabled=true")
        return cls(enabled, backend, mode, fullgraph, dynamic)


@dataclass(frozen=True, slots=True)
class DDPSpec:
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    gradient_as_bucket_view: bool = True
    static_graph: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DDPSpec:
        allowed = {
            "find_unused_parameters",
            "broadcast_buffers",
            "gradient_as_bucket_view",
            "static_graph",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"execution.ddp contains unknown keys: {', '.join(unknown)}")
        values = {
            name: value.get(name, default)
            for name, default in {
                "find_unused_parameters": False,
                "broadcast_buffers": True,
                "gradient_as_bucket_view": True,
                "static_graph": False,
            }.items()
        }
        if any(not isinstance(item, bool) for item in values.values()):
            raise SpecError("execution.ddp options must be booleans")
        if values["static_graph"] and values["find_unused_parameters"]:
            raise SpecError(
                "execution.ddp.static_graph cannot be combined with "
                "find_unused_parameters"
            )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class FSDP2Spec:
    wrap_policy: str = "model_declared"
    reshard_after_forward: bool = True
    cpu_offload: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FSDP2Spec:
        allowed = {"wrap_policy", "reshard_after_forward", "cpu_offload"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(
                f"execution.fsdp2 contains unknown keys: {', '.join(unknown)}"
            )
        wrap_policy = value.get("wrap_policy", "model_declared")
        if wrap_policy not in {"model_declared", "root"}:
            raise SpecError(
                "execution.fsdp2.wrap_policy must be model_declared or root"
            )
        reshard = value.get("reshard_after_forward", True)
        cpu_offload = value.get("cpu_offload", False)
        if not isinstance(reshard, bool) or not isinstance(cpu_offload, bool):
            raise SpecError(
                "execution.fsdp2 reshard_after_forward/cpu_offload must be booleans"
            )
        return cls(str(wrap_policy), reshard, cpu_offload)


@dataclass(frozen=True, slots=True)
class DeepSpeedSpec:
    zero_stage: int = 2
    offload_optimizer: str = "none"
    offload_parameters: str = "none"
    overlap_comm: bool = True
    contiguous_gradients: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DeepSpeedSpec:
        allowed = {
            "zero_stage",
            "offload_optimizer",
            "offload_parameters",
            "overlap_comm",
            "contiguous_gradients",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(
                f"execution.deepspeed contains unknown keys: {', '.join(unknown)}"
            )
        zero_stage = _strict_int(
            value.get("zero_stage", 2), field="execution.deepspeed.zero_stage"
        )
        if zero_stage not in {0, 1, 2, 3}:
            raise SpecError("execution.deepspeed.zero_stage must be 0, 1, 2, or 3")
        offload_optimizer = value.get("offload_optimizer", "none")
        offload_parameters = value.get("offload_parameters", "none")
        for field, selected in (
            ("offload_optimizer", offload_optimizer),
            ("offload_parameters", offload_parameters),
        ):
            if selected not in {"none", "cpu"}:
                raise SpecError(
                    f"execution.deepspeed.{field} must be none or cpu"
                )
        if zero_stage != 3 and offload_parameters != "none":
            raise SpecError(
                "DeepSpeed parameter offload is only valid with ZeRO stage 3"
            )
        overlap_comm = value.get("overlap_comm", True)
        contiguous_gradients = value.get("contiguous_gradients", True)
        if not isinstance(overlap_comm, bool) or not isinstance(
            contiguous_gradients, bool
        ):
            raise SpecError(
                "execution.deepspeed overlap_comm/contiguous_gradients "
                "must be booleans"
            )
        return cls(
            zero_stage,
            str(offload_optimizer),
            str(offload_parameters),
            overlap_comm,
            contiguous_gradients,
        )


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    backend: str = "single"
    process_group_backend: str = "auto"
    expected_world_size: int | None = None
    timeout_seconds: int = 1800
    ddp: DDPSpec = DDPSpec()
    fsdp2: FSDP2Spec = FSDP2Spec()
    deepspeed: DeepSpeedSpec = DeepSpeedSpec()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExecutionSpec:
        allowed = {
            "backend",
            "process_group_backend",
            "expected_world_size",
            "timeout_seconds",
            "ddp",
            "fsdp2",
            "deepspeed",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"execution contains unknown keys: {', '.join(unknown)}")
        backend = value.get("backend", "single")
        if backend not in {"single", "torch_ddp", "torch_fsdp2", "deepspeed"}:
            raise SpecError(
                "execution.backend must be single, torch_ddp, torch_fsdp2, "
                "or deepspeed"
            )
        process_group_backend = value.get("process_group_backend", "auto")
        if process_group_backend not in {"auto", "gloo", "nccl", "hccl"}:
            raise SpecError(
                "execution.process_group_backend must be auto, gloo, nccl, or hccl"
            )
        raw_world_size = value.get("expected_world_size")
        expected_world_size = (
            None
            if raw_world_size is None
            else _strict_int(
                raw_world_size,
                field="execution.expected_world_size",
            )
        )
        if expected_world_size is not None and expected_world_size <= 0:
            raise SpecError("execution.expected_world_size must be positive")
        timeout_seconds = _strict_int(
            value.get("timeout_seconds", 1800),
            field="execution.timeout_seconds",
        )
        if timeout_seconds <= 0:
            raise SpecError("execution.timeout_seconds must be positive")
        raw_children = {
            name: value.get(name, {}) for name in ("ddp", "fsdp2", "deepspeed")
        }
        if any(not isinstance(child, Mapping) for child in raw_children.values()):
            raise SpecError("execution backend options must be mappings")
        active_child = {
            "torch_ddp": "ddp",
            "torch_fsdp2": "fsdp2",
            "deepspeed": "deepspeed",
        }.get(str(backend))
        inactive_with_values = sorted(
            name
            for name, child in raw_children.items()
            if child and name != active_child
        )
        if inactive_with_values:
            raise SpecError(
                "execution options were supplied for inactive backends: "
                + ", ".join(inactive_with_values)
            )
        return cls(
            backend=str(backend),
            process_group_backend=str(process_group_backend),
            expected_world_size=expected_world_size,
            timeout_seconds=timeout_seconds,
            ddp=DDPSpec.from_mapping(raw_children["ddp"]),
            fsdp2=FSDP2Spec.from_mapping(raw_children["fsdp2"]),
            deepspeed=DeepSpeedSpec.from_mapping(raw_children["deepspeed"]),
        )


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    directory: Path
    every_steps: int = 1
    enabled: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CheckpointSpec:
        unknown = sorted(set(value) - {"directory", "every_steps", "enabled"})
        if unknown:
            raise SpecError(f"checkpoint contains unknown keys: {', '.join(unknown)}")
        raw_directory = value.get("directory")
        if not isinstance(raw_directory, str) or not raw_directory:
            raise SpecError("checkpoint.directory must be a non-empty path string")
        every_steps = _strict_int(
            value.get("every_steps", 1), field="checkpoint.every_steps"
        )
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise SpecError("checkpoint.enabled must be a boolean")
        if every_steps <= 0:
            raise SpecError("checkpoint.every_steps must be positive")
        return cls(Path(raw_directory), every_steps, enabled)


@dataclass(frozen=True, slots=True)
class DataLoaderSpec:
    """Machine-level settings for the default PyTorch data runtime."""

    num_workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    pin_memory: bool = False
    in_order: bool = True
    snapshot_every_n_steps: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DataLoaderSpec:
        allowed = {
            "num_workers",
            "prefetch_factor",
            "persistent_workers",
            "pin_memory",
            "in_order",
            "snapshot_every_n_steps",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"data_loader contains unknown keys: {', '.join(unknown)}")
        num_workers = _strict_int(
            value.get("num_workers", 0), field="data_loader.num_workers"
        )
        raw_prefetch = value.get("prefetch_factor")
        prefetch_factor = (
            None
            if raw_prefetch is None
            else _strict_int(raw_prefetch, field="data_loader.prefetch_factor")
        )
        persistent_workers = value.get("persistent_workers", False)
        pin_memory = value.get("pin_memory", False)
        in_order = value.get("in_order", True)
        snapshot_every_n_steps = _strict_int(
            value.get("snapshot_every_n_steps", 1),
            field="data_loader.snapshot_every_n_steps",
        )
        if num_workers < 0:
            raise SpecError("data_loader.num_workers must be non-negative")
        if prefetch_factor is not None and prefetch_factor <= 0:
            raise SpecError("data_loader.prefetch_factor must be positive when set")
        if num_workers == 0 and prefetch_factor is not None:
            raise SpecError("data_loader.prefetch_factor requires num_workers > 0")
        if not all(
            isinstance(item, bool)
            for item in (persistent_workers, pin_memory, in_order)
        ):
            raise SpecError(
                "data_loader persistent_workers/pin_memory/in_order must be booleans"
            )
        if persistent_workers and num_workers == 0:
            raise SpecError("data_loader.persistent_workers requires num_workers > 0")
        if not in_order:
            raise SpecError(
                "data_loader.in_order=false is not supported because TorchData "
                "does not guarantee resumable state for out-of-order delivery"
            )
        if snapshot_every_n_steps <= 0:
            raise SpecError("data_loader.snapshot_every_n_steps must be positive")
        return cls(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            in_order=in_order,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )


@dataclass(frozen=True, slots=True)
class RunSpec:
    schema_version: int
    name: str
    seed: int
    deterministic: bool
    device: str
    precision: str
    attention_kernel: str
    max_steps: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float | None
    optimizer: OptimizerSpec
    scheduler: SchedulerSpec
    activation_checkpointing: ActivationCheckpointSpec
    compile: CompileSpec
    execution: ExecutionSpec
    update_evidence: UpdateEvidenceSpec
    data_loader: DataLoaderSpec
    checkpoint: CheckpointSpec

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RunSpec:
        allowed = {
            "schema_version",
            "name",
            "seed",
            "deterministic",
            "device",
            "precision",
            "attention_kernel",
            "max_steps",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "optimizer",
            "scheduler",
            "activation_checkpointing",
            "compile",
            "execution",
            "update_evidence",
            "data_loader",
            "checkpoint",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"run contains unknown keys: {', '.join(unknown)}")
        version = value.get("schema_version")
        _strict_int(version, field="run.schema_version")
        if version != 1:
            raise SpecError(f"unsupported run schema_version: {version!r}")
        name = value.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SpecError("run.name must be a non-empty string")
        seed = _strict_int(value.get("seed", 0), field="run.seed")
        if not 0 <= seed < 2**32:
            raise SpecError("run.seed must be in [0, 2**32)")
        deterministic = value.get("deterministic", False)
        if not isinstance(deterministic, bool):
            raise SpecError("run.deterministic must be a boolean")
        max_steps = _strict_int(value.get("max_steps", 0), field="run.max_steps")
        batch_size = _strict_int(
            value.get("per_device_batch_size", 1),
            field="run.per_device_batch_size",
        )
        accumulation = _strict_int(
            value.get("gradient_accumulation_steps", 1),
            field="run.gradient_accumulation_steps",
        )
        if max_steps <= 0:
            raise SpecError("run.max_steps must be positive")
        if batch_size <= 0:
            raise SpecError("run.per_device_batch_size must be positive")
        if accumulation <= 0:
            raise SpecError("run.gradient_accumulation_steps must be positive")
        raw_grad_norm = value.get("max_grad_norm")
        max_grad_norm = None if raw_grad_norm is None else float(raw_grad_norm)
        if max_grad_norm is not None and (
            not math.isfinite(max_grad_norm) or max_grad_norm <= 0
        ):
            raise SpecError("run.max_grad_norm must be positive when set")
        raw_optimizer = value.get("optimizer", {})
        raw_scheduler = value.get("scheduler", {})
        raw_activation_checkpointing = value.get("activation_checkpointing", {})
        raw_compile = value.get("compile", {})
        raw_execution = value.get("execution", {})
        raw_update_evidence = value.get("update_evidence", {})
        raw_data_loader = value.get("data_loader", {})
        raw_checkpoint = value.get("checkpoint")
        if not isinstance(raw_optimizer, Mapping):
            raise SpecError("run.optimizer must be a mapping")
        if not isinstance(raw_scheduler, Mapping):
            raise SpecError("run.scheduler must be a mapping")
        if not isinstance(raw_activation_checkpointing, Mapping):
            raise SpecError("run.activation_checkpointing must be a mapping")
        if not isinstance(raw_compile, Mapping):
            raise SpecError("run.compile must be a mapping")
        if not isinstance(raw_execution, Mapping):
            raise SpecError("run.execution must be a mapping")
        if not isinstance(raw_update_evidence, Mapping):
            raise SpecError("run.update_evidence must be a mapping")
        if not isinstance(raw_data_loader, Mapping):
            raise SpecError("run.data_loader must be a mapping")
        if not isinstance(raw_checkpoint, Mapping):
            raise SpecError("run.checkpoint must be a mapping")
        precision = str(value.get("precision", "fp32"))
        if precision not in {"fp32", "bf16_mixed", "fp16_mixed", "bf16_true"}:
            raise SpecError(
                "run.precision must be fp32, bf16_mixed, fp16_mixed, or bf16_true"
            )
        attention_kernel = str(value.get("attention_kernel", "auto"))
        if attention_kernel not in {"auto", "eager", "sdpa", "flash_attention_2"}:
            raise SpecError(
                "run.attention_kernel must be auto, eager, sdpa, or flash_attention_2"
            )
        return cls(
            schema_version=version,
            name=name.strip(),
            seed=seed,
            deterministic=deterministic,
            device=str(value.get("device", "cpu")),
            precision=precision,
            attention_kernel=attention_kernel,
            max_steps=max_steps,
            per_device_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            max_grad_norm=max_grad_norm,
            optimizer=OptimizerSpec.from_mapping(raw_optimizer),
            scheduler=SchedulerSpec.from_mapping(raw_scheduler),
            activation_checkpointing=ActivationCheckpointSpec.from_mapping(
                raw_activation_checkpointing
            ),
            compile=CompileSpec.from_mapping(raw_compile),
            execution=ExecutionSpec.from_mapping(raw_execution),
            update_evidence=UpdateEvidenceSpec.from_mapping(raw_update_evidence),
            data_loader=DataLoaderSpec.from_mapping(raw_data_loader),
            checkpoint=CheckpointSpec.from_mapping(raw_checkpoint),
        )

    @property
    def semantic_identity(self) -> RunSpec:
        # The checkpoint directory is a physical output location, not execution
        # semantics.  Keeping it out of the run identity makes an otherwise
        # identical run movable without weakening any of the resume checks for
        # optimizer, scheduler, precision, topology, or batching configuration.
        return replace(
            self,
            checkpoint=replace(
                self.checkpoint,
                directory=Path("<physical-checkpoint-output>"),
            ),
        )

    @property
    def digest(self) -> str:
        return identity_digest(self.semantic_identity)

    @property
    def legacy_path_bound_digest(self) -> str:
        """Pre-fix v1 digest retained only for same-path checkpoint resume."""

        return identity_digest(self)
