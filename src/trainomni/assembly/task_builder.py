"""Typed executable task assembly.

The builder receives an already-preflighted task and only injects the narrow
BuildContext each module needs. It never passes RunSpec to semantic modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trainomni import BUILTIN_CODE_PROVENANCE, __version__
from trainomni.core.assets import task_asset_provenance
from trainomni.core.context import BuildContext
from trainomni.core.module import ModuleKind
from trainomni.core.provenance import builtin_source_sha256
from trainomni.core.resolver import ModuleResolver
from trainomni.specs.digest import identity_digest
from trainomni.specs.task import TaskSpec, semantic_module_identity

from .data_builder import build_data_stream
from .model_builder import build_model
from .preflight import preflight_task


@dataclass(frozen=True, slots=True)
class TaskAssembly:
    stream: Any | None
    evaluation_stream: Any | None
    evaluators: tuple[Any, ...]
    exporters: tuple[tuple[str, Any], ...]
    model: Any
    objective: Any | None
    parameter_policy: Any
    parameter_selection: Any
    components: MappingProxyType
    module_lock: MappingProxyType
    processor: Any | None = None
    reproducible: bool = True
    provenance_issues: tuple[str, ...] = ()


def module_lock(
    task: TaskSpec, *, task_root: Path | None = None
) -> MappingProxyType:
    lock = {
        f"{index:04d}:{reference.module_id}": identity_digest(
            semantic_module_identity(reference)
        )
        for index, reference in enumerate(task.module_refs())
    }
    lock["builtin-core:trainomni"] = identity_digest(
        {
            "framework_version": __version__,
            "builtin_code_provenance": BUILTIN_CODE_PROVENANCE,
            "builtin_source_sha256": builtin_source_sha256(),
        }
    )
    lock.update(
        {
            f"local-source:{source.module_id}": source.source_sha256
            for source in task.local_modules
        }
    )
    lock.update(task_asset_provenance(task, task_root=task_root).lock_entries)
    return MappingProxyType(lock)


def build_task(
    task: TaskSpec,
    resolver: ModuleResolver,
    *,
    task_root: Path | None = None,
    operation: str = "all",
) -> TaskAssembly:
    if operation not in {"all", "train", "evaluate", "export"}:
        raise ValueError(f"unknown task assembly operation: {operation!r}")
    preflight_task(task, resolver)
    base_context = BuildContext(task_digest=task.digest, task_root=task_root)
    built_model = build_model(task.model, resolver, context=base_context)
    model = built_model.model
    objective = (
        resolver.resolve(task.objective, kind=ModuleKind.OBJECTIVE).build(base_context)
        if operation in {"all", "train", "evaluate"}
        else None
    )
    parameter_policy = resolver.resolve(
        task.parameters, kind=ModuleKind.PARAMETER_POLICY
    ).build(base_context)
    parameter_selection = parameter_policy.apply(model)
    stream = (
        build_data_stream(task.data, resolver, context=base_context)
        if operation in {"all", "train"}
        else None
    )
    evaluation_stream = None
    if stream is not None:
        stream.bind_local_sources(task.local_modules, task_root)
    evaluators = ()
    if task.evaluation is not None and operation in {"all", "evaluate"}:
        evaluation_stream = build_data_stream(
            task.evaluation.data, resolver, context=base_context
        )
        evaluation_stream.bind_local_sources(task.local_modules, task_root)
        evaluators = tuple(
            resolver.resolve(reference, kind=ModuleKind.EVALUATOR).build(base_context)
            for reference in task.evaluation.evaluators
        )
    exporters = (
        tuple(
            (
                str(reference.module_id),
                resolver.resolve(reference, kind=ModuleKind.EXPORTER).build(base_context),
            )
            for reference in task.exporters
        )
        if operation in {"all", "export"}
        else ()
    )
    processor = None
    if operation == "export":
        model_io = resolver.resolve(task.data.model_io, kind=ModuleKind.MODEL_IO).build(
            base_context
        )
        processor = getattr(model_io, "processor", None)
    elif stream is not None:
        processor = getattr(stream.model_io, "processor", None)
    provenance = task_asset_provenance(task, task_root=task_root)
    return TaskAssembly(
        stream=stream,
        evaluation_stream=evaluation_stream,
        evaluators=evaluators,
        exporters=exporters,
        model=model,
        objective=objective,
        parameter_policy=parameter_policy,
        parameter_selection=parameter_selection,
        components=built_model.components,
        module_lock=module_lock(task, task_root=task_root),
        processor=processor,
        reproducible=provenance.reproducible,
        provenance_issues=provenance.issues,
    )
