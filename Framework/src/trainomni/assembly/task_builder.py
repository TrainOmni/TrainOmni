"""Typed executable task assembly.

The builder receives an already-preflighted task and only injects the narrow
BuildContext each module needs. It never passes RunSpec to semantic modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trainomni.core.context import BuildContext
from trainomni.core.module import ModuleKind
from trainomni.core.resolver import ModuleResolver
from trainomni.specs.digest import identity_digest
from trainomni.specs.task import TaskSpec

from .data_builder import build_data_stream
from .model_builder import build_model
from .preflight import preflight_task


@dataclass(frozen=True, slots=True)
class TaskAssembly:
    stream: Any
    evaluation_stream: Any | None
    evaluators: tuple[Any, ...]
    exporters: tuple[tuple[str, Any], ...]
    model: Any
    objective: Any
    parameter_policy: Any
    parameter_selection: Any
    components: MappingProxyType
    module_lock: MappingProxyType


def module_lock(task: TaskSpec) -> MappingProxyType:
    lock = {
        f"{index:04d}:{reference.module_id}": identity_digest(reference)
        for index, reference in enumerate(task.module_refs())
    }
    lock.update(
        {
            f"local-source:{source.module_id}": source.source_sha256
            for source in task.local_modules
        }
    )
    return MappingProxyType(lock)


def build_task(
    task: TaskSpec,
    resolver: ModuleResolver,
    *,
    task_root: Path | None = None,
) -> TaskAssembly:
    preflight_task(task, resolver)
    base_context = BuildContext(task_digest=task.digest, task_root=task_root)
    built_model = build_model(task.model, resolver, context=base_context)
    model = built_model.model
    objective = resolver.resolve(task.objective, kind=ModuleKind.OBJECTIVE).build(base_context)
    parameter_policy = resolver.resolve(
        task.parameters, kind=ModuleKind.PARAMETER_POLICY
    ).build(base_context)
    parameter_selection = parameter_policy.apply(model)
    stream = build_data_stream(task.data, resolver, context=base_context)
    evaluation_stream = None
    evaluators = ()
    if task.evaluation is not None:
        evaluation_stream = build_data_stream(
            task.evaluation.data, resolver, context=base_context
        )
        evaluators = tuple(
            resolver.resolve(reference, kind=ModuleKind.EVALUATOR).build(base_context)
            for reference in task.evaluation.evaluators
        )
    exporters = tuple(
        (
            str(reference.module_id),
            resolver.resolve(reference, kind=ModuleKind.EXPORTER).build(base_context),
        )
        for reference in task.exporters
    )
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
        module_lock=module_lock(task),
    )
