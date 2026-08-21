"""Read-only task inspection and preflight."""

from __future__ import annotations

from pathlib import Path

from trainomni.assembly.preflight import preflight_task
from trainomni.assembly.task_builder import module_lock
from trainomni.catalog.builtin import builtin_registry
from trainomni.catalog.local import registry_for_task
from trainomni.core.resolver import ModuleResolver
from trainomni.specs.loading import load_task


def inspect_task(
    task_path: str | Path, *, allow_local_code: bool = False
) -> dict[str, object]:
    path = Path(task_path).resolve()
    task = load_task(path)
    registry = registry_for_task(
        builtin_registry(),
        task,
        task_root=path.parent,
        allow_local_code=allow_local_code,
    )
    report = preflight_task(task, ModuleResolver(registry))
    return {
        "name": task.name,
        "task_digest": task.digest,
        "modules": dict(module_lock(task)),
        "capabilities": tuple(sorted(report.capabilities.values)),
        "components": tuple(name for name, _ in task.model.components),
        "local_modules": tuple(str(item.module_id) for item in task.local_modules),
    }
