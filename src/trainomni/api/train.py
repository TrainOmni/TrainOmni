"""Stable Python operation for assembling and training one task/run pair."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from trainomni.artifacts.manifest import materialize_run_identity
from trainomni.assembly.task_builder import TaskAssembly, build_task
from trainomni.catalog.builtin import builtin_registry
from trainomni.catalog.local import registry_for_task
from trainomni.core.errors import CheckpointError, SpecError
from trainomni.core.resolver import ModuleResolver
from trainomni.runtime.loop.engine import StepMetrics, TrainEngine
from trainomni.runtime.random import seed_everything
from trainomni.specs.loading import load_run, load_task
from trainomni.specs.run import CheckpointSpec
from trainomni.specs.task import TaskSpec


@dataclass(frozen=True, slots=True)
class TrainResult:
    task_digest: str
    run_digest: str
    records: tuple[StepMetrics, ...]
    final_step: int


def assemble(
    *,
    task_path: str | Path,
    allow_local_code: bool = False,
    operation: str = "all",
) -> tuple[TaskSpec, TaskAssembly]:
    path = Path(task_path).resolve()
    task = load_task(path)
    registry = registry_for_task(
        builtin_registry(),
        task,
        task_root=path.parent,
        allow_local_code=allow_local_code,
    )
    assembly = build_task(
        task,
        ModuleResolver(registry),
        task_root=path.parent,
        operation=operation,
    )
    return task, assembly


def load_resolved_run(run_path: str | Path):
    resolved_run_path = Path(run_path).resolve()
    run = load_run(resolved_run_path)
    if not run.checkpoint.directory.is_absolute():
        run = replace(
            run,
            checkpoint=CheckpointSpec(
                directory=(resolved_run_path.parent / run.checkpoint.directory).resolve(),
                every_steps=run.checkpoint.every_steps,
                enabled=run.checkpoint.enabled,
            ),
        )
    return run


def build_engine(*, task, assembly: TaskAssembly, run) -> TrainEngine:
    selection = assembly.parameter_selection
    engine = TrainEngine(
        model=assembly.model,
        objective=assembly.objective,
        parameter_selection=selection,
        stream=assembly.stream,
        run=run,
        task_digest=task.digest,
        module_lock=dict(assembly.module_lock),
        reproducible=assembly.reproducible,
        provenance_issues=assembly.provenance_issues,
    )
    try:
        def materialize_identity():
            materialize_run_identity(
                output_root=run.checkpoint.directory.parent,
                task=task,
                run=run,
                module_lock=assembly.module_lock,
                parameter_selection=selection,
                reproducible=assembly.reproducible,
                provenance_issues=assembly.provenance_issues,
            )

        engine.process.coordinate_primary(
            materialize_identity,
            owner="run identity materialization",
            error_type=CheckpointError,
        )
    except Exception:
        engine.close()
        raise
    return engine


def train(
    *,
    task_path: str | Path,
    run_path: str | Path,
    allow_local_code: bool = False,
    resume_from: str | Path | None = None,
    stop_after_steps: int | None = None,
) -> TrainResult:
    run = load_resolved_run(run_path)
    seed_everything(run.seed, deterministic=run.deterministic)
    task, assembly = assemble(
        task_path=task_path,
        allow_local_code=allow_local_code,
        operation="train",
    )
    engine = build_engine(task=task, assembly=assembly, run=run)
    try:
        if resume_from is not None:
            engine.resume(Path(resume_from).resolve())
        elif run.checkpoint.directory.exists() and any(
            run.checkpoint.directory.glob("step-*")
        ):
            raise SpecError(
                "checkpoint directory already contains steps; use --resume or a new output"
            )
        records = engine.train(stop_after_steps=stop_after_steps)
        return TrainResult(
            task_digest=task.digest,
            run_digest=run.digest,
            records=records,
            final_step=engine.global_step,
        )
    finally:
        engine.close()
