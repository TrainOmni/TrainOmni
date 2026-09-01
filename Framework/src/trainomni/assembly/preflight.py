"""Resolve and capability-check a task without constructing modules."""

from __future__ import annotations

from dataclasses import dataclass

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import CapabilityError
from trainomni.core.resolver import ModuleResolver, ResolvedModule
from trainomni.specs.task import DataPipelineSpec, TaskSpec


@dataclass(frozen=True, slots=True)
class PreflightReport:
    modules: tuple[ResolvedModule, ...]
    capabilities: CapabilitySet


def _preflight_data_path(
    spec: DataPipelineSpec,
    resolver: ModuleResolver,
    *,
    owner: str,
    external: CapabilitySet,
) -> CapabilitySet:
    source_module = resolver.resolve(spec.source, kind=spec.source.module_id.kind)
    child_contract = {"data.sample.omni", "data.source.stateful"}
    if spec.sources and not child_contract.issubset(
        source_module.descriptor.requires.values
    ):
        raise CapabilityError(
            f"{owner}:{spec.source.module_id} does not declare the stateful "
            "child-source composition contract"
        )
    references = (
        *(reference for _, reference in spec.sources),
        spec.source,
        *((spec.adapter,) if spec.adapter is not None else ()),
        *spec.transforms,
        spec.model_io,
        spec.supervision,
        spec.packer,
        spec.collator,
    )
    available = external
    for reference in references:
        resolved = resolver.resolve(reference, kind=reference.module_id.kind)
        available.require(
            resolved.descriptor.requires,
            owner=f"{owner}:{reference.module_id}",
        )
        available = available.union(resolved.descriptor.provides)
    return available


def preflight_task(task: TaskSpec, resolver: ModuleResolver) -> PreflightReport:
    resolved = tuple(
        resolver.resolve(reference, kind=reference.module_id.kind)
        for reference in task.module_refs()
    )
    data_references = {
        id(reference)
        for reference in (
            *(reference for _, reference in task.data.sources),
            task.data.source,
            *((task.data.adapter,) if task.data.adapter is not None else ()),
            *task.data.transforms,
            task.data.model_io,
            task.data.supervision,
            task.data.packer,
            task.data.collator,
        )
    }
    if task.evaluation is not None:
        data_references.update(
            id(reference)
            for reference in (
                *(reference for _, reference in task.evaluation.data.sources),
                task.evaluation.data.source,
                *(
                    (task.evaluation.data.adapter,)
                    if task.evaluation.data.adapter is not None
                    else ()
                ),
                *task.evaluation.data.transforms,
                task.evaluation.data.model_io,
                task.evaluation.data.supervision,
                task.evaluation.data.packer,
                task.evaluation.data.collator,
            )
        )
    external = CapabilitySet()
    for module in resolved:
        if id(module.reference) not in data_references:
            external = external.union(module.descriptor.provides)
    _preflight_data_path(
        task.data,
        resolver,
        owner="train-data",
        external=external,
    )
    if task.evaluation is not None:
        _preflight_data_path(
            task.evaluation.data,
            resolver,
            owner="evaluation-data",
            external=external,
        )
    return PreflightReport(resolved, resolver.preflight(resolved))
