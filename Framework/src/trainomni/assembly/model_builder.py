"""Assemble model components without importing concrete implementations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from trainomni.core.context import BuildContext
from trainomni.core.module import ModuleKind
from trainomni.core.resolver import ModuleResolver
from trainomni.specs.task import ModelAssemblySpec


@dataclass(frozen=True, slots=True)
class ModelAssemblyResult:
    model: Any
    components: MappingProxyType


def build_model(
    spec: ModelAssemblySpec,
    resolver: ModuleResolver,
    *,
    context: BuildContext,
) -> ModelAssemblyResult:
    components: dict[str, Any] = {}
    for name, reference in spec.components:
        resolved = resolver.resolve(reference, kind=reference.module_id.kind)
        components[name] = resolved.build(context)
    if spec.attention_policy is not None:
        components["__attention_policy__"] = resolver.resolve(
            spec.attention_policy, kind=ModuleKind.ATTENTION_POLICY
        ).build(context)
    model_context = BuildContext(
        task_digest=context.task_digest,
        task_root=context.task_root,
        components=MappingProxyType(components),
    )
    model = resolver.resolve(spec.implementation, kind=ModuleKind.MODEL).build(
        model_context
    )
    return ModelAssemblyResult(model=model, components=MappingProxyType(components))
