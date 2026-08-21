"""Resolve references without constructing unrelated modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability import CapabilitySet
from .module import ModuleDescriptor, ModuleKind, ModuleRef, parse_config
from .registry import ModuleRegistry


@dataclass(frozen=True, slots=True)
class ResolvedModule:
    reference: ModuleRef
    descriptor: ModuleDescriptor
    config: Any

    def build(self, context: Any) -> Any:
        return self.descriptor.factory(self.config, context)


class ModuleResolver:
    def __init__(self, registry: ModuleRegistry) -> None:
        self.registry = registry

    def resolve(self, reference: ModuleRef, *, kind: ModuleKind) -> ResolvedModule:
        descriptor = self.registry.descriptor(reference, expected_kind=kind)
        config = parse_config(descriptor.config_type, reference.config)
        return ResolvedModule(reference, descriptor, config)

    @staticmethod
    def preflight(
        modules: tuple[ResolvedModule, ...],
        *,
        external: CapabilitySet | None = None,
    ) -> CapabilitySet:
        external = external or CapabilitySet()
        all_provided = external
        for module in modules:
            all_provided = all_provided.union(module.descriptor.provides)
        for index, module in enumerate(modules):
            available = external
            for provider_index, provider in enumerate(modules):
                if provider_index != index:
                    available = available.union(provider.descriptor.provides)
            available.require(
                module.descriptor.requires,
                owner=str(module.reference.module_id),
            )
        return all_provided
