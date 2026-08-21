"""Module descriptor registry."""

from __future__ import annotations

from collections.abc import Iterable

from .errors import RegistryError
from .module import ModuleDescriptor, ModuleId, ModuleKind, ModuleRef


class ModuleRegistry:
    """Explicit registry with no import-time global singleton."""

    def __init__(self, descriptors: Iterable[ModuleDescriptor] = ()) -> None:
        self._descriptors: dict[ModuleId, ModuleDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: ModuleDescriptor) -> None:
        if descriptor.module_id in self._descriptors:
            raise RegistryError(f"duplicate module: {descriptor.module_id}")
        self._descriptors[descriptor.module_id] = descriptor

    def descriptor(
        self,
        reference: ModuleRef,
        *,
        expected_kind: ModuleKind | None = None,
    ) -> ModuleDescriptor:
        if expected_kind is not None and reference.module_id.kind is not expected_kind:
            raise RegistryError(
                f"expected {expected_kind.value}, got {reference.module_id.kind.value}: "
                f"{reference.module_id}"
            )
        try:
            return self._descriptors[reference.module_id]
        except KeyError as exc:
            available = ", ".join(str(item) for item in sorted(self._descriptors)) or "<none>"
            raise RegistryError(
                f"module not registered: {reference.module_id}; available: {available}"
            ) from exc

    def descriptors(self) -> tuple[ModuleDescriptor, ...]:
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))
