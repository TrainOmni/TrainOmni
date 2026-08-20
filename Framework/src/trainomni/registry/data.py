"""Explicitly trusted data-extension loading."""

from __future__ import annotations

from dataclasses import dataclass

from trainomni.data import ImporterRegistry, ReaderRegistry

from .models import PluginRegistryError, _load_explicit_target


@dataclass(frozen=True, slots=True)
class DataRegistries:
    readers: ReaderRegistry
    importers: ImporterRegistry


def load_data_plugins(specifications: list[str] | tuple[str, ...]) -> DataRegistries:
    readers = ReaderRegistry()
    importers = ImporterRegistry()
    for specification in specifications:
        target = _load_explicit_target(specification)
        plugin = target() if isinstance(target, type) else target
        register = getattr(plugin, "register", None)
        if not callable(register):
            raise PluginRegistryError(
                f"data plugin {specification!r} must implement register(readers, importers)"
            )
        register(readers, importers)
    return DataRegistries(readers, importers)
