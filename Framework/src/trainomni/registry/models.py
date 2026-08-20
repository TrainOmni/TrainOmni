"""Safe discovery and explicit loading of model-family plugins."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainomni.models import ModelPluginManifest, validate_plugin_shape

MODEL_PLUGIN_ENTRYPOINT = "trainomni.model_plugins"


class PluginRegistryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PluginRecord:
    manifest: ModelPluginManifest
    plugin: Any
    source: str
    external: bool

    def to_dict(self) -> dict[str, object]:
        capabilities = self.manifest.capabilities
        return {
            "plugin_id": self.manifest.plugin_id,
            "plugin_version": self.manifest.plugin_version,
            "api_version": self.manifest.api_version,
            "source": self.source,
            "external": self.external,
            "requires_remote_code": self.manifest.requires_remote_code,
            "component_ids": list(self.manifest.component_ids),
            "model_patterns": list(self.manifest.model_patterns),
            "dependency_constraints": list(
                self.manifest.dependency_constraints
            ),
            "capabilities": {
                "modalities": sorted(capabilities.modalities),
                "content_blocks": sorted(capabilities.content_blocks),
                "objectives": sorted(capabilities.objectives),
                "max_media_per_sample": capabilities.max_media_per_sample,
                "packing": capabilities.supports_packing,
                "padding_free": capabilities.supports_padding_free,
                "generation": capabilities.supports_generation,
                "attention_backends": sorted(capabilities.attention_backends),
                "parallelism": sorted(capabilities.parallelism),
                "engine_backends": sorted(capabilities.engine_backends),
                "export_formats": sorted(capabilities.export_formats),
            },
        }


@dataclass(frozen=True, slots=True)
class EntryPointCandidate:
    name: str
    value: str
    group: str = MODEL_PLUGIN_ENTRYPOINT

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "value": self.value, "group": self.group}


class ModelPluginRegistry:
    def __init__(self) -> None:
        self._records: dict[str, PluginRecord] = {}

    def register(
        self, plugin: Any, *, source: str, external: bool = False
    ) -> PluginRecord:
        report = validate_plugin_shape(plugin)
        if not report.valid:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in report.errors
            )
            raise PluginRegistryError(f"invalid model plugin from {source}: {details}")
        manifest = plugin.manifest
        existing = self._records.get(manifest.plugin_id)
        if existing is not None:
            if (
                existing.manifest.plugin_version == manifest.plugin_version
                and existing.source == source
            ):
                return existing
            raise PluginRegistryError(
                f"plugin ID {manifest.plugin_id!r} already registered from "
                f"{existing.source} at version {existing.manifest.plugin_version}"
            )
        record = PluginRecord(
            manifest=manifest, plugin=plugin, source=source, external=external
        )
        self._records[manifest.plugin_id] = record
        return record

    def get(self, plugin_id: str) -> PluginRecord:
        try:
            return self._records[plugin_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._records)) or "none"
            raise PluginRegistryError(
                f"model plugin {plugin_id!r} is not loaded; loaded plugins: {available}"
            ) from exc

    def records(self) -> tuple[PluginRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @staticmethod
    def entry_point_candidates() -> tuple[EntryPointCandidate, ...]:
        entry_points = importlib.metadata.entry_points()
        selected = entry_points.select(group=MODEL_PLUGIN_ENTRYPOINT)
        return tuple(
            EntryPointCandidate(name=item.name, value=item.value)
            for item in sorted(selected, key=lambda value: value.name)
        )

    def load_entry_points(self, *, allow_external: bool) -> tuple[PluginRecord, ...]:
        if not allow_external:
            raise PluginRegistryError(
                "loading installed model plugins requires explicit allow_external=True"
            )
        loaded = []
        entry_points = importlib.metadata.entry_points().select(
            group=MODEL_PLUGIN_ENTRYPOINT
        )
        for entry_point in sorted(entry_points, key=lambda value: value.name):
            try:
                target = entry_point.load()
                plugin = _materialize_plugin(target)
                loaded.append(
                    self.register(
                        plugin,
                        source=f"entrypoint:{entry_point.name}={entry_point.value}",
                        external=True,
                    )
                )
            except Exception as exc:
                raise PluginRegistryError(
                    f"failed to load model plugin entry point {entry_point.name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        return tuple(loaded)

    def load_explicit(
        self, specification: str, *, allow_external: bool
    ) -> PluginRecord:
        if not allow_external:
            raise PluginRegistryError(
                "explicit model plugin loading requires allow_external=True"
            )
        target = _load_explicit_target(specification)
        plugin = _materialize_plugin(target)
        return self.register(plugin, source=specification, external=True)


def _split_specification(specification: str) -> tuple[str, str]:
    try:
        location, attribute = specification.rsplit(":", 1)
    except ValueError as exc:
        raise PluginRegistryError(
            "plugin specification must be MODULE:ATTRIBUTE or FILE.py:ATTRIBUTE"
        ) from exc
    if not location.strip() or not attribute.strip():
        raise PluginRegistryError(
            "plugin specification must include a non-empty location and attribute"
        )
    return location, attribute


def _load_explicit_target(specification: str) -> Any:
    location, attribute = _split_specification(specification)
    candidate = Path(location)
    if candidate.suffix.lower() == ".py" or candidate.exists():
        path = candidate.resolve()
        if not path.is_file() or path.suffix.lower() != ".py":
            raise PluginRegistryError(f"plugin file does not exist: {path}")
        module_name = "_trainomni_external_" + hashlib.sha256(
            str(path).encode("utf-8")
        ).hexdigest()[:16]
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise PluginRegistryError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    else:
        module = importlib.import_module(location)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise PluginRegistryError(
            f"plugin target {attribute!r} is missing from {location!r}"
        ) from exc


def _materialize_plugin(target: Any) -> Any:
    if isinstance(target, type):
        target = target()
    if isinstance(getattr(target, "manifest", None), ModelPluginManifest):
        return target
    if callable(target):
        plugin = target()
        if isinstance(getattr(plugin, "manifest", None), ModelPluginManifest):
            return plugin
    raise PluginRegistryError(
        "plugin target must be an instance with ModelPluginManifest or a zero-arg factory"
    )
