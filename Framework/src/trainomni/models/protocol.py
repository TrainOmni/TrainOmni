"""Narrow, engine-neutral contracts for model-family integrations."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from trainomni.data import CanonicalSample

MODEL_PLUGIN_API_VERSION = "trainomni.model-plugin.v1"
_PLUGIN_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    modalities: frozenset[str]
    content_blocks: frozenset[str]
    objectives: frozenset[str]
    max_media_per_sample: int | None = None
    supports_packing: bool = False
    supports_padding_free: bool = False
    supports_generation: bool = True
    attention_backends: frozenset[str] = frozenset()
    parallelism: frozenset[str] = frozenset({"single"})
    engine_backends: frozenset[str] = frozenset({"torch"})
    export_formats: frozenset[str] = frozenset({"hf"})

    def __post_init__(self) -> None:
        for name in (
            "modalities",
            "content_blocks",
            "objectives",
            "parallelism",
            "engine_backends",
            "export_formats",
        ):
            if not getattr(self, name):
                raise ValueError(f"ModelCapabilities.{name} must not be empty")
        if self.max_media_per_sample is not None and self.max_media_per_sample < 0:
            raise ValueError("max_media_per_sample must be non-negative or None")


@dataclass(frozen=True, slots=True)
class ModelRequirements:
    modalities: frozenset[str] = frozenset()
    content_blocks: frozenset[str] = frozenset()
    objectives: frozenset[str] = frozenset()
    media_per_sample: int = 0
    require_packing: bool = False
    require_padding_free: bool = False
    require_generation: bool = False
    attention_backend: str | None = None
    parallelism: str | None = None
    engine_backend: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilityIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    compatible: bool
    issues: tuple[CapabilityIssue, ...]


def negotiate_capabilities(
    requirements: ModelRequirements, capabilities: ModelCapabilities
) -> CapabilityReport:
    """Reject a recipe/model mismatch before loading full model weights."""

    issues: list[CapabilityIssue] = []
    for field_name in ("modalities", "content_blocks", "objectives"):
        required = getattr(requirements, field_name)
        supported = getattr(capabilities, field_name)
        missing = required - supported
        if missing:
            issues.append(
                CapabilityIssue(
                    code=f"capability.{field_name}",
                    message=f"unsupported {field_name}: {sorted(missing)}",
                )
            )
    if (
        capabilities.max_media_per_sample is not None
        and requirements.media_per_sample > capabilities.max_media_per_sample
    ):
        issues.append(
            CapabilityIssue(
                code="capability.media_count",
                message=(
                    f"recipe needs {requirements.media_per_sample} media items, "
                    f"model limit is {capabilities.max_media_per_sample}"
                ),
            )
        )
    for required, supported, name in (
        (requirements.require_packing, capabilities.supports_packing, "packing"),
        (
            requirements.require_padding_free,
            capabilities.supports_padding_free,
            "padding_free",
        ),
        (
            requirements.require_generation,
            capabilities.supports_generation,
            "generation",
        ),
    ):
        if required and not supported:
            issues.append(
                CapabilityIssue(
                    code=f"capability.{name}", message=f"model does not support {name}"
                )
            )
    if (
        requirements.attention_backend is not None
        and requirements.attention_backend not in capabilities.attention_backends
    ):
        issues.append(
            CapabilityIssue(
                code="capability.attention_backend",
                message=(
                    f"attention backend {requirements.attention_backend!r} is unsupported"
                ),
            )
        )
    if (
        requirements.parallelism is not None
        and requirements.parallelism not in capabilities.parallelism
    ):
        issues.append(
            CapabilityIssue(
                code="capability.parallelism",
                message=f"parallelism {requirements.parallelism!r} is unsupported",
            )
        )
    if (
        requirements.engine_backend is not None
        and requirements.engine_backend not in capabilities.engine_backends
    ):
        issues.append(
            CapabilityIssue(
                code="capability.engine_backend",
                message=f"engine backend {requirements.engine_backend!r} is unsupported",
            )
        )
    return CapabilityReport(compatible=not issues, issues=tuple(issues))


@dataclass(frozen=True, slots=True)
class ModelPluginManifest:
    """Static identity and compatibility metadata for one model-family plugin."""

    plugin_id: str
    plugin_version: str
    capabilities: ModelCapabilities
    component_ids: tuple[str, ...] = ()
    model_patterns: tuple[str, ...] = ()
    dependency_constraints: tuple[str, ...] = ()
    requires_remote_code: bool = False
    api_version: str = MODEL_PLUGIN_API_VERSION

    def __post_init__(self) -> None:
        if not _PLUGIN_ID.fullmatch(self.plugin_id):
            raise ValueError(
                "plugin_id must be a lowercase dotted/dashed identifier, "
                f"got {self.plugin_id!r}"
            )
        if not self.plugin_version.strip():
            raise ValueError("plugin_version must not be blank")
        if not isinstance(self.capabilities, ModelCapabilities):
            raise TypeError("capabilities must be a ModelCapabilities instance")
        if not self.component_ids:
            raise ValueError("component_ids must not be empty")
        if len(set(self.component_ids)) != len(self.component_ids):
            raise ValueError("component_ids must not contain duplicates")
        if any(not item.strip() for item in self.component_ids):
            raise ValueError("component_ids must not contain blank values")
        if self.api_version != MODEL_PLUGIN_API_VERSION:
            raise ValueError(
                f"unsupported model plugin API {self.api_version!r}; "
                f"expected {MODEL_PLUGIN_API_VERSION!r}"
            )


@dataclass(frozen=True, slots=True)
class ComponentRule:
    component_id: str
    prefixes: tuple[str, ...]
    required: bool = True

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id must not be blank")
        if not self.prefixes or any(not prefix for prefix in self.prefixes):
            raise ValueError("component prefixes must not be empty")


@dataclass(frozen=True, slots=True)
class ComponentCatalog:
    rules: tuple[ComponentRule, ...]

    def __post_init__(self) -> None:
        component_ids = [rule.component_id for rule in self.rules]
        if not component_ids:
            raise ValueError("component catalog must not be empty")
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("component catalog IDs must be unique")

    def classify(
        self, parameter_names: Iterable[str]
    ) -> tuple[dict[str, tuple[str, ...]], tuple[CapabilityIssue, ...]]:
        """Assign every parameter to exactly one stable component."""

        assignments: dict[str, list[str]] = {rule.component_id: [] for rule in self.rules}
        issues: list[CapabilityIssue] = []
        for name in parameter_names:
            matches = [
                rule.component_id
                for rule in self.rules
                if any(name.startswith(prefix) for prefix in rule.prefixes)
            ]
            if not matches:
                issues.append(
                    CapabilityIssue(
                        code="component.unclassified",
                        message=f"parameter {name!r} matches no component",
                    )
                )
            elif len(matches) > 1:
                issues.append(
                    CapabilityIssue(
                        code="component.ambiguous",
                        message=f"parameter {name!r} matches {matches}",
                    )
                )
            else:
                assignments[matches[0]].append(name)
        for rule in self.rules:
            if rule.required and not assignments[rule.component_id]:
                issues.append(
                    CapabilityIssue(
                        code="component.empty",
                        message=f"required component {rule.component_id!r} is empty",
                    )
                )
        frozen = {key: tuple(value) for key, value in assignments.items()}
        return frozen, tuple(issues)


class ModelFamilyPlugin(Protocol):
    """Public boundary implemented by each native or composite model family."""

    manifest: ModelPluginManifest

    def capabilities(self) -> ModelCapabilities: ...

    def build(self, config: Mapping[str, Any]) -> Any: ...

    def component_catalog(self, bundle: Any) -> ComponentCatalog: ...

    def validate_sample(
        self, sample: CanonicalSample, objective: str
    ) -> tuple[CapabilityIssue, ...]: ...

    def encode(self, sample: CanonicalSample, context: Mapping[str, Any]) -> Any: ...

    def collate(self, samples: list[Any], plan: Mapping[str, Any]) -> Any: ...

    def export(self, bundle: Any, checkpoint: Any, target: Mapping[str, Any]) -> Any: ...
