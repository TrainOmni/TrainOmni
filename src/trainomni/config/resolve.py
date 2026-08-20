"""Resolve a validated user run specification against a model plugin."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from trainomni.contracts import ValidationIssue, ValidationReport
from trainomni.models import (
    ModelPluginManifest,
    ModelRequirements,
    negotiate_capabilities,
)

from .schema import RUN_SCHEMA_VERSION, RunSpec

RESOLVED_SCHEMA_VERSION = "trainomni.resolved-run.v1"


def _canonical_value(value: Any) -> Any:
    """Normalize typed config values without losing unordered-container semantics."""

    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def requirements_from_run(spec: RunSpec) -> ModelRequirements:
    return ModelRequirements(
        modalities=spec.stage.data.modalities,
        content_blocks=spec.stage.data.content_blocks,
        objectives=frozenset({spec.stage.objective}),
        media_per_sample=spec.stage.data.max_media_per_sample,
        require_packing=spec.stage.data.packing,
        require_padding_free=spec.stage.data.padding_free,
        require_generation=spec.stage.stage_type
        in {"online_rl", "agentic_rl", "evaluate_export"},
        attention_backend=spec.stage.engine.attention_backend,
        parallelism=spec.stage.engine.parallelism,
        engine_backend=spec.stage.engine.backend,
    )


@dataclass(frozen=True, slots=True)
class ResolvedRunSpec:
    run: RunSpec
    plugin_manifest: ModelPluginManifest
    requirements: ModelRequirements
    fingerprint: str
    source: Path | None = None
    schema_version: str = RESOLVED_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "source": str(self.source) if self.source is not None else None,
            "run": self.run.model_dump(mode="json"),
            "plugin": {
                "plugin_id": self.plugin_manifest.plugin_id,
                "plugin_version": self.plugin_manifest.plugin_version,
                "api_version": self.plugin_manifest.api_version,
            },
            "requirements": {
                "modalities": sorted(self.requirements.modalities),
                "content_blocks": sorted(self.requirements.content_blocks),
                "objectives": sorted(self.requirements.objectives),
                "media_per_sample": self.requirements.media_per_sample,
                "packing": self.requirements.require_packing,
                "padding_free": self.requirements.require_padding_free,
                "generation": self.requirements.require_generation,
                "attention_backend": self.requirements.attention_backend,
                "parallelism": self.requirements.parallelism,
                "engine_backend": self.requirements.engine_backend,
            },
        }


def resolve_run(
    spec: RunSpec,
    manifest: ModelPluginManifest,
    *,
    source: str | Path | None = None,
) -> tuple[ResolvedRunSpec | None, ValidationReport]:
    issues: list[ValidationIssue] = []
    if spec.schema_version != RUN_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                code="config.schema_version",
                message=f"unsupported run schema {spec.schema_version!r}",
                path="schema_version",
            )
        )
    if spec.model.plugin != manifest.plugin_id:
        issues.append(
            ValidationIssue(
                code="plugin.identity",
                message=(
                    f"recipe requests plugin {spec.model.plugin!r}, loaded plugin is "
                    f"{manifest.plugin_id!r}"
                ),
                path="model.plugin",
            )
        )
    unknown_components = set(spec.stage.component_policy) - set(manifest.component_ids)
    if unknown_components:
        issues.append(
            ValidationIssue(
                code="plugin.component_policy",
                message=(
                    "component policy references unsupported components: "
                    f"{sorted(unknown_components)}"
                ),
                path="stage.component_policy",
            )
        )
    if (
        spec.stage.stage_type != "evaluate_export"
        and spec.stage.component_policy
        and not any(
            policy.trainable for policy in spec.stage.component_policy.values()
        )
    ):
        issues.append(
            ValidationIssue(
                code="config.no_trainable_component",
                message="a training stage must make at least one component trainable",
                path="stage.component_policy",
            )
        )
    requirements = requirements_from_run(spec)
    capability_report = negotiate_capabilities(requirements, manifest.capabilities)
    issues.extend(
        ValidationIssue(code=item.code, message=item.message, path="stage")
        for item in capability_report.issues
    )
    report = ValidationReport(tuple(issues))
    if not report.valid:
        return None, report

    source_path = Path(source).resolve() if source is not None else None
    identity = {
        "run": spec,
        "plugin": {
            "plugin_id": manifest.plugin_id,
            "plugin_version": manifest.plugin_version,
            "api_version": manifest.api_version,
        },
    }
    resolved = ResolvedRunSpec(
        run=spec,
        plugin_manifest=manifest,
        requirements=requirements,
        fingerprint=canonical_fingerprint(identity),
        source=source_path,
    )
    return resolved, report
