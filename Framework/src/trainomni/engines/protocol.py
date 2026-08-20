"""Execution-engine contracts shared by loop-owned and delegated stages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from trainomni.contracts import ArtifactRef, ValidationIssue, ValidationReport

ENGINE_API_VERSION = "trainomni.engine.v1"
_ENGINE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class EngineKind(str, Enum):
    LOOP = "loop"
    DELEGATED_STAGE = "delegated_stage"


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    stage_types: frozenset[str]
    objectives: frozenset[str]
    parallelism: frozenset[str]
    precisions: frozenset[str]
    resume_levels: frozenset[str]
    supports_generation: bool = False
    supports_multiple_models: bool = False
    supports_rollout: bool = False


@dataclass(frozen=True, slots=True)
class EngineRequirements:
    stage_type: str
    objective: str
    parallelism: str
    precision: str
    resume_level: str
    require_generation: bool = False
    require_multiple_models: bool = False
    require_rollout: bool = False


@dataclass(frozen=True, slots=True)
class EngineManifest:
    engine_id: str
    engine_version: str
    kind: EngineKind
    capabilities: EngineCapabilities
    dependency_constraints: tuple[str, ...] = ()
    api_version: str = ENGINE_API_VERSION

    def __post_init__(self) -> None:
        if not _ENGINE_ID.fullmatch(self.engine_id):
            raise ValueError(f"invalid engine_id {self.engine_id!r}")
        if not self.engine_version.strip():
            raise ValueError("engine_version must not be blank")
        if self.api_version != ENGINE_API_VERSION:
            raise ValueError(
                f"unsupported engine API {self.api_version!r}; "
                f"expected {ENGINE_API_VERSION!r}"
            )


def negotiate_engine(
    requirements: EngineRequirements, capabilities: EngineCapabilities
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for value, supported, name in (
        (requirements.stage_type, capabilities.stage_types, "stage_type"),
        (requirements.objective, capabilities.objectives, "objective"),
        (requirements.parallelism, capabilities.parallelism, "parallelism"),
        (requirements.precision, capabilities.precisions, "precision"),
        (requirements.resume_level, capabilities.resume_levels, "resume_level"),
    ):
        if value not in supported:
            issues.append(
                ValidationIssue(
                    code=f"engine.{name}",
                    message=f"engine does not support {name} {value!r}",
                    path=f"stage.engine.{name}",
                )
            )
    for required, supported, name in (
        (
            requirements.require_generation,
            capabilities.supports_generation,
            "generation",
        ),
        (
            requirements.require_multiple_models,
            capabilities.supports_multiple_models,
            "multiple_models",
        ),
        (requirements.require_rollout, capabilities.supports_rollout, "rollout"),
    ):
        if required and not supported:
            issues.append(
                ValidationIssue(
                    code=f"engine.{name}",
                    message=f"engine does not support required {name}",
                    path="stage.engine",
                )
            )
    return ValidationReport(tuple(issues))


@dataclass(frozen=True, slots=True)
class PreparedStage:
    stage_id: str
    state: Any


@dataclass(frozen=True, slots=True)
class StageResult:
    stage_id: str
    status: str
    outputs: Mapping[str, ArtifactRef]
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EngineAdapter(Protocol):
    """Common envelope; implementations may own a loop or delegate a stage."""

    manifest: EngineManifest

    def validate(self, stage: Any, model: Any) -> ValidationReport: ...

    def prepare(self, context: Any) -> PreparedStage: ...

    def run(self, prepared: PreparedStage) -> StageResult: ...

    def checkpoint(self, prepared: PreparedStage, reason: str) -> ArtifactRef: ...

    def collect(self, result: StageResult) -> Any: ...
