"""Engine-neutral objective contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

OBJECTIVE_API_VERSION = "trainomni.objective.v1"
_OBJECTIVE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class ObjectiveRequirements:
    sample_objectives: frozenset[str]
    required_modalities: frozenset[str] = frozenset()
    required_content_blocks: frozenset[str] = frozenset()
    requires_reference_model: bool = False
    requires_teacher_model: bool = False
    requires_reward_provider: bool = False
    requires_rollout: bool = False


@dataclass(frozen=True, slots=True)
class ObjectiveManifest:
    objective_id: str
    objective_version: str
    requirements: ObjectiveRequirements
    supported_engines: frozenset[str]
    api_version: str = OBJECTIVE_API_VERSION

    def __post_init__(self) -> None:
        if not _OBJECTIVE_ID.fullmatch(self.objective_id):
            raise ValueError(f"invalid objective_id {self.objective_id!r}")
        if not self.objective_version.strip():
            raise ValueError("objective_version must not be blank")
        if not self.supported_engines:
            raise ValueError("supported_engines must not be empty")
        if self.api_version != OBJECTIVE_API_VERSION:
            raise ValueError(
                f"unsupported objective API {self.api_version!r}; "
                f"expected {OBJECTIVE_API_VERSION!r}"
            )


@dataclass(frozen=True, slots=True)
class LossTerm:
    value: Any
    denominator: int | float

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("loss denominator must be positive")


@dataclass(frozen=True, slots=True)
class LossOutput:
    total: Any
    terms: Mapping[str, LossTerm]
    metrics: Mapping[str, Any]
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("LossOutput.terms must not be empty")


class Objective(Protocol):
    manifest: ObjectiveManifest

    def prepare(self, batch: Any, context: Any) -> Any: ...

    def compute(self, models: Any, batch: Any) -> LossOutput: ...
