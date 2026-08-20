"""Evaluator contracts independent of the training engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    evaluator_id: str
    evaluator_version: str
    modalities: frozenset[str]
    requires_generation: bool = False
    delegated: bool = False

    def __post_init__(self) -> None:
        if not self.evaluator_id.strip() or not self.evaluator_version.strip():
            raise ValueError("evaluator identity/version must not be blank")


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    run_name: str
    model_bundle: Any
    batches: Any
    objective: Any | None
    output_dir: Any
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    evaluator_id: str
    metrics: Mapping[str, float]
    counts: Mapping[str, int] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))


class Evaluator(Protocol):
    manifest: EvaluationManifest

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...
