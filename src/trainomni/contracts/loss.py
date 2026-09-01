"""Loss and metric contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class LossTerm:
    value: Any
    weight: float
    numerator: Any
    denominator: Any


@dataclass(frozen=True, slots=True)
class ObjectiveMetric:
    """One metric with explicit effective-batch aggregation semantics."""

    aggregation: str
    numerator: Any
    denominator: Any | None = None

    def __post_init__(self) -> None:
        if self.aggregation not in {"sum", "weighted_mean"}:
            raise ValueError(
                "objective metric aggregation must be sum or weighted_mean"
            )
        if self.aggregation == "sum" and self.denominator is not None:
            raise ValueError("sum objective metrics cannot declare a denominator")
        if self.aggregation == "weighted_mean" and self.denominator is None:
            raise ValueError(
                "weighted_mean objective metrics require a denominator"
            )

    @classmethod
    def sum(cls, value: Any) -> ObjectiveMetric:
        return cls("sum", value)

    @classmethod
    def weighted_mean(
        cls, numerator: Any, denominator: Any
    ) -> ObjectiveMetric:
        return cls("weighted_mean", numerator, denominator)


@dataclass(frozen=True, slots=True)
class LossBundle:
    total: Any
    terms: Mapping[str, LossTerm]
    metrics: Mapping[str, ObjectiveMetric] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("loss bundle must contain at least one named term")
