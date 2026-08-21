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
class LossBundle:
    total: Any
    terms: Mapping[str, LossTerm]
    metrics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("loss bundle must contain at least one named term")
