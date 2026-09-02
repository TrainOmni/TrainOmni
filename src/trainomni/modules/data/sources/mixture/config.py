"""Deterministic source-mixture configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType


@dataclass(frozen=True, slots=True, kw_only=True)
class MixtureSourceConfig:
    weights: Mapping[str, float]
    seed: int = 0
    namespace_sample_ids: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("mixture seed must be an integer")
        if not isinstance(self.namespace_sample_ids, bool):
            raise TypeError("namespace_sample_ids must be a boolean")
        if not isinstance(self.weights, Mapping):
            raise TypeError("mixture weights must be a mapping")
        weights = {}
        for name, weight in self.weights.items():
            if not isinstance(name, str):
                raise TypeError("mixture source names must be strings")
            if not isinstance(weight, Real) or isinstance(weight, bool):
                raise TypeError(f"mixture weight for {name!r} must be numeric")
            weights[name] = float(weight)
        if not weights:
            raise ValueError("mixture requires at least one source weight")
        if any(not name or name.startswith("__") for name in weights):
            raise ValueError(
                "mixture source names must be non-empty and cannot start with '__'"
            )
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
            raise ValueError("mixture source weights must be finite and positive")
        object.__setattr__(
            self,
            "weights",
            MappingProxyType(dict(sorted(weights.items()))),
        )
