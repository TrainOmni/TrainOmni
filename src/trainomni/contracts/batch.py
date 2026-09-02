"""Final batch contract consumed by objectives and runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._identity import normalize_identity
from ._mapping import FrozenDict


def _pin_memory(value: Any) -> Any:
    if hasattr(value, "pin_memory") and callable(value.pin_memory):
        return value.pin_memory()
    if isinstance(value, Mapping):
        return {key: _pin_memory(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return tuple(_pin_memory(inner) for inner in value)
    if isinstance(value, list):
        return [_pin_memory(inner) for inner in value]
    return value


@dataclass(frozen=True, slots=True)
class EncodedSample:
    sample_id: str
    model_inputs: Mapping[str, Any]
    supervision: Mapping[str, Any] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_id",
            normalize_identity(self.sample_id, field="encoded sample_id"),
        )
        if not self.model_inputs:
            raise ValueError("encoded model_inputs must not be empty")
        object.__setattr__(self, "model_inputs", FrozenDict(self.model_inputs))
        object.__setattr__(self, "supervision", FrozenDict(self.supervision))


@dataclass(frozen=True, slots=True)
class SupervisedExample:
    sample_id: str
    model_inputs: Mapping[str, Any]
    labels: Any
    supervision: Mapping[str, Any] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_id",
            normalize_identity(self.sample_id, field="supervised sample_id"),
        )
        if not self.model_inputs:
            raise ValueError("supervised model_inputs must not be empty")
        object.__setattr__(self, "model_inputs", FrozenDict(self.model_inputs))
        object.__setattr__(self, "supervision", FrozenDict(self.supervision))


@dataclass(frozen=True, slots=True)
class OmniBatch:
    sample_ids: tuple[str, ...]
    model_inputs: Mapping[str, Any]
    labels: Any
    supervision: Mapping[str, Any] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_ids, (tuple, list)) or not self.sample_ids:
            raise ValueError("batch must contain at least one sample")
        sample_ids = tuple(
            normalize_identity(sample_id, field="batch sample_id")
            for sample_id in self.sample_ids
        )
        if not self.model_inputs:
            raise ValueError("batch.model_inputs must not be empty")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "model_inputs", FrozenDict(self.model_inputs))
        object.__setattr__(self, "supervision", FrozenDict(self.supervision))

    def pin_memory(self) -> OmniBatch:
        return OmniBatch(
            sample_ids=self.sample_ids,
            model_inputs=_pin_memory(self.model_inputs),
            labels=_pin_memory(self.labels),
            supervision=_pin_memory(self.supervision),
        )
