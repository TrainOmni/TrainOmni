"""Final batch contract consumed by objectives and runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class EncodedSample:
    sample_id: str
    model_inputs: Mapping[str, Any]
    supervision: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("encoded sample_id must not be empty")
        if not self.model_inputs:
            raise ValueError("encoded model_inputs must not be empty")


@dataclass(frozen=True, slots=True)
class SupervisedExample:
    sample_id: str
    model_inputs: Mapping[str, Any]
    labels: Any
    supervision: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class OmniBatch:
    sample_ids: tuple[str, ...]
    model_inputs: Mapping[str, Any]
    labels: Any
    supervision: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ValueError("batch must contain at least one sample")
        if not self.model_inputs:
            raise ValueError("batch.model_inputs must not be empty")
