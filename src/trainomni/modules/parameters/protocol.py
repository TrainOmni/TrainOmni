"""Parameter-policy contract.

A policy owns which parameters are trainable and how they are grouped. It does
not construct the optimizer and never receives the run specification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ParameterGroup:
    name: str
    parameters: tuple[Any, ...]
    options: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ParameterSelection:
    groups: tuple[ParameterGroup, ...]
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]

    @property
    def trainable_numel(self) -> int:
        return sum(parameter.numel() for group in self.groups for parameter in group.parameters)


class ParameterPolicy(Protocol):
    def apply(self, model: Any) -> ParameterSelection: ...
