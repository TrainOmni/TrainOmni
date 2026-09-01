"""Objective-declared model forward plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trainomni.core.errors import ObjectiveError


@dataclass(frozen=True, slots=True)
class OutputRequirements:
    logits: bool = True
    hidden_states: bool = False
    attentions: bool = False
    modal_features: bool = False


@dataclass(frozen=True, slots=True)
class ForwardRequest:
    name: str
    inputs: Mapping[str, Any]
    outputs: OutputRequirements = OutputRequirements()
    requires_grad: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("forward request name must not be empty")


@dataclass(frozen=True, slots=True)
class ForwardPlan:
    requests: tuple[ForwardRequest, ...]

    def __post_init__(self) -> None:
        names = [request.name for request in self.requests]
        if not names:
            raise ValueError("forward plan must contain at least one request")
        if len(names) != len(set(names)):
            raise ValueError("forward request names must be unique")

    @classmethod
    def single(cls, request: ForwardRequest) -> ForwardPlan:
        return cls((request,))


@dataclass(frozen=True, slots=True)
class ForwardResult:
    name: str
    output: Any

    def require(self, key: str) -> Any:
        if isinstance(self.output, Mapping):
            if key not in self.output:
                raise ObjectiveError(f"forward {self.name!r} did not return {key!r}")
            return self.output[key]
        if not hasattr(self.output, key):
            raise ObjectiveError(f"forward {self.name!r} did not expose {key!r}")
        return getattr(self.output, key)
