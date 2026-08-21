"""Objective module protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.forward import ForwardPlan, ForwardResult, OutputRequirements
from trainomni.contracts.loss import LossBundle
from trainomni.core.context import ObjectiveContext


@dataclass(frozen=True, slots=True)
class ObjectiveRequirements:
    outputs: OutputRequirements = field(default_factory=OutputRequirements)
    supervision_fields: frozenset[str] = frozenset()


class ObjectiveModule(Protocol):
    def requirements(self) -> ObjectiveRequirements: ...

    def plan(self, batch: OmniBatch, context: ObjectiveContext) -> ForwardPlan: ...

    def compute(
        self,
        batch: OmniBatch,
        outputs: Mapping[str, ForwardResult],
        context: ObjectiveContext,
    ) -> LossBundle: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
