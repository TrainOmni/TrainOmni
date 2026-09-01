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
    metric_aggregations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.metric_aggregations]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "objective metric_aggregations must contain unique non-empty names"
            )
        unknown = sorted(
            {
                aggregation
                for _, aggregation in self.metric_aggregations
                if aggregation not in {"sum", "weighted_mean"}
            }
        )
        if unknown:
            raise ValueError(
                "objective metric_aggregations contains unknown semantics: "
                + ", ".join(unknown)
            )


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
