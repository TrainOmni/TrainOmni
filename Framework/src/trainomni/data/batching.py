"""Deterministic multi-budget batch planning."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from trainomni.contracts import BatchBudget, BatchItem, BatchPlan, CostVector

if TYPE_CHECKING:
    from trainomni.models.io import EncodedSample


class BatchPlanningError(ValueError):
    pass


class GreedyBatchPlanner:
    def __init__(self, budget: BatchBudget, *, packing: bool = False) -> None:
        self.budget = budget
        self.packing = packing

    def plan(self, samples: Iterable[EncodedSample]) -> tuple[BatchPlan, ...]:
        batches: list[BatchPlan] = []
        current: list[EncodedSample] = []
        cost = CostVector()
        for sample in samples:
            single_exceeded = self.budget.exceeded_by(sample.cost, 1)
            if single_exceeded:
                raise BatchPlanningError(
                    f"sample {sample.sample_id!r} exceeds batch budget fields: "
                    f"{list(single_exceeded)}"
                )
            candidate_cost = cost + sample.cost
            exceeded = self.budget.exceeded_by(candidate_cost, len(current) + 1)
            if current and exceeded:
                batches.append(self._freeze(current, cost))
                current = []
                cost = CostVector()
            current.append(sample)
            cost = cost + sample.cost
        if current:
            batches.append(self._freeze(current, cost))
        return tuple(batches)

    def _freeze(
        self, samples: list[EncodedSample], total_cost: CostVector
    ) -> BatchPlan:
        return BatchPlan(
            items=tuple(
                BatchItem(
                    sample_id=sample.sample_id,
                    segment_id=index if self.packing else 0,
                    cost=sample.cost,
                )
                for index, sample in enumerate(samples)
            ),
            total_cost=total_cost,
            budget=self.budget,
            packing=self.packing,
        )
