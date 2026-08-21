"""Aggregate named scalar Objective metrics over held-out batches."""

from __future__ import annotations

import math

import torch

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import ObjectiveError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import TaskMetricsConfig


class TaskMetricsEvaluator:
    def __init__(self, config: TaskMetricsConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.totals = {name: 0.0 for name in self.config.metrics}
        self.denominator = 0.0

    def update(self, batch, loss) -> None:
        weight = float(len(batch.sample_ids) if self.config.weighting == "samples" else 1)
        values = {}
        for name in self.config.metrics:
            if name not in loss.metrics:
                raise ObjectiveError(f"objective metric {name!r} is missing")
            value = loss.metrics[name]
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ObjectiveError(f"objective metric {name!r} is not scalar")
                value = float(value.detach().float().item())
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                raise ObjectiveError(f"objective metric {name!r} is not numeric")
            if not math.isfinite(value):
                raise ObjectiveError(f"objective metric {name!r} is non-finite")
            values[name] = value
        for name, value in values.items():
            self.totals[name] += value * weight
        self.denominator += weight

    def compute(self):
        if self.denominator <= 0:
            raise ObjectiveError("task-metrics evaluator has no observations")
        return {
            f"{self.config.prefix}{name}": total / self.denominator
            for name, total in self.totals.items()
        }


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("evaluator:trainomni/objective_metrics@1"),
        config_type=TaskMetricsConfig,
        factory=lambda config, context: TaskMetricsEvaluator(config),
        provides=CapabilitySet.of({"evaluation.objective_metrics"}),
        requires=CapabilitySet.of({"objective.loss_bundle"}),
    )
