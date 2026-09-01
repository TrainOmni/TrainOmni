"""Aggregate named scalar Objective metrics over held-out batches."""

from __future__ import annotations

import math

import torch

from trainomni.contracts.loss import ObjectiveMetric
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import ObjectiveError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import TaskMetricsConfig


class TaskMetricsEvaluator:
    def __init__(self, config: TaskMetricsConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.aggregations = {name: None for name in self.config.metrics}
        self.numerators = {name: 0.0 for name in self.config.metrics}
        self.denominators = {name: 0.0 for name in self.config.metrics}

    def update(self, batch, loss) -> None:
        del batch
        values = {}
        for name in self.config.metrics:
            if name not in loss.metrics:
                raise ObjectiveError(f"objective metric {name!r} is missing")
            metric = loss.metrics[name]
            if not isinstance(metric, ObjectiveMetric):
                raise ObjectiveError(
                    f"objective metric {name!r} has no aggregation contract"
                )
            raw_values = [metric.numerator]
            if metric.denominator is not None:
                raw_values.append(metric.denominator)
            if any(
                not isinstance(value, torch.Tensor)
                or value.ndim != 0
                or not math.isfinite(float(value.detach().float().item()))
                for value in raw_values
            ):
                raise ObjectiveError(
                    f"objective metric {name!r} contains invalid scalar state"
                )
            numerator = float(metric.numerator.detach().float().item())
            denominator = (
                0.0
                if metric.denominator is None
                else float(metric.denominator.detach().float().item())
            )
            if metric.denominator is not None and denominator <= 0:
                raise ObjectiveError(
                    f"objective metric {name!r} denominator is not positive"
                )
            values[name] = (metric.aggregation, numerator, denominator)
        for name, (aggregation, numerator, denominator) in values.items():
            observed = self.aggregations[name]
            if observed is not None and observed != aggregation:
                raise ObjectiveError(
                    f"objective metric {name!r} changed aggregation semantics"
                )
            self.aggregations[name] = aggregation
            self.numerators[name] += numerator
            self.denominators[name] += denominator

    def compute(self):
        if any(value is None for value in self.aggregations.values()):
            raise ObjectiveError("task-metrics evaluator has no observations")
        result = {}
        for name, aggregation in self.aggregations.items():
            value = self.numerators[name]
            if aggregation == "weighted_mean":
                denominator = self.denominators[name]
                if denominator <= 0:
                    raise ObjectiveError(
                        f"objective metric {name!r} denominator is not positive"
                    )
                value /= denominator
            result[f"{self.config.prefix}{name}"] = value
        return result


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("evaluator:trainomni/objective_metrics@1"),
        config_type=TaskMetricsConfig,
        factory=lambda config, context: TaskMetricsEvaluator(config),
        provides=CapabilitySet.of({"evaluation.objective_metrics"}),
        requires=CapabilitySet.of({"objective.loss_bundle"}),
    )
