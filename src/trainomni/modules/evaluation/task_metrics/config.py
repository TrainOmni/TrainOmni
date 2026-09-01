"""Objective scalar-metric evaluator configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskMetricsConfig:
    metrics: tuple[str, ...]
    prefix: str = "eval/"
    weighting: str = "objective"

    def __post_init__(self) -> None:
        if not self.metrics or any(not name for name in self.metrics):
            raise ValueError("metrics must contain non-empty names")
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("metrics contains duplicates")
        if self.weighting != "objective":
            raise ValueError(
                "weighting must be objective; metric aggregation is declared by "
                "ObjectiveMetric"
            )
