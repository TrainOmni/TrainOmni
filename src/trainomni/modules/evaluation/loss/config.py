"""Loss evaluator configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class LossEvaluatorConfig:
    term: str = "token_ce"
    metric_name: str = "loss"

    def __post_init__(self) -> None:
        if not self.term or not self.metric_name:
            raise ValueError("term and metric_name must not be empty")
