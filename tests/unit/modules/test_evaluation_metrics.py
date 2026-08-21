from types import SimpleNamespace

import pytest
import torch

from trainomni.core.errors import ObjectiveError
from trainomni.modules.evaluation.task_metrics.config import TaskMetricsConfig
from trainomni.modules.evaluation.task_metrics.module import TaskMetricsEvaluator


def test_objective_metrics_are_sample_weighted() -> None:
    evaluator = TaskMetricsEvaluator(
        TaskMetricsConfig(metrics=("accuracy",), prefix="heldout/")
    )
    evaluator.update(
        SimpleNamespace(sample_ids=("a", "b")),
        SimpleNamespace(metrics={"accuracy": torch.tensor(0.5)}),
    )
    evaluator.update(
        SimpleNamespace(sample_ids=("c",)),
        SimpleNamespace(metrics={"accuracy": torch.tensor(1.0)}),
    )
    assert evaluator.compute() == {"heldout/accuracy": pytest.approx(2.0 / 3.0)}


def test_objective_metrics_fail_closed_before_partial_update() -> None:
    evaluator = TaskMetricsEvaluator(TaskMetricsConfig(metrics=("a", "b")))
    with pytest.raises(ObjectiveError, match="'b' is missing"):
        evaluator.update(
            SimpleNamespace(sample_ids=("x",)),
            SimpleNamespace(metrics={"a": torch.tensor(1.0)}),
        )
    assert evaluator.denominator == 0
    assert evaluator.totals == {"a": 0.0, "b": 0.0}
