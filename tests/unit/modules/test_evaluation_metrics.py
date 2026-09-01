from types import SimpleNamespace

import pytest
import torch

from trainomni.contracts.loss import ObjectiveMetric
from trainomni.core.errors import ObjectiveError
from trainomni.modules.evaluation.task_metrics.config import TaskMetricsConfig
from trainomni.modules.evaluation.task_metrics.module import TaskMetricsEvaluator


def test_objective_metrics_use_the_declared_weighted_mean() -> None:
    evaluator = TaskMetricsEvaluator(
        TaskMetricsConfig(metrics=("accuracy",), prefix="heldout/")
    )
    evaluator.update(
        SimpleNamespace(sample_ids=("a", "b")),
        SimpleNamespace(
            metrics={
                "accuracy": ObjectiveMetric.weighted_mean(
                    torch.tensor(1.0), torch.tensor(2)
                )
            }
        ),
    )
    evaluator.update(
        SimpleNamespace(sample_ids=("c",)),
        SimpleNamespace(
            metrics={
                "accuracy": ObjectiveMetric.weighted_mean(
                    torch.tensor(1.0), torch.tensor(1)
                )
            }
        ),
    )
    assert evaluator.compute() == {"heldout/accuracy": pytest.approx(2.0 / 3.0)}


def test_objective_metrics_fail_closed_before_partial_update() -> None:
    evaluator = TaskMetricsEvaluator(TaskMetricsConfig(metrics=("a", "b")))
    with pytest.raises(ObjectiveError, match="'b' is missing"):
        evaluator.update(
            SimpleNamespace(sample_ids=("x",)),
            SimpleNamespace(
                metrics={"a": ObjectiveMetric.sum(torch.tensor(1.0))}
            ),
        )
    assert evaluator.aggregations == {"a": None, "b": None}
    assert evaluator.numerators == {"a": 0.0, "b": 0.0}


def test_objective_metric_sum_is_not_divided_by_batches() -> None:
    evaluator = TaskMetricsEvaluator(
        TaskMetricsConfig(metrics=("supervised_tokens",), prefix="heldout/")
    )
    evaluator.update(
        SimpleNamespace(sample_ids=("a",)),
        SimpleNamespace(
            metrics={
                "supervised_tokens": ObjectiveMetric.sum(torch.tensor(3))
            }
        ),
    )
    evaluator.update(
        SimpleNamespace(sample_ids=("b", "c")),
        SimpleNamespace(
            metrics={
                "supervised_tokens": ObjectiveMetric.sum(torch.tensor(5))
            }
        ),
    )
    assert evaluator.compute() == {"heldout/supervised_tokens": 8.0}


def test_objective_metric_aggregation_cannot_change() -> None:
    evaluator = TaskMetricsEvaluator(TaskMetricsConfig(metrics=("metric",)))
    evaluator.update(
        SimpleNamespace(sample_ids=("a",)),
        SimpleNamespace(
            metrics={"metric": ObjectiveMetric.sum(torch.tensor(1.0))}
        ),
    )
    with pytest.raises(ObjectiveError, match="changed aggregation semantics"):
        evaluator.update(
            SimpleNamespace(sample_ids=("b",)),
            SimpleNamespace(
                metrics={
                    "metric": ObjectiveMetric.weighted_mean(
                        torch.tensor(1.0), torch.tensor(1.0)
                    )
                }
            ),
        )
