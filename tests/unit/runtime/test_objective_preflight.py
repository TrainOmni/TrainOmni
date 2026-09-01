import pytest
import torch
from torch import nn

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.forward import ForwardPlan, ForwardRequest, OutputRequirements
from trainomni.contracts.loss import LossBundle, LossTerm
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import ObjectiveError
from trainomni.modules.objectives.dense_kd.config import DenseKDConfig
from trainomni.modules.objectives.dense_kd.module import DenseKDObjective
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.loop.step import execute_forward_plan


class CountingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, **inputs):
        self.calls += 1
        return {"logits": torch.zeros(1, 3, 4)}


def test_missing_objective_supervision_fails_before_model_forward() -> None:
    model = CountingModel()
    batch = OmniBatch(
        sample_ids=("missing-cache",),
        model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        labels=torch.tensor([[1, 2, 3]]),
    )
    with pytest.raises(ObjectiveError, match="before forward"):
        execute_forward_plan(
            model=model,
            objective=DenseKDObjective(
                DenseKDConfig(producer_identity_sha256="a" * 64)
            ),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert model.calls == 0


def test_raw_scalar_objective_metric_fails_closed() -> None:
    class RawMetricObjective:
        def requirements(self):
            return type(
                "Requirements",
                (),
                {
                    "outputs": OutputRequirements(logits=True),
                    "supervision_fields": frozenset(),
                    "metric_aggregations": (("ambiguous", "weighted_mean"),),
                },
            )()

        def plan(self, batch, context):
            del context
            return ForwardPlan.single(
                ForwardRequest(
                    "policy",
                    batch.model_inputs,
                    OutputRequirements(logits=True),
                    requires_grad=True,
                )
            )

        def compute(self, batch, outputs, context):
            del batch, context
            total = outputs["policy"].require("logits").sum()
            one = torch.tensor(1)
            return LossBundle(
                total=total,
                terms={"loss": LossTerm(total, 1.0, total, one)},
                metrics={"ambiguous": torch.tensor(0.5)},
            )

    model = CountingModel()
    batch = OmniBatch(
        sample_ids=("raw-metric",),
        model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        labels=torch.tensor([[1, 2, 3]]),
    )
    with pytest.raises(ObjectiveError, match="aggregation semantics"):
        execute_forward_plan(
            model=model,
            objective=RawMetricObjective(),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
