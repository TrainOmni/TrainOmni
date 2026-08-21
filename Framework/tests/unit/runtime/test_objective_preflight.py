import pytest
import torch
from torch import nn

from trainomni.contracts.batch import OmniBatch
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
            objective=DenseKDObjective(DenseKDConfig()),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert model.calls == 0
