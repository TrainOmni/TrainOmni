import pytest
import torch
from torch import nn

from trainomni.contracts.batch import OmniBatch
from trainomni.core.errors import SpecError
from trainomni.runtime.device.context import DeviceContext


def test_true_bf16_casts_model_and_floating_inputs_but_not_token_ids() -> None:
    context = DeviceContext("cpu", "bf16_true")
    model = nn.Linear(3, 2)
    context.prepare_model(model)
    batch = context.move_batch(
        OmniBatch(
            sample_ids=("one",),
            model_inputs={
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "pixel_values": torch.ones(1, 3, dtype=torch.float32),
            },
            labels=torch.tensor([[1, 2]], dtype=torch.long),
            supervision={
                "reference_logps": torch.ones(1, 2, dtype=torch.float32),
                "chosen_inputs": {
                    "pixel_values": torch.ones(1, 3, dtype=torch.float32)
                },
            },
        )
    )
    assert model.weight.dtype == torch.bfloat16
    assert batch.model_inputs["pixel_values"].dtype == torch.bfloat16
    assert batch.model_inputs["input_ids"].dtype == torch.long
    assert batch.labels.dtype == torch.long
    assert batch.supervision["reference_logps"].dtype == torch.float32
    assert batch.supervision["chosen_inputs"]["pixel_values"].dtype == torch.float32
    forwarded = context.move(batch.supervision["chosen_inputs"])
    assert forwarded["pixel_values"].dtype == torch.bfloat16


def test_cpu_fp16_mixed_fails_explicitly() -> None:
    with pytest.raises(SpecError, match="not supported"):
        DeviceContext("cpu", "fp16_mixed")


def test_cpu_bf16_mixed_autocast_keeps_parameters_fp32() -> None:
    context = DeviceContext("cpu", "bf16_mixed")
    model = nn.Linear(4, 3)
    context.prepare_model(model)
    with context.autocast():
        output = model(torch.ones(2, 4))
    assert model.weight.dtype == torch.float32
    assert output.dtype == torch.bfloat16
