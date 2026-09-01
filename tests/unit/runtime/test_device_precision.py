from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trainomni.contracts.batch import OmniBatch
from trainomni.core.errors import SpecError
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.execution.process import ProcessContext
from trainomni.specs.run import ExecutionSpec


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


def test_distributed_local_device_is_bound_before_process_group_init(
    monkeypatch,
) -> None:
    events = []
    for name, value in {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29599",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_gloo_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda *args, **kwargs: events.append(("init", args, kwargs)),
    )
    monkeypatch.setattr(
        ProcessContext,
        "_bind_local_device",
        staticmethod(lambda requested, *, local_rank: events.append(("bind", requested, local_rank))),
    )

    context = ProcessContext.create(
        ExecutionSpec.from_mapping(
            {
                "backend": "torch_ddp",
                "process_group_backend": "gloo",
                "expected_world_size": 1,
            }
        ),
        requested_device="cuda:0",
    )

    assert [event[0] for event in events] == ["bind", "init"]
    assert context.local_rank == 0


def test_cuda_and_npu_device_binding_are_explicit(monkeypatch) -> None:
    cuda_calls = []
    monkeypatch.setattr(torch.cuda, "set_device", cuda_calls.append)
    ProcessContext._bind_local_device("cuda:2", local_rank=2)
    assert cuda_calls == [2]
    with pytest.raises(SpecError, match="disagrees with LOCAL_RANK"):
        ProcessContext._bind_local_device("cuda:1", local_rank=2)

    npu_calls = []
    original_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda value: (
            SimpleNamespace(type="npu", index=None)
            if value == "npu"
            else original_device(value)
        ),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(set_device=npu_calls.append),
        raising=False,
    )
    ProcessContext._bind_local_device("npu", local_rank=3)
    assert npu_calls == [3]
