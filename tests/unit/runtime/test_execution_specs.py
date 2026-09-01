from __future__ import annotations

from types import SimpleNamespace

import pytest

from trainomni.core.errors import SpecError
from trainomni.runtime.execution.deepspeed_backend import _config as deepspeed_config
from trainomni.runtime.execution.deepspeed_backend import build_deepspeed_backend
from trainomni.specs.run import CheckpointSpec, ExecutionSpec, RunSpec


def test_execution_spec_parses_each_backend_without_cross_backend_options() -> None:
    ddp = ExecutionSpec.from_mapping(
        {
            "backend": "torch_ddp",
            "expected_world_size": 1,
            "ddp": {"static_graph": True, "broadcast_buffers": False},
        }
    )
    assert ddp.backend == "torch_ddp"
    assert ddp.ddp.static_graph is True
    assert ddp.expected_world_size == 1

    fsdp = ExecutionSpec.from_mapping(
        {
            "backend": "torch_fsdp2",
            "fsdp2": {"wrap_policy": "root", "reshard_after_forward": False},
        }
    )
    assert fsdp.fsdp2.wrap_policy == "root"
    assert fsdp.fsdp2.reshard_after_forward is False

    deepspeed = ExecutionSpec.from_mapping(
        {
            "backend": "deepspeed",
            "deepspeed": {
                "zero_stage": 3,
                "offload_optimizer": "cpu",
                "offload_parameters": "cpu",
            },
        }
    )
    assert deepspeed.deepspeed.zero_stage == 3


@pytest.mark.parametrize(
    "value, message",
    [
        (
            {"backend": "single", "fsdp2": {"wrap_policy": "root"}},
            "inactive backends",
        ),
        (
            {
                "backend": "deepspeed",
                "deepspeed": {"zero_stage": 2, "offload_parameters": "cpu"},
            },
            "only valid with ZeRO stage 3",
        ),
        (
            {
                "backend": "torch_ddp",
                "ddp": {"static_graph": True, "find_unused_parameters": True},
            },
            "cannot be combined",
        ),
    ],
)
def test_execution_spec_rejects_ambiguous_or_invalid_combinations(value, message) -> None:
    with pytest.raises(SpecError, match=message):
        ExecutionSpec.from_mapping(value)


def test_checkpoint_can_be_explicitly_disabled_without_losing_output_root() -> None:
    checkpoint = CheckpointSpec.from_mapping(
        {"directory": "outputs/checkpoints", "every_steps": 8, "enabled": False}
    )

    assert checkpoint.directory.as_posix() == "outputs/checkpoints"
    assert checkpoint.every_steps == 8
    assert checkpoint.enabled is False

    with pytest.raises(SpecError, match="enabled must be a boolean"):
        CheckpointSpec.from_mapping(
            {"directory": "outputs/checkpoints", "enabled": "false"}
        )


def deepspeed_run(*, checkpoint_enabled: bool) -> RunSpec:
    return RunSpec.from_mapping(
        {
            "schema_version": 1,
            "name": "deepspeed-contract",
            "seed": 3,
            "device": "cuda",
            "precision": "bf16_true",
            "max_steps": 12,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 0.5,
            "optimizer": {"learning_rate": 1e-4},
            "execution": {
                "backend": "deepspeed",
                "expected_world_size": 8,
                "deepspeed": {
                    "zero_stage": 3,
                    "offload_optimizer": "cpu",
                    "offload_parameters": "cpu",
                },
            },
            "checkpoint": {
                "directory": "outputs/checkpoints",
                "enabled": checkpoint_enabled,
            },
        }
    )


def test_deepspeed_adapter_maps_run_contract_without_importing_upstream() -> None:
    config = deepspeed_config(deepspeed_run(checkpoint_enabled=False), SimpleNamespace(world_size=8))

    assert config["train_micro_batch_size_per_gpu"] == 2
    assert config["train_batch_size"] == 16
    assert config["gradient_accumulation_steps"] == 1
    assert config["gradient_clipping"] == 0.5
    assert config["bf16"] == {"enabled": True}
    assert config["zero_optimization"] == {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
    }


def test_deepspeed_checkpoint_gap_fails_before_import(monkeypatch) -> None:
    monkeypatch.setattr("trainomni.runtime.execution.deepspeed_backend.sys.platform", "linux")

    with pytest.raises(SpecError, match="checkpointing is not yet bridged"):
        build_deepspeed_backend(
            model=object(),
            selection=object(),
            run=deepspeed_run(checkpoint_enabled=True),
            process=SimpleNamespace(),
        )
