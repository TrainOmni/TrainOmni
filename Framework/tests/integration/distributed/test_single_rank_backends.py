from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.distribution import DistributionHints
from trainomni.core.errors import SpecError
from trainomni.modules.objectives.causal_lm.config import CausalLMConfig
from trainomni.modules.objectives.causal_lm.module import CausalLMObjective
from trainomni.modules.parameters.full.config import FullParameterConfig
from trainomni.modules.parameters.full.module import FullParameterPolicy
from trainomni.runtime.loop.engine import TrainEngine
from trainomni.specs.run import RunSpec


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 12)
        self.block0 = nn.Sequential(nn.Linear(12, 12), nn.GELU())
        self.block1 = nn.Sequential(nn.Linear(12, 12), nn.GELU())
        self.head = nn.Linear(12, 17)

    def distribution_hints(self):
        return DistributionHints(fsdp_units=("block0", "block1"))

    def forward(self, *, input_ids, **_):
        hidden = self.block1(self.block0(self.embedding(input_ids)))
        return SimpleNamespace(logits=self.head(hidden))


class Stream:
    def __init__(self) -> None:
        self.cursor = 0

    def next_batch(self, batch_size: int) -> OmniBatch:
        assert batch_size == 1
        offset = self.cursor % 3
        self.cursor += 1
        ids = torch.tensor([[1, 3 + offset, 7, 5, 2]], dtype=torch.long)
        labels = ids.clone()
        labels[:, :2] = -100
        return OmniBatch((f"sample-{self.cursor}",), {"input_ids": ids}, labels)

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])


def make_run(root: Path, backend: str, *, max_steps: int = 2) -> RunSpec:
    execution = {"backend": backend, "expected_world_size": 1}
    if backend == "torch_ddp":
        execution["ddp"] = {"static_graph": True}
    elif backend == "torch_fsdp2":
        execution["fsdp2"] = {"wrap_policy": "model_declared"}
    elif backend == "deepspeed":
        execution["deepspeed"] = {"zero_stage": 2}
    return RunSpec.from_mapping(
        {
            "schema_version": 1,
            "name": backend,
            "seed": 19,
            "deterministic": True,
            "device": "cuda",
            "precision": "bf16_true",
            "max_steps": max_steps,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
            "optimizer": {"learning_rate": 0.01, "foreach": False},
            "execution": execution,
            "update_evidence": {
                "enabled": True,
                "required_groups": ["all"],
                "sample_elements_per_group": 64,
            },
            "checkpoint": {"directory": str(root), "every_steps": 1},
        }
    )


def make_engine(root: Path, backend: str, *, max_steps: int = 2) -> TrainEngine:
    torch.manual_seed(23)
    model = TinyLM()
    selection = FullParameterPolicy(FullParameterConfig()).apply(model)
    return TrainEngine(
        model=model,
        objective=CausalLMObjective(CausalLMConfig()),
        parameter_selection=selection,
        stream=Stream(),
        run=make_run(root, backend, max_steps=max_steps),
        task_digest="d" * 64,
        module_lock={"model": "e" * 64},
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["torch_ddp", "torch_fsdp2"])
def test_single_gpu_distributed_backend_trains_checkpoints_and_resumes(
    tmp_path: Path, backend: str
) -> None:
    root = tmp_path / backend
    engine = make_engine(root, backend)
    try:
        first = engine.train(stop_after_steps=1)
        assert first[0].loss > 0
        assert first[0].parameter_evidence["all"]["changed_tensor_count"] > 0
        assert engine.execution.metadata()["world_size"] == 1
    finally:
        engine.close()
    checkpoint = root / "step-00000001"
    assert (checkpoint / "model.safetensors").is_file()

    resumed = make_engine(root, backend)
    try:
        resumed.resume(checkpoint)
        records = resumed.train()
        assert records[-1].global_step == 2
        assert torch.isfinite(torch.tensor(records[-1].loss))
    finally:
        resumed.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific fail-closed gate")
def test_deepspeed_is_rejected_before_upstream_import_on_native_windows(
    tmp_path: Path,
) -> None:
    with pytest.raises(SpecError, match="fail-closed on native Windows"):
        make_engine(tmp_path / "deepspeed", "deepspeed")
