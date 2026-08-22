from pathlib import Path

import torch
from torch import nn

from trainomni.contracts.batch import OmniBatch
from trainomni.modules.objectives.dense_kd.config import DenseKDConfig
from trainomni.modules.objectives.dense_kd.module import DenseKDObjective
from trainomni.modules.objectives.dpo.config import DPOConfig
from trainomni.modules.objectives.dpo.module import DPOObjective
from trainomni.modules.parameters.protocol import ParameterGroup, ParameterSelection
from trainomni.runtime.loop.engine import TrainEngine
from trainomni.specs.run import RunSpec


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 5)
        self.head = nn.Linear(5, 7)

    def forward(self, input_ids):
        return {"logits": self.head(self.embedding(input_ids))}


class RepeatingStream:
    def __init__(self, batch: OmniBatch) -> None:
        self.batch = batch
        self.cursor = 0

    def next_batch(self, batch_size: int):
        assert batch_size == len(self.batch.sample_ids)
        self.cursor += 1
        return self.batch

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        assert set(state) == {"cursor"}
        self.cursor = int(state["cursor"])


def kd_batch() -> OmniBatch:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    return OmniBatch(
        sample_ids=("kd",),
        model_inputs={"input_ids": input_ids},
        labels=torch.tensor([[-100, 2, 3, 4]]),
        supervision={"teacher_logits": torch.randn(1, 3, 7, dtype=torch.bfloat16)},
    )


def dpo_batch() -> OmniBatch:
    chosen = torch.tensor([[1, 2, 3, 4]])
    rejected = torch.tensor([[1, 2, 4, 3]])
    return OmniBatch(
        sample_ids=("pair",),
        model_inputs={"input_ids": chosen},
        labels=chosen,
        supervision={
            "chosen_inputs": {"input_ids": chosen},
            "rejected_inputs": {"input_ids": rejected},
            "chosen_labels": torch.tensor([[-100, 2, 3, 4]]),
            "rejected_labels": torch.tensor([[-100, 2, 4, 3]]),
            "chosen_reference_logps": torch.tensor([[-0.4, -0.3, -0.2]]),
            "rejected_reference_logps": torch.tensor([[-0.5, -0.6, -0.4]]),
        },
    )


def run_spec(root: Path, name: str) -> RunSpec:
    return RunSpec.from_mapping(
        {
            "schema_version": 1,
            "name": name,
            "seed": 17,
            "device": "cpu",
            "precision": "fp32",
            "max_steps": 2,
            "optimizer": {"learning_rate": 0.02, "foreach": False},
            "compile": (
                {"enabled": True, "backend": "eager"} if name == "kd" else {}
            ),
            "checkpoint": {"directory": str(root), "every_steps": 1},
        }
    )


def make_engine(root: Path, name: str, objective, batch: OmniBatch) -> TrainEngine:
    model = TinyPolicy()
    selection = ParameterSelection(
        groups=(ParameterGroup("policy", tuple(model.parameters()), {}),),
        trainable_names=tuple(name for name, _ in model.named_parameters()),
        frozen_names=(),
    )
    run = run_spec(root, name)
    return TrainEngine(
        model=model,
        objective=objective,
        parameter_selection=selection,
        stream=RepeatingStream(batch),
        run=run,
        task_digest=("a" if name == "kd" else "b") * 64,
        module_lock={"objective": ("c" if name == "kd" else "d") * 64},
    )


def test_dense_kd_and_dpo_use_the_same_engine_checkpoint_contract(tmp_path: Path) -> None:
    cases = (
        ("kd", DenseKDObjective(DenseKDConfig()), kd_batch()),
        ("dpo", DPOObjective(DPOConfig()), dpo_batch()),
    )
    for name, objective, batch in cases:
        torch.manual_seed(3)
        engine = make_engine(tmp_path / name, name, objective, batch)
        before = {key: value.detach().clone() for key, value in engine.model.state_dict().items()}
        records = engine.train()
        assert [record.global_step for record in records] == [1, 2]
        assert all(torch.isfinite(torch.tensor(record.loss)) for record in records)
        assert any(
            not torch.equal(before[key], value)
            for key, value in engine.model.state_dict().items()
        )
        assert (tmp_path / name / "step-00000002" / "model.safetensors").is_file()
        if name == "kd":
            from safetensors.torch import load_file

            keys = load_file(
                tmp_path / name / "step-00000002" / "model.safetensors"
            )
            assert all(not key.startswith("_orig_mod.") for key in keys)


def test_dpo_exact_resume_matches_uninterrupted(tmp_path: Path) -> None:
    torch.manual_seed(13)
    uninterrupted = make_engine(
        tmp_path / "uninterrupted", "dpo", DPOObjective(DPOConfig()), dpo_batch()
    )
    uninterrupted.train()

    torch.manual_seed(13)
    first = make_engine(
        tmp_path / "resumed", "dpo", DPOObjective(DPOConfig()), dpo_batch()
    )
    first.train(stop_after_steps=1)
    checkpoint = tmp_path / "resumed" / "step-00000001"

    torch.manual_seed(999)
    resumed = make_engine(
        tmp_path / "resumed", "dpo", DPOObjective(DPOConfig()), dpo_batch()
    )
    resumed.resume(checkpoint)
    resumed.train()

    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], expected), name
    assert resumed.stream.cursor == uninterrupted.stream.cursor == 2
