import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.features import ModalFeatures
from trainomni.core.errors import CheckpointError
from trainomni.modules.model.models.composite.module import CompositeModel
from trainomni.modules.objectives.causal_lm.config import CausalLMConfig
from trainomni.modules.objectives.causal_lm.module import CausalLMObjective
from trainomni.modules.parameters.full.config import FullParameterConfig
from trainomni.modules.parameters.full.module import FullParameterPolicy
from trainomni.runtime.loop.engine import TrainEngine
from trainomni.runtime.optimization.optimizer import build_optimizer
from trainomni.runtime.optimization.scheduler import build_scheduler
from trainomni.specs.run import RunSpec


class TinyEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, hidden)

    def forward(self, pixels: torch.Tensor) -> ModalFeatures:
        return ModalFeatures(self.projection(pixels).unsqueeze(1))


class TinyConnector(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden, hidden)

    def forward(self, features: ModalFeatures) -> ModalFeatures:
        return ModalFeatures(self.projection(features.embeddings))


class TinyLanguage(nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, vocab)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward_embeddings(self, embeddings: torch.Tensor, **kwargs):
        del kwargs
        return SimpleNamespace(logits=self.head(self.dropout(embeddings)))


class PrefixFusion(nn.Module):
    def forward(
        self,
        *,
        language,
        input_ids,
        modal_features,
        attention_mask=None,
        modal_positions=None,
        **kwargs,
    ):
        del modal_positions
        modal_features = modal_features.concatenate()
        embeddings = language.embed(input_ids)
        embeddings = torch.cat(
            (embeddings[:, :1] + modal_features.embeddings[:, :1], embeddings[:, 1:]),
            dim=1,
        )
        return language.forward_embeddings(
            embeddings, attention_mask=attention_mask, **kwargs
        )


class DeterministicStream:
    def __init__(self) -> None:
        self.cursor = 0
        self._examples = (
            ([1, 2, 3, 4], [1, 2, 3, 4], [0.2, -0.1, 0.3]),
            ([2, 1, 4, 3], [2, 1, 4, 3], [-0.4, 0.5, 0.1]),
            ([3, 4, 1, 2], [3, 4, 1, 2], [0.7, 0.1, -0.2]),
        )

    def next_batch(self, batch_size: int) -> OmniBatch:
        assert batch_size == 1
        index = self.cursor % len(self._examples)
        input_ids, labels, pixels = self._examples[index]
        self.cursor += 1
        return OmniBatch(
            sample_ids=(f"sample-{index}",),
            model_inputs={
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "pixel_values": torch.tensor([pixels], dtype=torch.float32),
            },
            labels=torch.tensor([labels], dtype=torch.long),
        )

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        if set(state) != {"cursor"}:
            raise ValueError("invalid stream state")
        self.cursor = int(state["cursor"])


def make_run(checkpoint_root: Path) -> RunSpec:
    return RunSpec.from_mapping(
        {
            "schema_version": 1,
            "name": "tiny-composite",
            "seed": 123,
            "device": "cpu",
            "precision": "fp32",
            "max_steps": 4,
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
            "optimizer": {
                "name": "adamw",
                "learning_rate": 0.01,
                "foreach": False,
            },
            "update_evidence": {
                "enabled": True,
                "required_groups": ["all"],
                "sample_elements_per_group": 32,
            },
            "checkpoint": {"directory": str(checkpoint_root), "every_steps": 2},
        }
    )


def make_engine(checkpoint_root: Path) -> TrainEngine:
    torch.manual_seed(123)
    hidden = 8
    model = CompositeModel(
        encoder=TinyEncoder(hidden),
        connector=TinyConnector(hidden),
        fusion=PrefixFusion(),
        language=TinyLanguage(vocab=11, hidden=hidden),
    )
    selection = FullParameterPolicy(FullParameterConfig()).apply(model)
    run = make_run(checkpoint_root)
    optimizer = build_optimizer(run.optimizer, selection)
    scheduler = build_scheduler(run.scheduler, optimizer, total_steps=run.max_steps)
    return TrainEngine(
        model=model,
        objective=CausalLMObjective(CausalLMConfig()),
        optimizer=optimizer,
        scheduler=scheduler,
        stream=DeterministicStream(),
        run=run,
        task_digest="tiny-task-digest",
        module_lock={"objective": "causal-lm-v1", "model": "tiny-composite-v1"},
    )


def assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_tiny_composite_train_checkpoint_and_exact_resume(tmp_path: Path) -> None:
    uninterrupted = make_engine(tmp_path / "uninterrupted")
    uninterrupted_records = uninterrupted.train()

    split = make_engine(tmp_path / "resumed")
    split.train(stop_after_steps=2)
    checkpoint = tmp_path / "resumed" / "step-00000002"
    assert (checkpoint / "manifest.json").is_file()
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "runtime.pt").is_file()
    manifest = json.loads(
        (checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    evidence = manifest["runtime_metadata"]["parameter_evidence"]["all"]
    assert evidence["changed_tensor_count"] > 1
    assert evidence["before_sha256"] != evidence["after_sha256"]
    optimizer_identity = manifest["runtime_metadata"]["optimizer"]
    assert optimizer_identity["name"] == "adamw"
    assert optimizer_identity["foreach"] is False
    assert optimizer_identity["quantized"] is False
    assert optimizer_identity["groups"][0]["name"] == "all"
    assert "torch.float32" in optimizer_identity["groups"][0]["state_dtypes"]

    resumed = make_engine(tmp_path / "resumed")
    resumed.resume(checkpoint)
    resumed_records = resumed.train()

    assert [record.global_step for record in resumed_records] == [3, 4]
    assert [record.loss for record in resumed_records] == pytest.approx(
        [record.loss for record in uninterrupted_records[2:]], rel=0, abs=0
    )
    assert [record.parameter_evidence for record in resumed_records] == [
        record.parameter_evidence for record in uninterrupted_records[2:]
    ]
    assert resumed.stream.cursor == uninterrupted.stream.cursor == 8
    assert_nested_equal(uninterrupted.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(
        uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    )
    assert_nested_equal(
        uninterrupted.scheduler.state_dict(), resumed.scheduler.state_dict()
    )


def test_checkpoint_identity_and_integrity_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "guarded"
    engine = make_engine(root)
    engine.train(stop_after_steps=1)
    checkpoint = root / "step-00000001"

    wrong_identity = make_engine(root)
    wrong_identity.checkpoints.task_digest = "different-task"
    with pytest.raises(CheckpointError, match="task_digest"):
        wrong_identity.resume(checkpoint)

    with (checkpoint / "model.safetensors").open("ab") as stream:
        stream.write(b"tampered")
    corrupted = make_engine(root)
    with pytest.raises(CheckpointError, match="digest mismatch"):
        corrupted.resume(checkpoint)
