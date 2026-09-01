from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from trainomni.assembly.task_builder import build_task
from trainomni.catalog.builtin import builtin_registry
from trainomni.contracts.batch import EncodedSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.core.resolver import ModuleResolver
from trainomni.runtime.evaluation import evaluate_batches
from trainomni.runtime.loop.engine import TrainEngine
from trainomni.specs.run import RunSpec
from trainomni.specs.task import TaskSpec


@dataclass(frozen=True)
class EmptyConfig:
    pass


class TensorModelIO:
    def encode(self, sample):
        blocks = {block.kind: block.value for block in sample.content}
        return EncodedSample(
            sample.sample_id,
            {
                "input_ids": torch.tensor(blocks["text"], dtype=torch.long),
                "pixel_values": torch.tensor(blocks["image"], dtype=torch.float32),
            },
        )


class TinyMonolithicVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(9, 6)
        self.vision = nn.Linear(3, 6)
        self.head = nn.Linear(6, 9)

    def forward(self, *, input_ids, pixel_values):
        hidden = self.embedding(input_ids)
        hidden = hidden.clone()
        hidden[:, 0] = hidden[:, 0] + self.vision(pixel_values)
        return {"logits": self.head(hidden)}


def make_task() -> TaskSpec:
    payload = {
            "schema_version": 1,
            "name": "tiny-monolithic",
            "data": {
                "source": {
                    "module": "data_source:trainomni/memory@1",
                    "config": {
                        "samples": [
                            {
                                "sample_id": "one",
                                "content": [
                                    {"kind": "text", "value": [1, 2, 3]},
                                    {"kind": "image", "value": [0.1, 0.2, 0.3]},
                                ],
                            }
                        ]
                    },
                },
                "transforms": [],
                "model_io": {"module": "model_io:test/tensor_mono@1"},
                "supervision": {"module": "supervision:trainomni/causal_lm@1"},
                "packer": {"module": "packer:trainomni/none@1"},
                "collator": {"module": "collator:trainomni/multimodal@1"},
            },
            "model": {
                "implementation": {"module": "model:test/tiny_monolithic@1"},
                "components": {},
            },
            "objective": {"module": "objective:trainomni/causal_lm@1"},
            "parameters": {"module": "parameter_policy:trainomni/full@1"},
            "exporters": [{"module": "exporter:trainomni/safetensors@1"}],
        }
    payload["evaluation"] = {
        "data": payload["data"],
        "evaluators": [
            {
                "module": "evaluator:trainomni/loss@1",
                "config": {"term": "token_ce", "metric_name": "eval_loss"},
            }
        ],
    }
    return TaskSpec.from_mapping(payload)


def make_registry():
    registry = builtin_registry()
    registry.register(
        ModuleDescriptor(
            ModuleId.parse("model_io:test/tensor_mono@1"),
            EmptyConfig,
            lambda config, context: TensorModelIO(),
            provides=CapabilitySet.of({"data.encoded"}),
            requires=CapabilitySet.of({"data.sample.omni"}),
        )
    )
    registry.register(
        ModuleDescriptor(
            ModuleId.parse("model:test/tiny_monolithic@1"),
            EmptyConfig,
            lambda config, context: TinyMonolithicVLM(),
            provides=CapabilitySet.of(
                {"model.monolithic", "model.output.logits", "model.parameters"}
            ),
        )
    )
    return registry


def make_engine(
    task,
    assembly,
    root: Path,
    *,
    device: str = "cpu",
    precision: str = "fp32",
    checkpoint_enabled: bool = True,
):
    run = RunSpec.from_mapping(
        {
            "schema_version": 1,
            "name": "mono",
            "seed": 2,
            "device": device,
            "precision": precision,
            "max_steps": 2,
            "optimizer": {"learning_rate": 0.01, "foreach": False},
            "checkpoint": {
                "directory": str(root),
                "every_steps": 1,
                "enabled": checkpoint_enabled,
            },
        }
    )
    selection = assembly.parameter_selection
    return TrainEngine(
        model=assembly.model,
        objective=assembly.objective,
        parameter_selection=selection,
        stream=assembly.stream,
        run=run,
        task_digest=task.digest,
        module_lock=dict(assembly.module_lock),
    )


def test_training_can_run_without_materializing_checkpoints(tmp_path: Path) -> None:
    task = make_task()
    assembly = build_task(task, ModuleResolver(make_registry()))
    root = tmp_path / "checkpoints"
    engine = make_engine(task, assembly, root, checkpoint_enabled=False)

    records = engine.train()

    assert [record.global_step for record in records] == [1, 2]
    assert not root.exists()
    with pytest.raises(SpecError, match="checkpointing is disabled"):
        engine.save_checkpoint()


def test_monolithic_model_uses_same_data_objective_and_runtime(tmp_path: Path) -> None:
    task = make_task()
    torch.manual_seed(5)
    uninterrupted_assembly = build_task(task, ModuleResolver(make_registry()))
    uninterrupted = make_engine(
        task, uninterrupted_assembly, tmp_path / "uninterrupted"
    )
    uninterrupted.train()

    torch.manual_seed(5)
    first_assembly = build_task(task, ModuleResolver(make_registry()))
    first = make_engine(task, first_assembly, tmp_path / "resumed")
    first.train(stop_after_steps=1)

    torch.manual_seed(99)
    resumed_assembly = build_task(task, ModuleResolver(make_registry()))
    resumed = make_engine(task, resumed_assembly, tmp_path / "resumed")
    resumed.resume(tmp_path / "resumed" / "step-00000001")
    records = resumed.train()
    assert [record.global_step for record in records] == [2]
    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], expected), name

    evaluated = evaluate_batches(
        model=resumed.model,
        objective=resumed_assembly.objective,
        stream=resumed_assembly.evaluation_stream,
        evaluators=resumed_assembly.evaluators,
        device=resumed.device,
        batches=1,
        batch_size=1,
    )
    assert evaluated.samples == 1
    assert evaluated.metrics["eval_loss"] > 0

    exporter_id, exporter = resumed_assembly.exporters[0]
    artifact = exporter.export(
        model=resumed.model,
        destination=tmp_path / "exported",
        identity={"exporter": exporter_id, "task_digest": task.digest},
    )
    reloaded = TinyMonolithicVLM().eval()
    reloaded.load_state_dict(load_file(artifact.uri), strict=True)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.tensor([[0.1, 0.2, 0.3]]),
    }
    resumed.model.eval()
    with torch.inference_mode():
        expected_logits = resumed.model(**inputs)["logits"]
        actual_logits = reloaded(**inputs)["logits"]
    assert torch.equal(actual_logits, expected_logits)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="requires a CUDA device with BF16 support",
)
def test_cuda_monolithic_train_resume_evaluate_export(tmp_path: Path) -> None:
    task = make_task()
    torch.manual_seed(5)
    first_assembly = build_task(task, ModuleResolver(make_registry()))
    first = make_engine(
        task,
        first_assembly,
        tmp_path / "checkpoints",
        device="cuda:0",
        precision="bf16_true",
    )
    first.train(stop_after_steps=1)

    torch.manual_seed(99)
    resumed_assembly = build_task(task, ModuleResolver(make_registry()))
    resumed = make_engine(
        task,
        resumed_assembly,
        tmp_path / "checkpoints",
        device="cuda:0",
        precision="bf16_true",
    )
    resumed.resume(tmp_path / "checkpoints" / "step-00000001")
    records = resumed.train()
    assert [record.global_step for record in records] == [2]
    assert records[0].cuda_max_allocated_bytes > 0
    assert records[0].cuda_max_reserved_bytes > 0

    evaluated = evaluate_batches(
        model=resumed.model,
        objective=resumed_assembly.objective,
        stream=resumed_assembly.evaluation_stream,
        evaluators=resumed_assembly.evaluators,
        device=resumed.device,
        batches=1,
        batch_size=1,
    )
    assert evaluated.samples == 1
    assert evaluated.metrics["eval_loss"] > 0

    exporter_id, exporter = resumed_assembly.exporters[0]
    artifact = exporter.export(
        model=resumed.model,
        destination=tmp_path / "exported",
        identity={"exporter": exporter_id, "task_digest": task.digest},
    )
    exported = load_file(artifact.uri)
    assert set(exported) == set(resumed.model.state_dict())
