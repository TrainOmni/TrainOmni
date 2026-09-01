import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from trainomni.api.evaluate import evaluate
from trainomni.api.export import export_artifact
from trainomni.api.train import assemble, train
from trainomni.catalog.local import source_tree_digest
from trainomni.core.errors import CapabilityError

MODULE_TOML = """[module]
id = "{module_id}"
entrypoint = "module:descriptor"
api_version = 1
"""


def write_local_module(root: Path, name: str, module_id: str, source: str) -> dict:
    directory = root / "modules" / name
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    (directory / "module.py").write_text(source, encoding="utf-8")
    (directory / "module.toml").write_text(
        MODULE_TOML.format(module_id=module_id), encoding="utf-8"
    )
    return {
        "module": module_id,
        "path": f"modules/{name}",
        "source_sha256": source_tree_digest(directory),
    }


def create_task(task_root: Path, *, use_mixture: bool = False) -> Path:
    local_modules = [
        write_local_module(
            task_root,
            "model_io",
            "model_io:test/tensor@1",
            """from dataclasses import dataclass
import torch
from trainomni.contracts.batch import EncodedSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.module import ModuleDescriptor, ModuleId
@dataclass(frozen=True)
class Config: pass
class ModelIO:
    def encode(self, sample):
        blocks = {block.kind: block.value for block in sample.content}
        return EncodedSample(
            sample.sample_id,
            {
                "input_ids": torch.tensor(blocks["text"], dtype=torch.long),
                "pixel_values": torch.tensor(blocks["image"], dtype=torch.float32),
            },
        )
def descriptor():
    return ModuleDescriptor(
        ModuleId.parse("model_io:test/tensor@1"), Config,
        lambda config, context: ModelIO(),
        provides=CapabilitySet.of({"data.encoded"}),
        requires=CapabilitySet.of({"data.sample.omni"}),
    )
""",
        ),
        write_local_module(
            task_root,
            "encoder",
            "encoder:test/tiny@1",
            """from dataclasses import dataclass
from torch import nn
from trainomni.contracts.features import ModalFeatures
from trainomni.core.capability import CapabilitySet
from trainomni.core.module import ModuleDescriptor, ModuleId
@dataclass(frozen=True)
class Config:
    hidden: int = 8
class Encoder(nn.Module):
    def __init__(self, hidden): super().__init__(); self.projection = nn.Linear(3, hidden)
    def forward(self, values): return ModalFeatures(self.projection(values).unsqueeze(1))
def descriptor():
    return ModuleDescriptor(
        ModuleId.parse("encoder:test/tiny@1"), Config,
        lambda config, context: Encoder(config.hidden),
        provides=CapabilitySet.of({"component.encoder"}),
    )
""",
        ),
        write_local_module(
            task_root,
            "connector",
            "connector:test/tiny@1",
            """from dataclasses import dataclass
from torch import nn
from trainomni.contracts.features import ModalFeatures
from trainomni.core.capability import CapabilitySet
from trainomni.core.module import ModuleDescriptor, ModuleId
@dataclass(frozen=True)
class Config:
    hidden: int = 8
class Connector(nn.Module):
    def __init__(self, hidden): super().__init__(); self.projection = nn.Linear(hidden, hidden)
    def forward(self, features): return ModalFeatures(self.projection(features.embeddings))
def descriptor():
    return ModuleDescriptor(
        ModuleId.parse("connector:test/tiny@1"), Config,
        lambda config, context: Connector(config.hidden),
        provides=CapabilitySet.of({"component.connector"}),
    )
""",
        ),
        write_local_module(
            task_root,
            "fusion",
            "fusion:test/prefix@1",
            """from dataclasses import dataclass
from torch import cat, nn
from trainomni.core.capability import CapabilitySet
from trainomni.core.module import ModuleDescriptor, ModuleId
@dataclass(frozen=True)
class Config: pass
class Fusion(nn.Module):
    def forward(self, *, language, input_ids, modal_features, attention_mask=None,
                modal_positions=None, **kwargs):
        modal_features = modal_features.concatenate()
        embeddings = language.embed(input_ids)
        embeddings = cat((embeddings[:, :1] + modal_features.embeddings[:, :1],
                          embeddings[:, 1:]), dim=1)
        return language.forward_embeddings(embeddings, attention_mask=attention_mask)
def descriptor():
    return ModuleDescriptor(
        ModuleId.parse("fusion:test/prefix@1"), Config,
        lambda config, context: Fusion(),
        provides=CapabilitySet.of({"component.fusion"}),
    )
""",
        ),
        write_local_module(
            task_root,
            "language",
            "language:test/tiny@1",
            """from dataclasses import dataclass
from types import SimpleNamespace
from torch import nn
from trainomni.core.capability import CapabilitySet
from trainomni.core.module import ModuleDescriptor, ModuleId
@dataclass(frozen=True)
class Config:
    vocab: int = 11
    hidden: int = 8
class Language(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__(); self.embedding = nn.Embedding(vocab, hidden); self.head = nn.Linear(hidden, vocab)
    def embed(self, input_ids): return self.embedding(input_ids)
    def forward_embeddings(self, embeddings, **kwargs):
        return SimpleNamespace(logits=self.head(embeddings))
def descriptor():
    return ModuleDescriptor(
        ModuleId.parse("language:test/tiny@1"), Config,
        lambda config, context: Language(config.vocab, config.hidden),
        provides=CapabilitySet.of({"component.language"}),
    )
""",
        ),
    ]
    samples = [
        {
            "sample_id": "one",
            "content": [
                {"kind": "text", "value": [1, 2, 3, 4]},
                {"kind": "image", "value": [0.2, -0.1, 0.3]},
            ],
        },
        {
            "sample_id": "two",
            "content": [
                {"kind": "text", "value": [2, 4, 1, 3]},
                {"kind": "image", "value": [-0.2, 0.4, 0.1]},
            ],
        },
    ]
    data = {
        "source": {
            "module": "data_source:trainomni/memory@1",
            "config": {"samples": samples},
        },
        "transforms": [],
        "model_io": {"module": "model_io:test/tensor@1"},
        "supervision": {"module": "supervision:trainomni/causal_lm@1"},
        "packer": {"module": "packer:trainomni/none@1"},
        "collator": {"module": "collator:trainomni/multimodal@1"},
    }
    if use_mixture:
        data["sources"] = {
            "first": {
                "module": "data_source:trainomni/memory@1",
                "config": {"samples": [samples[0]]},
            },
            "second": {
                "module": "data_source:trainomni/memory@1",
                "config": {"samples": [samples[1]]},
            },
        }
        data["source"] = {
            "module": "data_source:trainomni/mixture@1",
            "config": {"weights": {"first": 1.0, "second": 3.0}, "seed": 19},
        }
    task = {
        "schema_version": 1,
        "name": "file-driven-tiny-composite",
        "local_modules": local_modules,
        "data": data,
        "model": {
            "implementation": {"module": "model:trainomni/composite@1"},
            "components": {
                "encoder": {"module": "encoder:test/tiny@1"},
                "connector": {"module": "connector:test/tiny@1"},
                "fusion": {"module": "fusion:test/prefix@1"},
                "language": {"module": "language:test/tiny@1"},
            },
        },
        "objective": {"module": "objective:trainomni/causal_lm@1"},
        "parameters": {"module": "parameter_policy:trainomni/full@1"},
        "exporters": [{"module": "exporter:trainomni/safetensors@1"}],
    }
    task["evaluation"] = {
        "data": task["data"],
        "evaluators": [
            {
                "module": "evaluator:trainomni/loss@1",
                "config": {"term": "token_ce", "metric_name": "eval_loss"},
            }
        ],
    }
    path = task_root / "task.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    return path


def create_run(
    run_root: Path,
    *,
    device: str = "cpu",
    precision: str = "fp32",
) -> Path:
    path = run_root / "run.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "api-smoke",
                "seed": 7,
                "device": device,
                "precision": precision,
                "max_steps": 4,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 0.01,
                    "foreach": False,
                },
                "checkpoint": {
                    "directory": "outputs/checkpoints",
                    "every_steps": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_separate_task_and_run_files_drive_the_same_runtime(tmp_path: Path) -> None:
    task_path = create_task(tmp_path / "task")
    run_root = tmp_path / "run"
    run_root.mkdir()
    run_path = create_run(run_root)
    first = train(
        task_path=task_path,
        run_path=run_path,
        allow_local_code=True,
        stop_after_steps=2,
    )
    checkpoint = run_root / "outputs" / "checkpoints" / "step-00000002"
    assert first.final_step == 2
    assert (run_root / "outputs" / "resolved" / "task.resolved.json").is_file()
    assert (run_root / "outputs" / "resolved" / "run.resolved.json").is_file()
    assert (run_root / "outputs" / "resolved" / "modules.lock.json").is_file()
    assert (run_root / "outputs" / "resolved" / "parameters.json").is_file()
    assert (run_root / "outputs" / "run-manifest.json").is_file()
    assert (run_root / "outputs" / "metrics" / "events.jsonl").is_file()
    environment = dict(os.environ)
    source_root = str(Path(__file__).parents[3] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trainomni",
            "train",
            "--task",
            str(task_path),
            "--run",
            str(run_path),
            "--allow-local-code",
            "--resume",
            str(checkpoint),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["final_step"] == 4
    assert receipt["steps_executed"] == 2
    evaluated = evaluate(
        task_path=task_path,
        run_path=run_path,
        checkpoint=run_root / "outputs" / "checkpoints" / "step-00000004",
        batches=2,
        allow_local_code=True,
    )
    assert evaluated.samples == 4
    assert evaluated.metrics["eval_loss"] > 0
    assert evaluated.receipt.is_file()
    exported = export_artifact(
        task_path=task_path,
        run_path=run_path,
        checkpoint=run_root / "outputs" / "checkpoints" / "step-00000004",
        allow_local_code=True,
    )
    assert exported.artifact.kind == "safetensors"
    assert len(exported.artifact.digest) == 64
    from safetensors.torch import load_file

    tensors = load_file(exported.artifact.uri)
    assert "encoder.projection.weight" in tensors
    assert "connector.projection.weight" in tensors
    assert "language.head.weight" in tensors


def test_named_child_sources_require_a_composing_source(tmp_path: Path) -> None:
    task_path = create_task(tmp_path / "task", use_mixture=True)
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    payload["data"]["source"] = {
        "module": "data_source:trainomni/memory@1",
        "config": {
            "samples": [
                {
                    "sample_id": "unused-top-level",
                    "content": [{"kind": "text", "value": [1, 2]}],
                }
            ]
        },
    }
    task_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CapabilityError, match="child-source composition contract"):
        assemble(task_path=task_path, allow_local_code=True)


def test_native_lora_task_trains_evaluates_and_exports_adapter(tmp_path: Path) -> None:
    task_path = create_task(tmp_path / "task")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["parameters"] = {
        "module": "parameter_policy:trainomni/lora_linear@1",
        "config": {
            "target_patterns": ["connector\\.projection", "language\\.head"],
            "rank": 2,
            "alpha": 4.0,
            "group_name": "lora",
        },
    }
    task["exporters"] = [
        {"module": "exporter:trainomni/lora_adapter@1"}
    ]
    task_path.write_text(json.dumps(task), encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()
    run_path = create_run(run_root)

    trained = train(
        task_path=task_path,
        run_path=run_path,
        allow_local_code=True,
        stop_after_steps=2,
    )
    checkpoint = run_root / "outputs" / "checkpoints" / "step-00000002"
    assert trained.final_step == 2
    evaluated = evaluate(
        task_path=task_path,
        run_path=run_path,
        checkpoint=checkpoint,
        batches=1,
        allow_local_code=True,
    )
    assert evaluated.metrics["eval_loss"] > 0
    exported = export_artifact(
        task_path=task_path,
        run_path=run_path,
        checkpoint=checkpoint,
        allow_local_code=True,
    )
    manifest = json.loads(
        (Path(exported.artifact.uri) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["kind"] == "trainomni_linear_lora"
    assert set(manifest["modules"]) == {
        "connector.projection",
        "language.head",
    }


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="requires a CUDA device with BF16 support",
)
def test_cuda_composite_fresh_process_lifecycle(tmp_path: Path) -> None:
    task_path = create_task(tmp_path / "task", use_mixture=True)

    uninterrupted_root = tmp_path / "uninterrupted"
    uninterrupted_root.mkdir()
    uninterrupted_run = create_run(
        uninterrupted_root, device="cuda:0", precision="bf16_true"
    )
    uninterrupted = train(
        task_path=task_path,
        run_path=uninterrupted_run,
        allow_local_code=True,
    )
    assert uninterrupted.final_step == 4

    resumed_root = tmp_path / "resumed"
    resumed_root.mkdir()
    resumed_run = create_run(
        resumed_root, device="cuda:0", precision="bf16_true"
    )
    first = train(
        task_path=task_path,
        run_path=resumed_run,
        allow_local_code=True,
        stop_after_steps=2,
    )
    assert first.final_step == 2
    checkpoint = resumed_root / "outputs" / "checkpoints" / "step-00000002"

    environment = dict(os.environ)
    source_root = str(Path(__file__).parents[3] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trainomni",
            "train",
            "--task",
            str(task_path),
            "--run",
            str(resumed_run),
            "--allow-local-code",
            "--resume",
            str(checkpoint),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["final_step"] == 4

    uninterrupted_state = load_file(
        uninterrupted_root
        / "outputs"
        / "checkpoints"
        / "step-00000004"
        / "model.safetensors"
    )
    resumed_checkpoint = (
        resumed_root / "outputs" / "checkpoints" / "step-00000004"
    )
    resumed_state = load_file(resumed_checkpoint / "model.safetensors")
    assert set(resumed_state) == set(uninterrupted_state)
    for name, expected in uninterrupted_state.items():
        torch.testing.assert_close(resumed_state[name], expected, rtol=0, atol=0)

    events = [
        json.loads(line)
        for line in (
            resumed_root / "outputs" / "metrics" / "events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    steps = [event for event in events if event["event"] == "optimizer_step"]
    assert len(steps) == 4
    assert all(event["cuda_max_allocated_bytes"] > 0 for event in steps)
    assert all(event["cuda_max_reserved_bytes"] > 0 for event in steps)
    assert [event["data_metrics"]["data/mixture/samples"] for event in steps] == [
        2,
        4,
        6,
        8,
    ]
    assert all(
        sum(
            value
            for name, value in event["data_metrics"].items()
            if name.startswith("data/source/")
        )
        == event["data_metrics"]["data/mixture/samples"]
        for event in steps
    )

    evaluated = evaluate(
        task_path=task_path,
        run_path=resumed_run,
        checkpoint=resumed_checkpoint,
        batches=2,
        allow_local_code=True,
    )
    assert evaluated.samples == 4
    assert evaluated.metrics["eval_loss"] > 0
    exported = export_artifact(
        task_path=task_path,
        run_path=resumed_run,
        checkpoint=resumed_checkpoint,
        allow_local_code=True,
    )
    exported_state = load_file(exported.artifact.uri)
    assert set(exported_state) == set(resumed_state)
