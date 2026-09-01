from pathlib import Path

import pytest

from trainomni.assembly.task_builder import module_lock
from trainomni.core.errors import SpecError
from trainomni.specs.loading import load_run, load_task
from trainomni.specs.run import RunSpec
from trainomni.specs.task import TaskSpec


def ref(kind: str, name: str) -> dict:
    return {"module": f"{kind}:test/{name}@1"}


def task_payload() -> dict:
    return {
        "schema_version": 1,
        "name": "tiny-vlm",
        "data": {
            "source": ref("data_source", "memory"),
            "transforms": [],
            "model_io": ref("model_io", "tiny"),
            "supervision": ref("supervision", "causal"),
            "packer": ref("packer", "none"),
            "collator": ref("collator", "tiny"),
        },
        "model": {
            "implementation": ref("model", "composite"),
            "components": {
                "encoder": ref("encoder", "tiny"),
                "connector": ref("connector", "tiny"),
                "fusion": ref("fusion", "prefix"),
                "language": ref("language", "tiny"),
            },
        },
        "objective": ref("objective", "causal"),
        "parameters": ref("parameter_policy", "full"),
    }


def run_payload(directory: Path) -> dict:
    return {
        "schema_version": 1,
        "name": "smoke",
        "seed": 17,
        "device": "cpu",
        "precision": "fp32",
        "max_steps": 4,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 1.0,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.001,
            "foreach": False,
        },
        "checkpoint": {"directory": str(directory), "every_steps": 2},
    }


def test_task_and_run_have_separate_stable_identities(tmp_path: Path) -> None:
    task = TaskSpec.from_mapping(task_payload())
    run = RunSpec.from_mapping(run_payload(tmp_path / "checkpoints"))
    assert task.digest != run.digest
    assert task == TaskSpec.from_mapping(task_payload())
    assert run.optimizer.foreach is False
    assert run.gradient_accumulation_steps == 2
    assert run.per_device_batch_size == 2


def test_run_identity_excludes_only_physical_checkpoint_directory(
    tmp_path: Path,
) -> None:
    first = run_payload(tmp_path / "first" / "checkpoints")
    moved = run_payload(tmp_path / "moved" / "checkpoints")
    assert RunSpec.from_mapping(first).digest == RunSpec.from_mapping(moved).digest
    assert (
        RunSpec.from_mapping(first).legacy_path_bound_digest
        != RunSpec.from_mapping(moved).legacy_path_bound_digest
    )

    moved["per_device_batch_size"] = 3
    assert RunSpec.from_mapping(first).digest != RunSpec.from_mapping(moved).digest


def test_columnar_task_identity_excludes_physical_paths_but_keeps_manifest() -> None:
    left = task_payload()
    left["data"]["source"] = {
        "module": "data_source:trainomni/parquet@1",
        "config": {
            "dataset_id": "snapshot",
            "paths": ["D:/stage-a/train/*.parquet"],
            "dataset_manifest_sha256": "a" * 64,
        },
    }
    right = task_payload()
    right["data"]["source"] = {
        "module": "data_source:trainomni/parquet@1",
        "config": {
            "dataset_id": "snapshot",
            "paths": ["/mnt/stage-b/train/*.parquet"],
            "dataset_manifest_sha256": "a" * 64,
        },
    }
    left_task = TaskSpec.from_mapping(left)
    right_task = TaskSpec.from_mapping(right)
    assert left_task.digest == right_task.digest
    assert dict(module_lock(left_task)) == dict(module_lock(right_task))

    right["data"]["source"]["config"]["dataset_manifest_sha256"] = "b" * 64
    changed = TaskSpec.from_mapping(right)
    assert changed.digest != left_task.digest
    assert dict(module_lock(changed)) != dict(module_lock(left_task))


def test_named_child_sources_are_canonical_and_part_of_task_identity() -> None:
    left = task_payload()
    left["data"]["sources"] = {
        "zeta": ref("data_source", "zeta"),
        "alpha": ref("data_source", "alpha"),
    }
    left["data"]["source"] = ref("data_source", "mixture")
    right = task_payload()
    right["data"]["sources"] = {
        "alpha": ref("data_source", "alpha"),
        "zeta": ref("data_source", "zeta"),
    }
    right["data"]["source"] = ref("data_source", "mixture")
    left_task = TaskSpec.from_mapping(left)
    right_task = TaskSpec.from_mapping(right)
    assert tuple(name for name, _ in left_task.data.sources) == ("alpha", "zeta")
    assert left_task.digest == right_task.digest
    assert len(left_task.module_refs()) == len(TaskSpec.from_mapping(task_payload()).module_refs()) + 2


def test_data_adapter_is_typed_and_part_of_task_identity() -> None:
    payload = task_payload()
    payload["data"]["adapter"] = ref("data_adapter", "msswift")
    task = TaskSpec.from_mapping(payload)
    assert str(task.data.adapter.module_id) == "data_adapter:test/msswift@1"
    assert task.data.adapter in task.module_refs()
    assert task.digest != TaskSpec.from_mapping(task_payload()).digest


def test_nested_unknown_keys_fail_closed(tmp_path: Path) -> None:
    payload = task_payload()
    payload["model"]["implementaton"] = payload["model"]["implementation"]
    with pytest.raises(SpecError, match="model contains unknown keys"):
        TaskSpec.from_mapping(payload)

    run = run_payload(tmp_path / "checkpoints")
    run["checkpoint"]["every_step"] = 2
    with pytest.raises(SpecError, match="checkpoint contains unknown keys"):
        RunSpec.from_mapping(run)


def test_compile_options_require_explicit_enable(tmp_path: Path) -> None:
    run = run_payload(tmp_path / "checkpoints")
    run["compile"] = {"backend": "eager"}
    with pytest.raises(SpecError, match="enabled=true"):
        RunSpec.from_mapping(run)

    run["compile"] = {
        "enabled": True,
        "backend": "eager",
        "fullgraph": True,
        "dynamic": False,
    }
    resolved = RunSpec.from_mapping(run)
    assert resolved.compile.enabled
    assert resolved.compile.backend == "eager"
    assert resolved.compile.fullgraph


def test_yaml_task_and_run_loading_matches_direct_specs(tmp_path: Path) -> None:
    import yaml

    task_path = tmp_path / "task.yaml"
    run_path = tmp_path / "run.yaml"
    task_path.write_text(yaml.safe_dump(task_payload()), encoding="utf-8")
    run_path.write_text(
        yaml.safe_dump(run_payload(tmp_path / "checkpoints")),
        encoding="utf-8",
    )
    assert load_task(task_path) == TaskSpec.from_mapping(task_payload())
    assert load_run(run_path) == RunSpec.from_mapping(
        run_payload(tmp_path / "checkpoints")
    )
