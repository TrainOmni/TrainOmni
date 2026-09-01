from pathlib import Path

import pytest
from torch import nn

from trainomni.assembly import task_builder
from trainomni.core.assets import task_asset_provenance, validate_asset_fields
from trainomni.core.errors import SpecError
from trainomni.runtime.loop.engine import TrainEngine
from trainomni.specs.run import RunSpec
from trainomni.specs.task import TaskSpec


def _task(
    *,
    revision: str | None = None,
    manifest: str | None = None,
    model_location: str = "producer/model",
    processor_location: str = "producer/processor",
) -> TaskSpec:
    asset = {
        **({"revision": revision} if revision is not None else {}),
        **(
            {"asset_manifest_sha256": manifest}
            if manifest is not None
            else {}
        ),
    }
    return TaskSpec.from_mapping(
        {
            "schema_version": 1,
            "name": "asset-identity",
            "data": {
                "source": {
                    "module": "data_source:trainomni/memory@1",
                    "config": {
                        "samples": [
                            {
                                "sample_id": "one",
                                "content": [{"kind": "text", "value": "hello"}],
                            }
                        ]
                    },
                },
                "transforms": [],
                "model_io": {
                    "module": "model_io:trainomni/transformers@1",
                    "config": {
                        "processor_name_or_path": processor_location,
                        **asset,
                    },
                },
                "supervision": {"module": "supervision:trainomni/causal_lm@1"},
                "packer": {"module": "packer:trainomni/none@1"},
                "collator": {"module": "collator:trainomni/multimodal@1"},
            },
            "model": {
                "implementation": {
                    "module": "model:trainomni/monolithic_transformers@1",
                    "config": {
                        "model_name_or_path": model_location,
                        **asset,
                    },
                },
                "components": {},
            },
            "objective": {"module": "objective:trainomni/causal_lm@1"},
            "parameters": {"module": "parameter_policy:trainomni/full@1"},
        }
    )


@pytest.mark.parametrize(
    "identity",
    (
        {"revision": "a" * 40},
        {"manifest": "b" * 64},
    ),
)
def test_transformers_assets_require_one_immutable_identity(identity: dict) -> None:
    provenance = task_asset_provenance(_task(**identity))
    assert provenance.reproducible
    assert provenance.issues == ()
    assert len(provenance.lock_entries) == 2

    unpinned = task_asset_provenance(_task())
    assert not unpinned.reproducible
    assert len(unpinned.issues) == 2


def test_asset_identity_format_is_strict() -> None:
    with pytest.raises(ValueError, match="immutable"):
        validate_asset_fields(revision="main", asset_manifest_sha256=None)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_asset_fields(revision=None, asset_manifest_sha256="A" * 64)


def test_columnar_snapshot_manifest_participates_in_provenance() -> None:
    payload = _task(revision="a" * 40)
    mapping = {
        "schema_version": payload.schema_version,
        "name": payload.name,
        "data": {
            "source": {
                "module": "data_source:trainomni/parquet@1",
                "config": {
                    "dataset_id": "dataset-v1",
                    "paths": ["data/*.parquet"],
                    "dataset_manifest_sha256": "c" * 64,
                },
            },
            "transforms": [],
            "model_io": {
                "module": "model_io:trainomni/transformers@1",
                "config": {
                    "processor_name_or_path": "producer/processor",
                    "revision": "a" * 40,
                },
            },
            "supervision": {"module": "supervision:trainomni/causal_lm@1"},
            "packer": {"module": "packer:trainomni/none@1"},
            "collator": {"module": "collator:trainomni/multimodal@1"},
        },
        "model": {
            "implementation": {
                "module": "model:trainomni/monolithic_transformers@1",
                "config": {
                    "model_name_or_path": "producer/model",
                    "revision": "a" * 40,
                },
            },
            "components": {},
        },
        "objective": {"module": "objective:trainomni/causal_lm@1"},
        "parameters": {"module": "parameter_policy:trainomni/full@1"},
    }
    pinned = task_asset_provenance(TaskSpec.from_mapping(mapping))
    assert pinned.reproducible
    mapping["data"]["source"]["config"].pop("dataset_manifest_sha256")
    unpinned = task_asset_provenance(TaskSpec.from_mapping(mapping))
    assert not unpinned.reproducible
    assert any("dataset_manifest_sha256" in issue for issue in unpinned.issues)


def test_builtin_provenance_changes_module_lock(monkeypatch) -> None:
    task = _task(revision="a" * 40)
    original = dict(task_builder.module_lock(task))
    assert "builtin-core:trainomni" in original
    monkeypatch.setattr(task_builder, "builtin_source_sha256", lambda: "f" * 64)
    monkeypatch.setattr(
        task_builder,
        "BUILTIN_CODE_PROVENANCE",
        "different-builtin-source",
    )
    changed = dict(task_builder.module_lock(task))
    assert changed["builtin-core:trainomni"] != original["builtin-core:trainomni"]
    assert {
        key: value for key, value in changed.items() if key != "builtin-core:trainomni"
    } == {
        key: value for key, value in original.items() if key != "builtin-core:trainomni"
    }


def test_local_asset_manifest_makes_the_physical_staging_root_relocatable() -> None:
    left = _task(
        manifest="b" * 64,
        model_location="D:/models/stage-a/model",
        processor_location="D:/models/stage-a/processor",
    )
    right = _task(
        manifest="b" * 64,
        model_location="/mnt/models/stage-b/model",
        processor_location="/mnt/models/stage-b/processor",
    )
    assert left.digest == right.digest
    assert dict(task_builder.module_lock(left)) == dict(
        task_builder.module_lock(right)
    )

    remote_left = _task(revision="a" * 40, model_location="owner/model-a")
    remote_right = _task(revision="a" * 40, model_location="owner/model-b")
    assert remote_left.digest != remote_right.digest


def test_unpinned_assets_cannot_claim_exact_resume(tmp_path: Path) -> None:
    run = RunSpec.from_mapping(
        {
            "schema_version": 1,
            "name": "unpinned",
            "seed": 0,
            "device": "cpu",
            "precision": "fp32",
            "max_steps": 1,
            "optimizer": {"learning_rate": 0.01, "foreach": False},
            "checkpoint": {"directory": str(tmp_path / "checkpoints")},
        }
    )
    with pytest.raises(SpecError, match="immutable external asset identity"):
        TrainEngine(
            model=nn.Linear(1, 1),
            objective=None,
            parameter_selection=None,
            stream=None,
            run=run,
            task_digest="a" * 64,
            module_lock={},
            reproducible=False,
            provenance_issues=("model is unpinned",),
        )
