"""Materialize immutable configs for the medium-data real-VLM route matrix."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "medium-validation" / "configs"
DATA = ROOT / "data" / "medium-v1"
CACHE = ROOT / "cache"


@dataclass(frozen=True, slots=True)
class Route:
    name: str
    task_template: str
    run_template: Path
    train_file: str
    validation_file: str
    text_tokens: int
    initial_artifact: str
    cache_directory: str | None = None


ROUTES = (
    Route(
        "alignment",
        "stage-01-alignment.task.json",
        ROOT / "runs" / "01-alignment-v2" / "run.json",
        "diagram-train.jsonl",
        "diagram-validation.jsonl",
        64,
        "none",
    ),
    Route(
        "multimodal_cpt",
        "stage-02-pretraining.task.json",
        ROOT / "runs" / "02-pretraining-v3" / "run.json",
        "diagram-train.jsonl",
        "diagram-validation.jsonl",
        64,
        "none",
    ),
    Route(
        "full_sft",
        "stage-03-sft.task.json",
        ROOT / "runs" / "03-sft" / "run.json",
        "intergps-train.jsonl",
        "intergps-validation.jsonl",
        64,
        "none",
    ),
    Route(
        "lora_sft",
        "stage-06-lora-sft.task.json",
        ROOT / "runs" / "06-lora-sft" / "run.json",
        "intergps-train.jsonl",
        "intergps-validation.jsonl",
        64,
        "stage04",
    ),
    Route(
        "offline_dense_kd",
        "stage-04-kd.task.json",
        ROOT / "runs" / "04-kd" / "run.json",
        "kd-train.jsonl",
        "kd-validation.jsonl",
        32,
        "stage04",
        "medium-kd-v1",
    ),
    Route(
        "offline_reference_dpo",
        "stage-05-dpo.task.json",
        ROOT / "runs" / "05-dpo-v2" / "run.json",
        "dpo-train.jsonl",
        "dpo-validation.jsonl",
        32,
        "stage04",
        "medium-dpo-v1",
    ),
    Route(
        "lora_dpo",
        "stage-07-lora-dpo.task.json",
        ROOT / "runs" / "07-lora-dpo" / "run.json",
        "dpo-train.jsonl",
        "dpo-validation.jsonl",
        32,
        "stage04",
        "medium-dpo-v1",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def patch_pipeline(
    pipeline: dict[str, Any],
    *,
    data_path: Path,
    text_tokens: int,
    cache_index: Path | None,
) -> None:
    pipeline["source"]["config"].update(
        {
            "path": data_path.as_posix(),
            "sha256": sha256(data_path),
            "repeat": True,
        }
    )
    pipeline["model_io"]["config"]["max_text_tokens"] = text_tokens
    transforms = pipeline.get("transforms", [])
    if cache_index is None:
        if transforms:
            raise ValueError("non-cache route unexpectedly contains data transforms")
        return
    if len(transforms) != 1 or transforms[0].get("module") != (
        "sample_transform:trainomni/tensor_cache@1"
    ):
        raise ValueError("cache route must contain exactly one tensor-cache transform")
    transforms[0]["config"] = {
        "index_path": cache_index.as_posix(),
        "index_sha256": sha256(cache_index),
    }


def materialize_task(route: Route) -> dict[str, Any]:
    task = deepcopy(load_json(ROOT / route.task_template))
    task["name"] = f"real-vlm-medium-v1-{route.name.replace('_', '-')}"
    cache_index = (
        None
        if route.cache_directory is None
        else (CACHE / route.cache_directory / "index.json").resolve()
    )
    patch_pipeline(
        task["data"],
        data_path=(DATA / route.train_file).resolve(),
        text_tokens=route.text_tokens,
        cache_index=cache_index,
    )
    patch_pipeline(
        task["evaluation"]["data"],
        data_path=(DATA / route.validation_file).resolve(),
        text_tokens=route.text_tokens,
        cache_index=cache_index,
    )
    model_config = task["model"]["implementation"]["config"]
    if route.initial_artifact == "none":
        model_config.pop("initial_artifact", None)
        model_config.pop("initial_artifact_sha256", None)
    elif route.initial_artifact == "stage04":
        artifact = (ROOT / "artifacts" / "stage-04-kd").resolve()
        artifact_manifest = load_json(artifact / "manifest.json")
        model_config["initial_artifact"] = artifact.as_posix()
        model_config["initial_artifact_sha256"] = str(artifact_manifest["sha256"])
    else:
        raise ValueError(f"unknown initial-artifact selection: {route.initial_artifact}")
    return task


def materialize_run(route: Route) -> dict[str, Any]:
    run = deepcopy(load_json(route.run_template))
    run["name"] = f"real-vlm-medium-v1-{route.name.replace('_', '-')}"
    run["seed"] = 26082220 + ROUTES.index(route)
    run["attention_kernel"] = "sdpa"
    run["max_steps"] = 16
    run["execution"] = {"backend": "single", "expected_world_size": 1}
    run["update_evidence"]["every_steps"] = 4
    run["checkpoint"] = {
        "directory": (
            ROOT
            / "medium-validation"
            / "runs"
            / route.name
            / "checkpoints"
        ).resolve().as_posix(),
        # The medium-data pass studies forward/backward/loss behavior. Full
        # checkpoint/evaluation/export paths are covered by the preceding real
        # route gates and are not duplicated into multi-GB payloads here.
        "enabled": False,
        "every_steps": 16,
    }
    return run


def validate_cache_coverage(route: Route, task: dict[str, Any]) -> None:
    if route.cache_directory is None:
        return
    index_path = Path(task["data"]["transforms"][0]["config"]["index_path"])
    samples = load_json(index_path).get("samples")
    if not isinstance(samples, dict):
        raise TypeError(f"cache samples must be an object: {index_path}")
    required: set[str] = set()
    for filename in (route.train_file, route.validation_file):
        for line in (DATA / filename).read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            required.add(str(value["sample_id"]))
    missing = sorted(required - set(samples))
    if missing:
        raise ValueError(f"cache {index_path} is missing samples: {missing[:3]}")


def main() -> None:
    manifest = load_json(DATA / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("medium dataset manifest schema mismatch")
    route_receipts = []
    for route in ROUTES:
        task = materialize_task(route)
        validate_cache_coverage(route, task)
        run = materialize_run(route)
        # Local-code provenance intentionally confines module paths beneath the
        # task file's directory, so task specs remain at the consumer root.
        task_path = ROOT / f"medium-{route.name}.task.json"
        run_path = OUTPUT / f"{route.name}.run.json"
        write_json(task_path, task)
        write_json(run_path, run)
        route_receipts.append(
            {
                "name": route.name,
                "task": task_path.relative_to(ROOT).as_posix(),
                "task_sha256": sha256(task_path),
                "run": run_path.relative_to(ROOT).as_posix(),
                "run_sha256": sha256(run_path),
                "train_data_sha256": task["data"]["source"]["config"]["sha256"],
                "validation_data_sha256": task["evaluation"]["data"]["source"]
                ["config"]["sha256"],
                "cache_index_sha256": (
                    None
                    if route.cache_directory is None
                    else task["data"]["transforms"][0]["config"]["index_sha256"]
                ),
            }
        )
    write_json(
        OUTPUT.parent / "config-manifest.json",
        {
            "schema_version": 1,
            "dataset_manifest_sha256": sha256(DATA / "manifest.json"),
            "routes": route_receipts,
        },
    )
    print(f"materialized {len(ROUTES)} routes in {OUTPUT}")


if __name__ == "__main__":
    main()
