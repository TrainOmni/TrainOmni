"""Fail closed over the medium-data route outputs and emit a compact receipt."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT / "medium-validation"


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def slope(values: list[float]) -> float:
    x_mean = (len(values) + 1) / 2
    y_mean = mean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(1, len(values) + 1))
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values, start=1)
    ) / denominator


def series_summary(values: list[float]) -> dict[str, Any]:
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise ValueError("each medium route must contain 16 finite loss values")
    first_window = mean(values[:4])
    last_window = mean(values[-4:])
    return {
        "values": values,
        "first": values[0],
        "last": values[-1],
        "minimum": min(values),
        "maximum": max(values),
        "first_4_mean": first_window,
        "last_4_mean": last_window,
        "last_vs_first_4_fraction": (last_window - first_window)
        / max(abs(first_window), 1e-12),
        "least_squares_slope_per_step": slope(values),
    }


def summarize_route(route: dict[str, Any]) -> dict[str, Any]:
    name = str(route["name"])
    task_path = ROOT / route["task"]
    run_path = ROOT / route["run"]
    task = load_json(task_path)
    run = load_json(run_path)
    output = VALIDATION / "runs" / name
    events_path = output / "metrics" / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    initialized = [event for event in events if event.get("event") == "engine_initialized"]
    steps = [event for event in events if event.get("event") == "optimizer_step"]
    if len(initialized) != 1:
        raise ValueError(f"{name}: expected one engine_initialized event")
    if [int(event["global_step"]) for event in steps] != list(range(1, 17)):
        raise ValueError(f"{name}: optimizer-step sequence is incomplete")
    engine = initialized[0]
    execution = engine.get("execution", {})
    if (
        engine.get("device") != "cuda:0"
        or engine.get("precision") != "bf16_true"
        or engine.get("attention_kernel") != "sdpa"
        or execution.get("backend") != "single"
        or execution.get("world_size") != 1
    ):
        raise ValueError(f"{name}: execution identity does not match the gate")
    losses = [float(event["loss"]) for event in steps]
    grad_norms = [float(event["grad_norm"]) for event in steps]
    if not all(math.isfinite(value) and value >= 0 for value in grad_norms):
        raise ValueError(f"{name}: gradients are not finite")
    for event in steps:
        per_rank = event.get("data_metrics_by_rank")
        if per_rank != [{"rank": 0, "metrics": event.get("data_metrics", {})}]:
            raise ValueError(f"{name}: rank-structured data metrics are invalid")
    expected_groups = tuple(run["update_evidence"]["required_groups"])
    evidence_steps: dict[str, Any] = {}
    for event in steps:
        step = int(event["global_step"])
        evidence = event.get("parameter_evidence", {})
        if step % int(run["update_evidence"]["every_steps"]) != 0:
            continue
        if set(evidence) != set(expected_groups):
            raise ValueError(f"{name}: step {step} update evidence is incomplete")
        for group, proof in evidence.items():
            if (
                int(proof["changed_tensor_count"]) <= 0
                or int(proof["changed_sampled_elements"]) <= 0
                or not math.isfinite(float(proof["gradient_norm"]))
            ):
                raise ValueError(f"{name}: group {group} did not prove an update")
        evidence_steps[str(step)] = {
            group: {
                key: proof[key]
                for key in (
                    "before_sha256",
                    "after_sha256",
                    "gradient_norm",
                    "changed_tensor_count",
                    "changed_sampled_elements",
                    "sampled_elements",
                    "max_abs_sampled_update",
                )
            }
            for group, proof in evidence.items()
        }
    if len(evidence_steps) != 4:
        raise ValueError(f"{name}: expected four component-update evidence points")
    if (output / "checkpoints").exists():
        raise ValueError(f"{name}: medium gate unexpectedly materialized a checkpoint")
    loss_terms: dict[str, list[float]] = {}
    for event in steps:
        for term, value in event.get("loss_terms", {}).items():
            loss_terms.setdefault(str(term), []).append(float(value))
    term_summaries = {
        term: series_summary(values) for term, values in sorted(loss_terms.items())
    }
    return {
        "task": task_path.relative_to(ROOT).as_posix(),
        "task_file_sha256": sha256(task_path),
        "task_digest": engine["task_digest"],
        "run": run_path.relative_to(ROOT).as_posix(),
        "run_file_sha256": sha256(run_path),
        "run_digest": engine["run_digest"],
        "train_source": task["data"]["source"]["config"],
        "validation_source": task["evaluation"]["data"]["source"]["config"],
        "execution": execution,
        "steps": 16,
        "micro_batches": sum(int(event["micro_batches"]) for event in steps),
        "loss": series_summary(losses),
        "loss_terms": term_summaries,
        "grad_norm": {
            "minimum": min(grad_norms),
            "maximum": max(grad_norms),
            "mean": mean(grad_norms),
        },
        "gpu_memory": {
            "max_allocated_bytes": max(
                int(event["cuda_max_allocated_bytes"]) for event in steps
            ),
            "max_reserved_bytes": max(
                int(event["cuda_max_reserved_bytes"]) for event in steps
            ),
        },
        "update_evidence": evidence_steps,
        "checkpoint_materialized": False,
    }


def main() -> None:
    config_manifest_path = VALIDATION / "config-manifest.json"
    config_manifest = load_json(config_manifest_path)
    routes = config_manifest.get("routes")
    if not isinstance(routes, list) or len(routes) != 7:
        raise ValueError("medium config manifest must contain seven routes")
    summaries = {str(route["name"]): summarize_route(route) for route in routes}
    receipt = {
        "schema_version": 1,
        "claim_boundary": (
            "Single-GPU engineering validation over varied medium fixtures; "
            "loss observations are not model-quality claims."
        ),
        "dataset_manifest_sha256": config_manifest["dataset_manifest_sha256"],
        "config_manifest_sha256": sha256(config_manifest_path),
        "routes": summaries,
        "totals": {
            "routes": len(summaries),
            "optimizer_steps": sum(value["steps"] for value in summaries.values()),
            "micro_batches": sum(
                value["micro_batches"] for value in summaries.values()
            ),
        },
    }
    output = VALIDATION / "receipt.json"
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"verified {len(summaries)} routes / 112 optimizer steps: {output}")


if __name__ == "__main__":
    main()
