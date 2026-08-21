"""Real-VLM verification of Framework's task-local extension boundaries.

Large checkpoints are created under the system temporary directory and removed
after each route.  Only compact, reproducible receipts remain in this project.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RECEIPT_ROOT = ROOT / "extension-validation"

ROUTES: dict[str, dict[str, Any]] = {
    "custom-objective": {
        "task": "extension-custom-objective.task.json",
        "steps": 2,
        "batch_size": 1,
        "kernel": "auto",
        "seed": 26082401,
    },
    "attention-eager": {
        "task": "extension-attention.task.json",
        "steps": 2,
        "batch_size": 1,
        "kernel": "eager",
        "seed": 26082402,
    },
    "attention-sdpa": {
        "task": "extension-attention.task.json",
        "steps": 2,
        "batch_size": 1,
        "kernel": "sdpa",
        "seed": 26082403,
    },
    "weighted-mixture": {
        "task": "extension-mixture.task.json",
        "steps": 4,
        "batch_size": 2,
        "kernel": "auto",
        "seed": 26082404,
    },
    "sequence-packing": {
        "task": "extension-packing.task.json",
        "steps": 2,
        "batch_size": 1,
        "kernel": "eager",
        "seed": 26082405,
    },
    "predecoded-video": {
        "task": "extension-video.task.json",
        "steps": 2,
        "batch_size": 1,
        "kernel": "auto",
        "seed": 26082406,
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _materialize_run(route: str, temporary_root: Path) -> Path:
    config = ROUTES[route]
    run = {
        "schema_version": 1,
        "name": f"real-vlm-extension-{route}",
        "seed": config["seed"],
        "deterministic": True,
        "device": "cuda:0",
        "precision": "bf16_true",
        "attention_kernel": config["kernel"],
        "max_steps": config["steps"],
        "per_device_batch_size": config["batch_size"],
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 5e-4,
            "weight_decay": 0.0,
            "foreach": False,
        },
        "scheduler": {"name": "constant", "warmup_steps": 0},
        "activation_checkpointing": {"enabled": False},
        "compile": {"enabled": False},
        "update_evidence": {
            "enabled": True,
            "every_steps": 1,
            "required_groups": ["connector"],
            "sample_elements_per_group": 8192,
        },
        "checkpoint": {
            "directory": str(temporary_root / "outputs" / "checkpoints"),
            "every_steps": config["steps"],
        },
    }
    path = temporary_root / "run.json"
    _write_json(path, run)
    return path


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source_root = str(ROOT.parent / "Framework" / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run_worker(phase: str, route: str, run_path: Path) -> dict[str, Any]:
    task_path = ROOT / ROUTES[route]["task"]
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            phase,
            "--route",
            route,
            "--task",
            str(task_path),
            "--run",
            str(run_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_worker_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{route}/{phase} failed\nstdout:\n{completed.stdout}"
            f"\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{route}/{phase} produced no receipt")
    return json.loads(lines[-1])


def _data_only_stream(task_path: Path):
    from trainomni.assembly.data_builder import build_data_stream
    from trainomni.assembly.preflight import preflight_task
    from trainomni.catalog.builtin import builtin_registry
    from trainomni.catalog.local import registry_for_task
    from trainomni.core.context import BuildContext
    from trainomni.core.resolver import ModuleResolver
    from trainomni.specs.loading import load_task

    task = load_task(task_path)
    registry = registry_for_task(
        builtin_registry(),
        task,
        task_root=task_path.parent,
        allow_local_code=True,
    )
    resolver = ModuleResolver(registry)
    report = preflight_task(task, resolver)
    context = BuildContext(task_digest=task.digest, task_root=task_path.parent)
    return task, report, build_data_stream(task.data, resolver, context=context), resolver


def _inspect(route: str, task_path: Path) -> dict[str, Any]:
    import torch

    task, report, stream, resolver = _data_only_stream(task_path)
    common: dict[str, Any] = {
        "task_digest": task.digest,
        "capabilities": sorted(report.capabilities.values),
    }
    if route == "weighted-mixture":
        batch = stream.next_batch(8)
        common.update(
            {
                "sample_ids": list(batch.sample_ids),
                "source_metrics": stream.metrics(),
            }
        )
    elif route == "sequence-packing":
        batch = stream.next_batch(1)
        segments = batch.model_inputs["packed_segment_ids"][0]
        block = batch.model_inputs["packed_attention_mask"][0, 0]
        valid = batch.model_inputs["attention_mask"][0].bool()
        segment_zero = (segments == 0) & valid
        segment_one = (segments == 1) & valid
        second_start = int(torch.nonzero(segment_one, as_tuple=False)[0].item())
        common.update(
            {
                "sample_ids": list(batch.sample_ids),
                "packed_lengths": batch.supervision["packed_lengths"][0].tolist(),
                "valid_tokens": int(valid.sum().item()),
                "segment_ids": segments[valid].tolist(),
                "second_segment_boundary_label": int(
                    batch.labels[0, second_start].item()
                ),
                "cross_segment_attention_entries": int(
                    block[segment_zero][:, segment_one].sum().item()
                    + block[segment_one][:, segment_zero].sum().item()
                ),
            }
        )

        from trainomni.assembly.preflight import preflight_task
        from trainomni.core.errors import CapabilityError
        from trainomni.core.module import ModuleId, ModuleRef

        incompatible = replace(
            task,
            model=replace(
                task.model,
                attention_policy=ModuleRef(
                    ModuleId.parse("attention_policy:trainomni/model_default@1")
                ),
            ),
        )
        try:
            preflight_task(incompatible, resolver)
        except CapabilityError as exc:
            common["incompatible_attention_fail_closed"] = str(exc)
        else:
            raise RuntimeError("packed task accepted a non-packed attention policy")
    elif route == "predecoded-video":
        batch = stream.next_batch(1)
        grids = batch.model_inputs["image_grid_thw"]
        common.update(
            {
                "sample_ids": list(batch.sample_ids),
                "frame_count": int(batch.model_inputs["image_counts"][0].item()),
                "encoded_grid_rows": int(grids.shape[0]),
                "image_grid_thw": grids.tolist(),
            }
        )
    else:
        batch = stream.next_batch(1)
        common["sample_ids"] = list(batch.sample_ids)
    return common


def _train(route: str, task_path: Path, run_path: Path) -> dict[str, Any]:
    from trainomni.api.train import train

    result = train(
        task_path=task_path,
        run_path=run_path,
        allow_local_code=True,
    )
    output_root = run_path.parent / "outputs"
    events = [
        json.loads(line)
        for line in (output_root / "metrics" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    initialized = next(event for event in events if event["event"] == "engine_initialized")
    final = asdict(result.records[-1])
    checkpoint = (
        output_root
        / "checkpoints"
        / f"step-{result.final_step:08d}"
    )
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    del result
    gc.collect()
    return {
        "engine": {
            "device": initialized["device"],
            "precision": initialized["precision"],
            "attention_kernel": initialized["attention_kernel"],
            "attention_kernel_modules": initialized.get("attention_kernel_modules", []),
        },
        "final_step": final,
        "checkpoint": {
            "path": str(checkpoint),
            "global_step": manifest["global_step"],
            "task_digest": manifest["task_digest"],
            "run_digest": manifest["run_digest"],
            "model_sha256": manifest["model_sha256"],
        },
    }


def _evaluate(route: str, task_path: Path, run_path: Path) -> dict[str, Any]:
    from trainomni.api.evaluate import evaluate

    steps = ROUTES[route]["steps"]
    checkpoint = run_path.parent / "outputs" / "checkpoints" / f"step-{steps:08d}"
    result = evaluate(
        task_path=task_path,
        run_path=run_path,
        checkpoint=checkpoint,
        batches=1,
        allow_local_code=True,
    )
    return {
        "batches": result.batches,
        "samples": result.samples,
        "metrics": result.metrics,
    }


def _worker(phase: str, route: str, task_path: Path, run_path: Path) -> None:
    if phase == "inspect":
        result = _inspect(route, task_path)
    elif phase == "train":
        result = _train(route, task_path, run_path)
    elif phase == "evaluate":
        result = _evaluate(route, task_path, run_path)
    else:
        raise RuntimeError(f"unknown worker phase: {phase}")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def verify_route(route: str) -> dict[str, Any]:
    if route not in ROUTES:
        raise ValueError(f"unknown route: {route}")
    with tempfile.TemporaryDirectory(prefix=f"trainomni-extension-{route}-") as raw:
        temporary_root = Path(raw).resolve()
        run_path = _materialize_run(route, temporary_root)
        inspection = _run_worker("inspect", route, run_path)
        training = _run_worker("train", route, run_path)
        evaluation = _run_worker("evaluate", route, run_path)
        final = training["final_step"]
        connector_evidence = final["parameter_evidence"]["connector"]
        if connector_evidence["changed_tensor_count"] <= 0 or (
            connector_evidence["before_sha256"]
            == connector_evidence["after_sha256"]
        ):
            raise RuntimeError(f"{route} produced no connector parameter update")
        if training["checkpoint"]["global_step"] != ROUTES[route]["steps"]:
            raise RuntimeError(f"{route} checkpoint step mismatch")
        receipt = {
            "schema_version": 1,
            "route": route,
            "task": str((ROOT / ROUTES[route]["task"]).resolve()),
            "inspection": inspection,
            "training": training,
            "held_out_evaluation": evaluation,
            "retention": {
                "large_temporary_checkpoint_removed_after_receipt": True,
                "durable_receipt_contains_model_digest": True,
            },
        }
        receipt["training"]["checkpoint"].pop("path", None)
        receipt_path = RECEIPT_ROOT / f"{route}.json"
        _write_json(receipt_path, receipt)
        return {
            "route": route,
            "receipt": str(receipt_path),
            "loss": final["loss"],
            "evaluation": evaluation["metrics"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", nargs="?", choices=tuple(ROUTES))
    parser.add_argument("--worker", choices=("inspect", "train", "evaluate"))
    parser.add_argument("--route", dest="worker_route", choices=tuple(ROUTES))
    parser.add_argument("--task", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        if args.worker_route is None or args.task is None or args.run is None:
            parser.error("--worker requires --route, --task, and --run")
        _worker(
            args.worker,
            args.worker_route,
            args.task.resolve(),
            args.run.resolve(),
        )
        return
    if args.route is None:
        parser.error("route is required")
    print(json.dumps(verify_route(args.route), sort_keys=True))


if __name__ == "__main__":
    main()
