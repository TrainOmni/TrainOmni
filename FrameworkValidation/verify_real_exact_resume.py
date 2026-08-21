"""Fresh-process exact-resume verification for the five real VLM stages.

The parent process materializes one immutable Task/Run pair in a temporary
directory, executes an uninterrupted four-step reference, then reuses the same
Task/Run identity for a two-step checkpoint followed by a fresh-process resume
to step four.  Large checkpoints live only in the system temporary directory;
the durable result is a compact JSON receipt under ``resume-validation``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RECEIPT_ROOT = ROOT / "resume-validation"
BASE_ARTIFACT = ROOT / "artifacts" / "stage-04-kd"
BASE_ARTIFACT_DIGEST = (
    "b76286a5eabdc8ec88e70d4a1267f458531d65b97635da211157abf1a4ac5a98"
)

ROUTES: dict[str, dict[str, Any]] = {
    "full-sft": {
        "task": "stage-03-sft.task.json",
        "seed": 26082201,
        "learning_rate": 1e-3,
        "required_groups": ["vision_encoder", "connector", "language_model"],
        "activation_checkpointing": True,
        "base_override": True,
    },
    "pretraining": {
        "task": "stage-02-pretraining.task.json",
        "seed": 26082202,
        "learning_rate": 1e-3,
        "required_groups": ["vision_encoder", "connector", "language_model"],
        "activation_checkpointing": True,
        "base_override": True,
    },
    "alignment": {
        "task": "stage-01-alignment.task.json",
        "seed": 26082203,
        "learning_rate": 1e-3,
        "required_groups": ["connector"],
        "activation_checkpointing": False,
        "base_override": False,
    },
    "offline-kd": {
        "task": "stage-04-kd.task.json",
        "seed": 26082204,
        "learning_rate": 5e-4,
        "required_groups": ["connector"],
        "activation_checkpointing": False,
        "base_override": True,
    },
    "offline-dpo": {
        "task": "stage-05-dpo.task.json",
        "seed": 26082205,
        "learning_rate": 5e-4,
        "required_groups": ["connector"],
        "activation_checkpointing": False,
        "base_override": False,
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _materialize_specs(route: str, temporary_root: Path) -> tuple[Path, Path]:
    config = ROUTES[route]
    source_path = ROOT / config["task"]
    task = json.loads(source_path.read_text(encoding="utf-8"))
    if config["base_override"]:
        model_config = task["model"]["implementation"]["config"]
        model_config["initial_artifact"] = str(BASE_ARTIFACT)
        model_config["initial_artifact_sha256"] = BASE_ARTIFACT_DIGEST
    task["name"] = f"real-vlm-exact-resume-{route}"
    # Task-local code is intentionally confined to its Task root. Keep this
    # generated task beside the shared modules/data, while only large Run
    # outputs live in the system temporary directory.
    task_path = ROOT / f"resume-{route}.task.json"
    _write_json(task_path, task)

    activation_checkpointing = {"enabled": False}
    if config["activation_checkpointing"]:
        activation_checkpointing = {
            "enabled": True,
            "components": ["vision_encoder", "language_model"],
            "use_reentrant": False,
        }
    run = {
        "schema_version": 1,
        "name": f"real-vlm-exact-resume-{route}",
        "seed": config["seed"],
        "deterministic": True,
        "device": "cuda:0",
        "precision": "bf16_true",
        "attention_kernel": "auto",
        "max_steps": 4,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "optimizer": {
            "name": "adamw",
            "learning_rate": config["learning_rate"],
            "weight_decay": 0.0,
            "foreach": False,
        },
        "scheduler": {
            "name": "linear",
            "warmup_steps": 0,
            "min_lr_ratio": 0.1,
        },
        "activation_checkpointing": activation_checkpointing,
        "compile": {"enabled": False},
        "update_evidence": {
            "enabled": True,
            "every_steps": 1,
            "required_groups": config["required_groups"],
            "sample_elements_per_group": 8192,
        },
        "checkpoint": {
            "directory": str(temporary_root / "outputs" / "checkpoints"),
            "every_steps": 4,
        },
    }
    run_path = temporary_root / "run.json"
    _write_json(run_path, run)
    return task_path, run_path


def _last_optimizer_step(events_path: Path) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    steps = [event for event in events if event.get("event") == "optimizer_step"]
    if not steps:
        raise RuntimeError(f"no optimizer_step records in {events_path}")
    return steps[-1]


def _state_receipt(
    temporary_root: Path, *, logical_state: dict[str, str]
) -> dict[str, Any]:
    checkpoint = temporary_root / "outputs" / "checkpoints" / "step-00000004"
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    step = _last_optimizer_step(temporary_root / "outputs" / "metrics" / "events.jsonl")
    stable_step = {
        key: step[key]
        for key in (
            "global_step",
            "loss",
            "grad_norm",
            "micro_batches",
            "learning_rate",
            "loss_terms",
            "objective_metrics",
            "data_metrics",
            "parameter_evidence",
        )
    }
    return {
        "checkpoint": {
            "global_step": manifest["global_step"],
            "logical_model_sha256": logical_state["logical_model_sha256"],
            "logical_optimizer_sha256": logical_state["logical_optimizer_sha256"],
            "runtime_sha256": manifest["runtime_sha256"],
            "runtime_metadata": manifest["runtime_metadata"],
        },
        "container_digests": {
            "model_sha256": manifest["model_sha256"],
            "optimizer_sha256": manifest["optimizer_sha256"],
        },
        "final_step": stable_step,
    }


def _clean_outputs(temporary_root: Path) -> None:
    output_root = (temporary_root / "outputs").resolve()
    expected_parent = temporary_root.resolve()
    if output_root.parent != expected_parent:
        raise RuntimeError("refusing to clean outputs outside the temporary root")
    if output_root.exists():
        shutil.rmtree(output_root)


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source_root = str(ROOT.parent / "Framework" / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _update_tensor_digest(digest: Any, tensor: Any) -> None:
    import torch

    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _update_value_digest(digest: Any, value: Any) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        digest.update(b"tensor:")
        _update_tensor_digest(digest, value)
        return
    if isinstance(value, dict):
        digest.update(b"mapping:")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_value_digest(digest, key)
            _update_value_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"sequence:{type(value).__name__}:".encode())
        for item in value:
            _update_value_digest(digest, item)
        return
    digest.update(
        json.dumps(
            {"type": type(value).__name__, "value": value},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _logical_checkpoint_digests(checkpoint: Path) -> dict[str, str]:
    import torch
    from safetensors import safe_open

    model_digest = hashlib.sha256()
    with safe_open(
        checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as source:
        for name in sorted(source.keys()):
            model_digest.update(name.encode("utf-8"))
            _update_tensor_digest(model_digest, source.get_tensor(name))

    optimizer_payload = torch.load(
        checkpoint / "optimizer.pt", map_location="cpu", weights_only=False
    )
    optimizer_digest = hashlib.sha256()
    _update_value_digest(optimizer_digest, optimizer_payload["optimizer"])
    return {
        "logical_model_sha256": model_digest.hexdigest(),
        "logical_optimizer_sha256": optimizer_digest.hexdigest(),
    }


def _run_worker(phase: str, task_path: Path, run_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            phase,
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
            f"{phase} worker failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{phase} worker produced no receipt")
    return json.loads(lines[-1])


def _worker(phase: str, task_path: Path, run_path: Path) -> None:
    from trainomni.api.evaluate import evaluate
    from trainomni.api.train import assemble, build_engine, load_resolved_run, train
    from trainomni.runtime.random import seed_everything

    if phase == "uninterrupted":
        result = train(
            task_path=task_path,
            run_path=run_path,
            allow_local_code=True,
        )
        print(json.dumps({"final_step": result.final_step}, sort_keys=True))
        return
    if phase == "split":
        result = train(
            task_path=task_path,
            run_path=run_path,
            allow_local_code=True,
            stop_after_steps=2,
        )
        print(json.dumps({"final_step": result.final_step}, sort_keys=True))
        return
    run = load_resolved_run(run_path)
    checkpoint = run.checkpoint.directory / "step-00000002"
    if phase == "resume":
        seed_everything(run.seed, deterministic=run.deterministic)
        task, assembly = assemble(task_path=task_path, allow_local_code=True)
        engine = build_engine(task=task, assembly=assembly, run=run)
        engine.resume(checkpoint)
        if engine.global_step != 2:
            raise RuntimeError("resume did not restore global step 2")
        checkpoint = checkpoint.resolve()
        expected_parent = run.checkpoint.directory.resolve()
        if checkpoint.parent != expected_parent:
            raise RuntimeError("refusing to remove checkpoint outside the run root")
        shutil.rmtree(checkpoint)
        records = engine.train()
        print(
            json.dumps(
                {"final_step": engine.global_step, "steps_executed": len(records)},
                sort_keys=True,
            )
        )
        return
    if phase == "digest":
        final_checkpoint = run.checkpoint.directory / "step-00000004"
        print(json.dumps(_logical_checkpoint_digests(final_checkpoint), sort_keys=True))
        return
    if phase == "evaluate":
        final_checkpoint = run.checkpoint.directory / "step-00000004"
        result = evaluate(
            task_path=task_path,
            run_path=run_path,
            checkpoint=final_checkpoint,
            batches=1,
            allow_local_code=True,
        )
        print(
            json.dumps(
                {
                    "batches": result.batches,
                    "samples": result.samples,
                    "metrics": result.metrics,
                },
                sort_keys=True,
            )
        )
        return
    raise RuntimeError(f"unknown worker phase: {phase}")


def verify_route(route: str) -> dict[str, Any]:
    if route not in ROUTES:
        raise ValueError(f"unknown route: {route}")
    with tempfile.TemporaryDirectory(prefix=f"trainomni-resume-{route}-") as raw:
        temporary_root = Path(raw).resolve()
        task_path, run_path = _materialize_specs(route, temporary_root)

        uninterrupted_worker = _run_worker("uninterrupted", task_path, run_path)
        uninterrupted_logical = _run_worker("digest", task_path, run_path)
        uninterrupted = _state_receipt(
            temporary_root, logical_state=uninterrupted_logical
        )
        _clean_outputs(temporary_root)

        split_worker = _run_worker("split", task_path, run_path)
        resume_worker = _run_worker("resume", task_path, run_path)
        resumed_logical = _run_worker("digest", task_path, run_path)
        resumed = _state_receipt(temporary_root, logical_state=resumed_logical)
        evaluation = _run_worker("evaluate", task_path, run_path)

        checkpoint_equal = uninterrupted["checkpoint"] == resumed["checkpoint"]
        final_step_equal = uninterrupted["final_step"] == resumed["final_step"]
        if not checkpoint_equal or not final_step_equal:
            checkpoint_differences = {
                key: {
                    "uninterrupted": uninterrupted["checkpoint"].get(key),
                    "resumed": resumed["checkpoint"].get(key),
                }
                for key in sorted(
                    set(uninterrupted["checkpoint"]) | set(resumed["checkpoint"])
                )
                if uninterrupted["checkpoint"].get(key)
                != resumed["checkpoint"].get(key)
            }
            final_step_differences = {
                key: {
                    "uninterrupted": uninterrupted["final_step"].get(key),
                    "resumed": resumed["final_step"].get(key),
                }
                for key in sorted(
                    set(uninterrupted["final_step"]) | set(resumed["final_step"])
                )
                if uninterrupted["final_step"].get(key)
                != resumed["final_step"].get(key)
            }
            raise RuntimeError(
                json.dumps(
                    {
                        "checkpoint_equal": checkpoint_equal,
                        "final_step_equal": final_step_equal,
                        "checkpoint_differences": checkpoint_differences,
                        "final_step_differences": final_step_differences,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        receipt = {
            "schema_version": 1,
            "route": route,
            "comparison": {
                "uninterrupted_steps": 4,
                "split_steps": 2,
                "resumed_steps": 4,
                "fresh_process": True,
                "checkpoint_logical_state_equal": checkpoint_equal,
                "final_step_evidence_equal": final_step_equal,
                "container_file_digests_compared": False,
            },
            "workers": {
                "uninterrupted": uninterrupted_worker,
                "split": split_worker,
                "resume": resume_worker,
            },
            "state": resumed,
            "held_out_evaluation": evaluation,
            "retention": {
                "large_temporary_checkpoints_removed": True,
                "durable_checkpoint_digests_retained": True,
                "container_files_may_serialize_equal_state_differently": True,
            },
        }
        receipt_path = RECEIPT_ROOT / f"{route}.json"
        _write_json(receipt_path, receipt)
        return {"receipt": str(receipt_path), **receipt["comparison"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", nargs="?", choices=tuple(ROUTES))
    parser.add_argument(
        "--worker",
        choices=("uninterrupted", "split", "resume", "digest", "evaluate"),
    )
    parser.add_argument("--task", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        if args.task is None or args.run is None:
            parser.error("--worker requires --task and --run")
        _worker(args.worker, args.task.resolve(), args.run.resolve())
        return
    if args.route is None:
        parser.error("route is required")
    print(json.dumps(verify_route(args.route), sort_keys=True))


if __name__ == "__main__":
    main()
