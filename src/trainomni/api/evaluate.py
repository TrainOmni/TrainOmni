"""Stable held-out evaluation operation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainomni.core.errors import SpecError
from trainomni.runtime.data_loader import build_stateful_batch_loader
from trainomni.runtime.evaluation import evaluate_batches
from trainomni.runtime.random import seed_everything
from trainomni.specs.digest import canonical_value, identity_digest

from ._checkpoint import load_model_checkpoint
from .train import assemble, load_resolved_run


@dataclass(frozen=True, slots=True)
class EvaluateResult:
    checkpoint: Path
    batches: int
    samples: int
    metrics: dict[str, Any]
    receipt: Path


def evaluate(
    *,
    task_path: str | Path,
    run_path: str | Path,
    checkpoint: str | Path,
    batches: int,
    allow_local_code: bool = False,
) -> EvaluateResult:
    run = load_resolved_run(run_path)
    seed_everything(run.seed, deterministic=run.deterministic)
    task, assembly = assemble(
        task_path=task_path,
        allow_local_code=allow_local_code,
        operation="evaluate",
    )
    if assembly.evaluation_stream is None or not assembly.evaluators:
        raise SpecError("task does not define an evaluation data path and evaluators")
    model, execution_model, device, checkpoint_path, manifest = load_model_checkpoint(
        task=task,
        assembly=assembly,
        run=run,
        checkpoint=checkpoint,
        restore_objective=True,
    )
    evaluation_stream = build_stateful_batch_loader(
        assembly.evaluation_stream,
        batch_size=run.per_device_batch_size,
        spec=run.data_loader,
    )
    try:
        result = evaluate_batches(
            model=model,
            objective=assembly.objective,
            stream=evaluation_stream,
            evaluators=assembly.evaluators,
            device=device,
            batches=batches,
            batch_size=run.per_device_batch_size,
            execution_model=execution_model,
        )
    finally:
        close_stream = getattr(evaluation_stream, "close", None)
        if callable(close_stream):
            close_stream()
    evaluation_config = {
        "schema_version": 1,
        "checkpoint": {
            "framework_version": manifest.framework_version,
            "task_digest": manifest.task_digest,
            "training_run_digest": manifest.run_digest,
            "module_lock": dict(sorted(manifest.module_lock.items())),
            "global_step": manifest.global_step,
            "model_sha256": manifest.model_sha256,
            "runtime_sha256": manifest.runtime_sha256,
        },
        "execution": {
            "seed": run.seed,
            "deterministic": run.deterministic,
            "device": run.device,
            "precision": run.precision,
            "attention_kernel": run.attention_kernel,
            "compile": canonical_value(run.compile),
            "per_device_batch_size": run.per_device_batch_size,
            "data_loader": canonical_value(run.data_loader),
            "batches": batches,
        },
    }
    evaluation_digest = identity_digest(evaluation_config)
    receipt = (
        run.checkpoint.directory.parent
        / "evaluations"
        / manifest.model_sha256
        / f"{evaluation_digest}.json"
    )
    payload = {
        "schema_version": 2,
        "evaluation_digest": evaluation_digest,
        "configuration": evaluation_config,
        "samples": result.samples,
        "metrics": result.metrics,
    }
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if receipt.exists() and receipt.read_text(encoding="utf-8") != content:
        raise SpecError(f"evaluation receipt already differs: {receipt}")
    if not receipt.exists():
        receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt.with_name(f".{receipt.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, receipt)
    return EvaluateResult(
        checkpoint=checkpoint_path,
        batches=result.batches,
        samples=result.samples,
        metrics=result.metrics,
        receipt=receipt,
    )
