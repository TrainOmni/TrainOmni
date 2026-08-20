"""Evaluation runtime assembly and durable result manifest."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trainomni.config import ResolvedRunSpec
from trainomni.data import (
    ImporterRegistry,
    MixtureStream,
    ReaderRegistry,
    StatefulBatchStream,
    batch_budget_from_data,
    open_dataset_streams,
)
from trainomni.evaluation import EvaluationRequest, EvaluatorRegistry
from trainomni.models import ModelBuildContext, ModelBundle
from trainomni.objectives import ObjectiveRegistry, resolve_objective

from .export import load_checkpoint_weights
from .seed import seed_everything


def evaluate_run(
    resolved: ResolvedRunSpec,
    plugin: Any,
    *,
    output_dir: Path,
    evaluator_id: str = "loss",
    max_batches: int = 100,
    evaluator_config: Mapping[str, Any] | None = None,
    checkpoint: Path | None = None,
    trusted_checkpoint: bool = False,
    readers: ReaderRegistry | None = None,
    importers: ImporterRegistry | None = None,
) -> Mapping[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(resolved.run.seed, resolved.run.stage.engine.config)
    bundle = plugin.build(
        ModelBuildContext(
            config=resolved.run.model.config,
            stage_id=resolved.run.stage.stage_id,
            output_dir=output_dir,
            mode="evaluate",
        )
    )
    if not isinstance(bundle, ModelBundle):
        raise TypeError("plugin.build() must return ModelBundle for evaluation")
    if checkpoint is not None:
        load_checkpoint_weights(
            checkpoint.resolve(), bundle, trusted_checkpoint=trusted_checkpoint
        )
    binding = resolve_objective(resolved.run.stage, ObjectiveRegistry())
    streams = open_dataset_streams(
        resolved.run.stage.data.datasets,
        source_config=resolved.source,
        readers=readers,
        importers=importers,
    )
    batches = StatefulBatchStream(
        MixtureStream(streams, seed=resolved.run.seed, repeat=False),
        plugin=plugin,
        sample_objective=resolved.run.stage.objective,
        stage_id=resolved.run.stage.stage_id,
        budget=batch_budget_from_data(resolved.run.stage.data),
        packing=resolved.run.stage.data.packing,
        data_spec=resolved.run.stage.data,
    )
    evaluator = EvaluatorRegistry().get(evaluator_id)
    result = evaluator.evaluate(
        EvaluationRequest(
            run_name=resolved.run.name,
            model_bundle=bundle,
            batches=batches,
            objective=binding.objective,
            output_dir=output_dir,
            config={"max_batches": max_batches, **dict(evaluator_config or {})},
        )
    )
    payload = {
        "schema_version": "trainomni.evaluation.v1",
        "run_fingerprint": resolved.fingerprint,
        "evaluator": result.evaluator_id,
        "metrics": dict(result.metrics),
        "counts": dict(result.counts),
        "artifacts": dict(result.artifacts),
    }
    target = output_dir / "evaluation.json"
    temporary = target.with_suffix(f".tmp-{os.getpid()}.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return payload
