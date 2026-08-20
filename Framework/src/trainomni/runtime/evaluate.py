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
from trainomni.engines.torch_engine import (
    configure_torch_precision,
    import_torch,
    move_model_batch,
    prepare_models_for_evaluation,
    select_torch_device,
    torch_autocast_context,
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
    execution = {
        "backend": resolved.run.stage.engine.backend,
        "precision": resolved.run.stage.engine.precision,
        "device": None,
        "mode": "delegated" if evaluator.manifest.delegated else "local",
    }
    request_batches: Any = batches
    if evaluator.manifest.delegated:
        result = evaluator.evaluate(
            _evaluation_request(
                resolved,
                bundle,
                request_batches,
                binding.objective,
                output_dir,
                max_batches,
                evaluator_config,
            )
        )
    else:
        if resolved.run.stage.engine.backend != "torch":
            raise RuntimeError(
                "local evaluation requires stage.engine.backend='torch'; "
                "use a delegated evaluator for backend-owned execution"
            )
        torch = import_torch()
        device = select_torch_device(torch, resolved.run.stage.engine.config)
        precision = resolved.run.stage.engine.precision
        configure_torch_precision(torch, precision, device)
        prepare_models_for_evaluation(bundle, device)
        request_batches = _batches_on_device(batches, device)
        execution["device"] = str(device)
        with torch.inference_mode(), torch_autocast_context(
            torch, precision, device
        ):
            result = evaluator.evaluate(
                _evaluation_request(
                    resolved,
                    bundle,
                    request_batches,
                    binding.objective,
                    output_dir,
                    max_batches,
                    evaluator_config,
                )
            )
    payload = {
        "schema_version": "trainomni.evaluation.v1",
        "run_fingerprint": resolved.fingerprint,
        "evaluator": result.evaluator_id,
        "execution": execution,
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


def _evaluation_request(
    resolved: ResolvedRunSpec,
    bundle: ModelBundle,
    batches: Any,
    objective: Any,
    output_dir: Path,
    max_batches: int,
    evaluator_config: Mapping[str, Any] | None,
) -> EvaluationRequest:
    return EvaluationRequest(
        run_name=resolved.run.name,
        model_bundle=bundle,
        batches=batches,
        objective=objective,
        output_dir=output_dir,
        config={"max_batches": max_batches, **dict(evaluator_config or {})},
    )


def _batches_on_device(batches: Any, device: Any) -> Any:
    for batch in batches:
        yield move_model_batch(batch, device)
