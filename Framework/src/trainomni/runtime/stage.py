"""Assemble registered data/model/objective/engine pieces for one stage."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainomni.config import ResolvedRunSpec
from trainomni.contracts import ArtifactRef
from trainomni.data import (
    DistributedBatchStream,
    ImporterRegistry,
    MixtureStream,
    ReaderRegistry,
    StatefulBatchStream,
    batch_budget_from_data,
    open_dataset_streams,
)
from trainomni.engines import (
    DelegatedStageContext,
    EngineKind,
    EngineRegistry,
    EngineRequirements,
    StageResult,
    TorchStageContext,
    negotiate_engine,
)
from trainomni.models import ModelBuildContext, ModelBundle
from trainomni.objectives import ObjectiveRegistry, resolve_objective

from .logging import JsonlRunLogger
from .provenance import write_provenance
from .seed import seed_everything


class StageExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StageRunRequest:
    resolved: ResolvedRunSpec
    plugin: Any
    output_dir: Path
    input_artifacts: Mapping[str, ArtifactRef]
    resume_from: str | None = None
    trusted_resume: bool = False
    trusted_input_artifacts: bool = False
    readers: ReaderRegistry | None = None
    importers: ImporterRegistry | None = None


def execute_stage(
    request: StageRunRequest,
    *,
    objectives: ObjectiveRegistry | None = None,
    engines: EngineRegistry | None = None,
) -> StageResult:
    """Execute a resolved stage with no model-family branches in the core."""

    objective_registry = objectives or ObjectiveRegistry()
    engine_registry = engines or EngineRegistry()
    run = request.resolved.run
    stage = run.stage
    binding = resolve_objective(stage, objective_registry)
    engine = engine_registry.get(stage.engine.backend)
    requirements = binding.objective.manifest.requirements
    report = negotiate_engine(
        EngineRequirements(
            stage_type=stage.stage_type,
            objective=binding.implementation_id,
            parallelism=stage.engine.parallelism,
            precision=stage.engine.precision,
            resume_level=stage.checkpoint.resume_level,
            require_generation=requirements.requires_rollout,
            require_multiple_models=(
                requirements.requires_reference_model
                or requirements.requires_teacher_model
            ),
            require_rollout=requirements.requires_rollout,
        ),
        engine.manifest.capabilities,
    )
    if not report.valid:
        details = "; ".join(f"{item.code}: {item.message}" for item in report.issues)
        raise StageExecutionError(f"engine capability negotiation failed: {details}")
    output_dir = request.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(run.seed, stage.engine.config)
    is_primary = int(os.environ.get("RANK", "0")) == 0
    if is_primary:
        write_provenance(request.resolved, output_dir)
    if engine.manifest.kind is EngineKind.DELEGATED_STAGE:
        validation = engine.validate(stage, None)
        if not validation.valid:
            details = "; ".join(
                f"{item.code}: {item.message}" for item in validation.issues
            )
            raise StageExecutionError(f"delegated engine validation failed: {details}")
        context = DelegatedStageContext(
            stage_id=stage.stage_id,
            output_dir=output_dir,
            config=stage.engine.config,
            request_payload=_redact(
                {
                    "resolved": request.resolved.to_dict(),
                    "input_artifacts": {
                        key: {
                            "artifact_id": value.artifact_id,
                            "selector": value.selector,
                            "uri": value.uri,
                        }
                        for key, value in request.input_artifacts.items()
                    },
                }
            ),
        )
        result = engine.collect(engine.run(engine.prepare(context)))
        if is_primary:
            _write_run_manifest(
                request,
                engine.manifest.engine_version,
                binding.implementation_id,
                result,
            )
        return result
    build_context = ModelBuildContext(
        config=run.model.config,
        stage_id=stage.stage_id,
        output_dir=output_dir,
        input_artifacts=request.input_artifacts,
    )
    bundle = request.plugin.build(build_context)
    if not isinstance(bundle, ModelBundle):
        raise StageExecutionError(
            f"plugin.build() must return ModelBundle for execution, "
            f"got {type(bundle).__name__}"
        )
    model_input = request.input_artifacts.get("model") or request.input_artifacts.get(
        "checkpoint"
    )
    if model_input is not None and model_input.uri is not None:
        from .export import load_checkpoint_weights

        load_checkpoint_weights(
            Path(model_input.uri),
            bundle,
            trusted_checkpoint=request.trusted_input_artifacts,
        )
    streams = open_dataset_streams(
        stage.data.datasets,
        source_config=request.resolved.source,
        readers=request.readers,
        importers=request.importers,
    )
    mixture = MixtureStream(
        streams,
        seed=run.seed,
        repeat=bool(stage.data.config.get("repeat", True)),
    )
    batches = StatefulBatchStream(
        mixture,
        plugin=request.plugin,
        sample_objective=stage.objective,
        stage_id=stage.stage_id,
        budget=batch_budget_from_data(stage.data),
        packing=stage.data.packing,
        data_spec=stage.data,
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        batches = DistributedBatchStream(
            batches,
            rank=int(os.environ.get("RANK", "0")),
            world_size=world_size,
        )
    if stage.engine.backend != "torch":
        raise StageExecutionError(
            f"engine {stage.engine.backend!r} requires a registered context factory"
        )
    context = TorchStageContext(
        resolved=request.resolved,
        plugin=request.plugin,
        bundle=bundle,
        objective=binding,
        batches=batches,
        output_dir=output_dir,
        resume_from=request.resume_from,
        trusted_resume=request.trusted_resume,
        callbacks=(JsonlRunLogger(output_dir / "metrics.jsonl"),) if is_primary else (),
    )
    prepared = engine.prepare(context)
    result = engine.collect(engine.run(prepared))
    if is_primary:
        _write_run_manifest(
            request, engine.manifest.engine_version, binding.implementation_id, result
        )
    return result


def _write_run_manifest(
    request: StageRunRequest,
    engine_version: str,
    objective_impl: str,
    result: StageResult,
) -> None:
    payload = {
        "schema_version": "trainomni.executed-stage.v1",
        "run": request.resolved.run.name,
        "stage_id": result.stage_id,
        "run_fingerprint": request.resolved.fingerprint,
        "plugin": {
            "id": request.resolved.plugin_manifest.plugin_id,
            "version": request.resolved.plugin_manifest.plugin_version,
        },
        "engine": {
            "id": request.resolved.run.stage.engine.backend,
            "version": engine_version,
        },
        "objective_impl": objective_impl,
        "status": result.status,
        "metrics": dict(result.metrics),
        "outputs": {
            key: {"ref": str(value), "uri": value.uri}
            for key, value in result.outputs.items()
        },
        "input_artifacts": {
            key: str(value) for key, value in request.input_artifacts.items()
        },
    }
    target = request.output_dir.resolve() / "run-manifest.json"
    temporary = target.with_suffix(f".tmp-{os.getpid()}.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _redact(value: Any, key: str = "") -> Any:
    sensitive = {"token", "password", "secret", "api_key", "environment"}
    if any(item in key.lower() for item in sensitive):
        return "***"
    if isinstance(value, Mapping):
        return {str(name): _redact(item, str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, key) for item in value]
    return value
