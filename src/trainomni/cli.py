"""TrainOmni control-plane CLI.

M1 intentionally exposes validation and inspection before a training command.
External Python plugins are loaded only through explicit ``--plugin`` flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from trainomni.config import (
    ConfigLoadError,
    ResolvedRunSpec,
    load_run_spec,
    resolve_run,
)
from trainomni.contracts import ArtifactRef, BatchBudget, ValidationReport
from trainomni.data import (
    BatchPlanningError,
    DataImportError,
    DataReadError,
    GreedyBatchPlanner,
    SampleValidationError,
    inspect_imported_sample,
    open_dataset_streams,
    take_round_robin,
    validate_sample_against_data,
)
from trainomni.models import (
    EncodedSample,
    ModelBatch,
    inspect_encoded_sample,
    inspect_model_batch,
)
from trainomni.recipes import load_pipeline_spec, resolve_pipeline
from trainomni.registry import (
    DataRegistries,
    ModelPluginRegistry,
    PluginRegistryError,
    load_data_plugins,
)
from trainomni.runtime import (
    PipelineExecutor,
    StageRunRequest,
    evaluate_run,
    execute_stage,
    export_model,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainomni",
        description="Composable, VLM-first omni-modal training control plane",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE:ATTR|FILE.py:ATTR",
        help="explicitly trust and load a model plugin; repeatable",
    )
    parser.add_argument(
        "--data-plugin",
        action="append",
        default=[],
        metavar="MODULE:ATTR|FILE.py:ATTR",
        help="explicitly trust and load a reader/importer plugin; repeatable",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate and resolve a run spec")
    validate.add_argument("config", type=Path)

    plugins = commands.add_parser("plugins", help="inspect model plugin availability")
    plugin_commands = plugins.add_subparsers(dest="plugins_command", required=True)
    plugin_commands.add_parser(
        "list", help="list explicitly loaded and installed entry-point plugins"
    )

    inspect = commands.add_parser("inspect", help="inspect resolved framework state")
    inspect_commands = inspect.add_subparsers(dest="inspect_command", required=True)
    inspect_model = inspect_commands.add_parser(
        "model", help="inspect model manifest, capabilities and component policy"
    )
    inspect_model.add_argument("config", type=Path)
    inspect_data = inspect_commands.add_parser(
        "data", help="read, import and inspect canonical samples"
    )
    inspect_data.add_argument("config", type=Path)
    inspect_data.add_argument("--samples", type=int, default=3)
    inspect_data.add_argument(
        "--include-canonical",
        action="store_true",
        help="include the complete normalized canonical sample",
    )
    inspect_batch = inspect_commands.add_parser(
        "batch", help="encode, plan and collate samples without loading weights"
    )
    inspect_batch.add_argument("config", type=Path)
    inspect_batch.add_argument("--samples", type=int, default=2)

    dry_run = commands.add_parser(
        "dry-run", help="resolve and print a run plan without loading model weights"
    )
    dry_run.add_argument("config", type=Path)

    plan = commands.add_parser("plan", help="validate and topologically plan a pipeline DAG")
    plan.add_argument("pipeline", type=Path)

    train = commands.add_parser("train", help="execute one resolved training stage")
    train.add_argument("config", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--resume", metavar="CHECKPOINT_PATH_OR_NAME")
    train.add_argument(
        "--trusted-resume",
        action="store_true",
        help="allow trusted local pickle checkpoint deserialization",
    )

    run_pipeline = commands.add_parser("run", help="execute a resolved pipeline DAG")
    run_pipeline.add_argument("pipeline", type=Path)
    run_pipeline.add_argument("--output-dir", type=Path, required=True)
    run_pipeline.add_argument("--resume", action="store_true")
    run_pipeline.add_argument(
        "--trusted-resume",
        action="store_true",
        help="trust exact checkpoints referenced by a resumed pipeline state",
    )

    export = commands.add_parser("export", help="export a checkpoint through the model plugin")
    export.add_argument("config", type=Path)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--format", default="hf")
    export.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help="allow trusted local pickle checkpoint deserialization",
    )

    evaluate = commands.add_parser("evaluate", help="evaluate a model through a registered provider")
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--evaluator", default="loss")
    evaluate.add_argument("--max-batches", type=int, default=100)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help="allow trusted local pickle checkpoint deserialization",
    )
    return parser


def _registry(plugin_specs: Sequence[str]) -> ModelPluginRegistry:
    registry = ModelPluginRegistry()
    for specification in plugin_specs:
        registry.load_explicit(specification, allow_external=True)
    return registry


def _resolve_config(
    config: Path, registry: ModelPluginRegistry
) -> tuple[ResolvedRunSpec | None, ValidationReport]:
    spec = load_run_spec(config)
    record = registry.get(spec.model.plugin)
    return resolve_run(spec, record.manifest, source=config)


def _format_report(report: ValidationReport) -> str:
    if report.valid and not report.issues:
        return "valid"
    lines = []
    for issue in report.issues:
        location = f" ({issue.path})" if issue.path else ""
        lines.append(
            f"{issue.severity.value.upper()} {issue.code}{location}: {issue.message}"
        )
    return "\n".join(lines)


def _emit(value: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    elif isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _is_primary_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _plugins_payload(registry: ModelPluginRegistry) -> dict[str, object]:
    return {
        "loaded": [record.to_dict() for record in registry.records()],
        "installed_not_loaded": [
            candidate.to_dict() for candidate in registry.entry_point_candidates()
        ],
        "security": (
            "installed entry points are listed without import; use --plugin or an "
            "explicit future trust policy to execute external code"
        ),
    }


def _model_payload(resolved: ResolvedRunSpec) -> dict[str, object]:
    manifest = resolved.plugin_manifest
    return {
        "plugin": {
            "plugin_id": manifest.plugin_id,
            "plugin_version": manifest.plugin_version,
            "api_version": manifest.api_version,
            "source_patterns": list(manifest.model_patterns),
            "component_ids": list(manifest.component_ids),
            "requires_remote_code": manifest.requires_remote_code,
        },
        "capabilities": {
            "modalities": sorted(manifest.capabilities.modalities),
            "content_blocks": sorted(manifest.capabilities.content_blocks),
            "objectives": sorted(manifest.capabilities.objectives),
            "parallelism": sorted(manifest.capabilities.parallelism),
            "engine_backends": sorted(manifest.capabilities.engine_backends),
            "packing": manifest.capabilities.supports_packing,
            "padding_free": manifest.capabilities.supports_padding_free,
            "generation": manifest.capabilities.supports_generation,
        },
        "requested_component_policy": {
            component: policy.model_dump(mode="json")
            for component, policy in sorted(
                resolved.run.stage.component_policy.items()
            )
        },
        "note": "static inspection only; no model weights were loaded",
    }


def _dry_run_payload(resolved: ResolvedRunSpec) -> dict[str, object]:
    run = resolved.run
    payload = resolved.to_dict()
    payload["plan"] = {
        "stage_id": run.stage.stage_id,
        "stage_type": run.stage.stage_type,
        "objective": run.stage.objective,
        "datasets": [item.dataset_id for item in run.stage.data.datasets],
        "engine": run.stage.engine.backend,
        "parallelism": run.stage.engine.parallelism,
        "precision": run.stage.engine.precision,
        "checkpoint_resume_level": run.stage.checkpoint.resume_level,
        "will_load_weights": False,
        "will_execute_training": False,
    }
    return payload


def _artifact_inputs(values: dict[str, str]) -> dict[str, ArtifactRef]:
    result = {}
    for slot, value in values.items():
        if value.startswith("artifact://"):
            body = value[len("artifact://") :]
            artifact_id, separator, selector = body.rpartition("/")
            result[slot] = (
                ArtifactRef(artifact_id, selector)
                if separator and artifact_id
                else ArtifactRef(body)
            )
        else:
            result[slot] = ArtifactRef(value)
    return result


def _data_payload(
    resolved: ResolvedRunSpec,
    *,
    limit: int,
    include_canonical: bool,
    registries: DataRegistries,
) -> dict[str, object]:
    streams = open_dataset_streams(
        resolved.run.stage.data.datasets,
        source_config=resolved.source,
        readers=registries.readers,
        importers=registries.importers,
    )
    imported = take_round_robin(streams, limit)
    return {
        "run": resolved.run.name,
        "fingerprint": resolved.fingerprint,
        "requested_samples": limit,
        "returned_samples": len(imported),
        "datasets": [stream.spec.dataset_id for stream in streams],
        "samples": [
            inspect_imported_sample(item, include_canonical=include_canonical)
            for item in imported
        ],
    }


def _batch_budget(resolved: ResolvedRunSpec, limit: int) -> BatchBudget:
    config = resolved.run.stage.data.config
    allowed = {
        "max_samples",
        "max_text_tokens",
        "max_vision_tokens",
        "max_pixels",
        "max_frames",
        "max_audio_seconds",
        "max_model_units",
        "repeat",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown data batch config fields: {sorted(unknown)}")
    return BatchBudget(max_samples=config.get("max_samples", limit), **{
        key: value for key, value in config.items() if key not in {"max_samples", "repeat"}
    })


def _batch_payload(
    resolved: ResolvedRunSpec,
    plugin: Any,
    *,
    limit: int,
    registries: DataRegistries,
) -> dict[str, object]:
    streams = open_dataset_streams(
        resolved.run.stage.data.datasets,
        source_config=resolved.source,
        readers=registries.readers,
        importers=registries.importers,
    )
    imported = take_round_robin(streams, limit)
    if not imported:
        raise ValueError("no samples were available for batch inspection")
    encoded: list[EncodedSample] = []
    for item in imported:
        validate_sample_against_data(
            item.sample, resolved.run.stage.data, resolved.run.stage.objective
        )
        sample_issues = plugin.validate_sample(
            item.sample, resolved.run.stage.objective
        )
        if sample_issues:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in sample_issues
            )
            raise ValueError(f"model plugin rejected sample {item.sample.id!r}: {details}")
        value = plugin.encode(
            item.sample,
            {
                "stage_id": resolved.run.stage.stage_id,
                "objective": resolved.run.stage.objective,
                "inspect": True,
                "source_trace": item.trace.to_dict(),
            },
        )
        if not isinstance(value, EncodedSample):
            raise TypeError(
                f"plugin.encode() must return EncodedSample, got {type(value).__name__}"
            )
        if value.sample_id != item.sample.id:
            raise ValueError("plugin.encode() changed the canonical sample ID")
        encoded.append(value)

    planner = GreedyBatchPlanner(
        _batch_budget(resolved, limit), packing=resolved.run.stage.data.packing
    )
    plans = planner.plan(encoded)
    by_id = {sample.sample_id: sample for sample in encoded}
    batches = []
    for plan in plans:
        members = [by_id[item.sample_id] for item in plan.items]
        batch = plugin.collate(members, plan)
        if not isinstance(batch, ModelBatch):
            raise TypeError(
                f"plugin.collate() must return ModelBatch, got {type(batch).__name__}"
            )
        batches.append(batch)
    return {
        "run": resolved.run.name,
        "fingerprint": resolved.fingerprint,
        "encoded": [inspect_encoded_sample(item) for item in encoded],
        "batches": [inspect_model_batch(batch) for batch in batches],
        "will_load_weights": False,
        "will_execute_training": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        registry = _registry(args.plugin)
        data_registries = load_data_plugins(args.data_plugin)
        if args.command == "plugins":
            _emit(_plugins_payload(registry), as_json=args.json)
            return 0

        if args.command == "plan":
            pipeline = load_pipeline_spec(args.pipeline)
            record = registry.get(pipeline.model.plugin)
            resolved_pipeline, pipeline_report = resolve_pipeline(
                pipeline, record.manifest, source=args.pipeline
            )
            if resolved_pipeline is None:
                if args.json:
                    _emit(pipeline_report.to_dict(), as_json=True)
                else:
                    print(_format_report(pipeline_report), file=sys.stderr)
                return 2
            _emit(resolved_pipeline.to_dict(), as_json=args.json)
            return 0

        if args.command == "run":
            pipeline = load_pipeline_spec(args.pipeline)
            record = registry.get(pipeline.model.plugin)
            resolved_pipeline, pipeline_report = resolve_pipeline(
                pipeline, record.manifest, source=args.pipeline
            )
            if resolved_pipeline is None:
                if args.json:
                    _emit(pipeline_report.to_dict(), as_json=True)
                else:
                    print(_format_report(pipeline_report), file=sys.stderr)
                return 2
            result = PipelineExecutor(
                resolved_pipeline,
                plugin=record.plugin,
                output_dir=args.output_dir,
                readers=data_registries.readers,
                importers=data_registries.importers,
            ).run(resume=args.resume, trusted_resume=args.trusted_resume)
            if _is_primary_process():
                _emit(result.to_dict(), as_json=args.json)
            return 0

        resolved, report = _resolve_config(args.config, registry)
        if resolved is None:
            if args.json:
                _emit(report.to_dict(), as_json=True)
            else:
                print(_format_report(report), file=sys.stderr)
            return 2

        if args.command == "validate":
            payload = {
                "valid": True,
                "fingerprint": resolved.fingerprint,
                "run": resolved.run.name,
                "plugin": resolved.plugin_manifest.plugin_id,
                "stage": resolved.run.stage.stage_id,
                "issues": [],
            }
            _emit(payload if args.json else (
                f"valid: {resolved.run.name}\n"
                f"fingerprint: {resolved.fingerprint}\n"
                f"plugin: {resolved.plugin_manifest.plugin_id}\n"
                f"stage: {resolved.run.stage.stage_id}"
            ), as_json=args.json)
        elif args.command == "inspect" and args.inspect_command == "model":
            _emit(_model_payload(resolved), as_json=args.json)
        elif args.command == "inspect" and args.inspect_command == "data":
            _emit(
                _data_payload(
                    resolved,
                    limit=args.samples,
                    include_canonical=args.include_canonical,
                    registries=data_registries,
                ),
                as_json=args.json,
            )
        elif args.command == "inspect" and args.inspect_command == "batch":
            record = registry.get(resolved.run.model.plugin)
            _emit(
                _batch_payload(
                    resolved,
                    record.plugin,
                    limit=args.samples,
                    registries=data_registries,
                ),
                as_json=args.json,
            )
        elif args.command == "dry-run":
            _emit(_dry_run_payload(resolved), as_json=args.json)
        elif args.command == "train":
            record = registry.get(resolved.run.model.plugin)
            result = execute_stage(
                StageRunRequest(
                    resolved=resolved,
                    plugin=record.plugin,
                    output_dir=args.output_dir,
                    input_artifacts=_artifact_inputs(resolved.run.stage.inputs),
                    resume_from=args.resume,
                    trusted_resume=args.trusted_resume,
                    readers=data_registries.readers,
                    importers=data_registries.importers,
                )
            )
            if _is_primary_process():
                _emit(
                    {
                        "stage_id": result.stage_id,
                        "status": result.status,
                        "metrics": dict(result.metrics),
                        "outputs": {
                            key: {"ref": str(value), "uri": value.uri}
                            for key, value in result.outputs.items()
                        },
                    },
                    as_json=args.json,
                )
        elif args.command == "export":
            record = registry.get(resolved.run.model.plugin)
            payload = export_model(
                resolved,
                record.plugin,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                export_format=args.format,
                trusted_checkpoint=args.trusted_checkpoint,
            )
            _emit(payload, as_json=args.json)
        elif args.command == "evaluate":
            record = registry.get(resolved.run.model.plugin)
            payload = evaluate_run(
                resolved,
                record.plugin,
                output_dir=args.output_dir,
                evaluator_id=args.evaluator,
                max_batches=args.max_batches,
                checkpoint=args.checkpoint,
                trusted_checkpoint=args.trusted_checkpoint,
                readers=data_registries.readers,
                importers=data_registries.importers,
            )
            _emit(payload, as_json=args.json)
        else:  # pragma: no cover - argparse prevents this branch
            raise AssertionError(f"unhandled command: {args.command}")
        return 0
    except (
        ConfigLoadError,
        PluginRegistryError,
        DataImportError,
        DataReadError,
        BatchPlanningError,
        SampleValidationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        if args.json:
            _emit(
                {
                    "valid": False,
                    "issues": [
                        {
                            "code": "cli.error",
                            "severity": "error",
                            "message": str(exc),
                        }
                    ],
                },
                as_json=True,
            )
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
