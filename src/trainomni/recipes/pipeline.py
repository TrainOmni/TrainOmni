"""Typed pipeline DAG, deterministic planning and stage readiness."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trainomni.config import (
    ModelSpec,
    RunSpec,
    StageSpec,
    canonical_fingerprint,
    resolve_run,
)
from trainomni.contracts import ValidationIssue, ValidationReport
from trainomni.models import ModelPluginManifest

PIPELINE_SCHEMA_VERSION = "trainomni.pipeline.v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StageEdge(_StrictModel):
    from_stage: str = Field(min_length=1)
    to_stage: str = Field(min_length=1)
    input_slot: str = Field(default="model", min_length=1)
    selector: Literal["last", "best", "approved"] = "last"


class PipelineSpec(_StrictModel):
    schema_version: Literal["trainomni.pipeline.v1"] = PIPELINE_SCHEMA_VERSION
    name: str = Field(min_length=1)
    seed: int = Field(default=0, ge=0)
    model: ModelSpec
    stages: tuple[StageSpec, ...]
    edges: tuple[StageEdge, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> PipelineSpec:
        if not self.stages:
            raise ValueError("pipeline requires at least one stage")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("pipeline stage IDs must be unique")
        known = set(stage_ids)
        edge_keys = set()
        for edge in self.edges:
            if edge.from_stage not in known or edge.to_stage not in known:
                raise ValueError(
                    f"pipeline edge references unknown stage: "
                    f"{edge.from_stage!r} -> {edge.to_stage!r}"
                )
            if edge.from_stage == edge.to_stage:
                raise ValueError("pipeline stage cannot depend on itself")
            key = (edge.from_stage, edge.to_stage, edge.input_slot)
            if key in edge_keys:
                raise ValueError(f"duplicate pipeline edge {key}")
            edge_keys.add(key)
        topological_order(self)
        return self


def topological_order(spec: PipelineSpec) -> tuple[str, ...]:
    stage_order = [stage.stage_id for stage in spec.stages]
    order_index = {stage_id: index for index, stage_id in enumerate(stage_order)}
    incoming = {stage_id: 0 for stage_id in stage_order}
    outgoing: dict[str, list[str]] = {stage_id: [] for stage_id in stage_order}
    for edge in spec.edges:
        incoming[edge.to_stage] += 1
        outgoing[edge.from_stage].append(edge.to_stage)
    ready = sorted(
        (stage_id for stage_id, count in incoming.items() if count == 0),
        key=order_index.__getitem__,
    )
    result = []
    while ready:
        stage_id = ready.pop(0)
        result.append(stage_id)
        for child in sorted(outgoing[stage_id], key=order_index.__getitem__):
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
                ready.sort(key=order_index.__getitem__)
    if len(result) != len(stage_order):
        cyclic = sorted(stage_id for stage_id, count in incoming.items() if count > 0)
        raise ValueError(f"pipeline graph contains a cycle involving {cyclic}")
    return tuple(result)


def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    source = Path(path).resolve()
    try:
        raw = source.read_text(encoding="utf-8")
        value = json.loads(raw) if source.suffix.lower() == ".json" else yaml.safe_load(raw)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse pipeline {source}: {exc}") from exc
    return PipelineSpec.model_validate(value)


@dataclass(frozen=True, slots=True)
class ResolvedPipeline:
    spec: PipelineSpec
    order: tuple[str, ...]
    stage_fingerprints: Mapping[str, str]
    fingerprint: str
    source: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "name": self.spec.name,
            "fingerprint": self.fingerprint,
            "source": str(self.source) if self.source else None,
            "order": list(self.order),
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "stage_type": stage.stage_type,
                    "objective": stage.objective,
                    "fingerprint": self.stage_fingerprints[stage.stage_id],
                    "engine": stage.engine.backend,
                    "parallelism": stage.engine.parallelism,
                }
                for stage in self.spec.stages
            ],
            "edges": [edge.model_dump(mode="json") for edge in self.spec.edges],
        }


def resolve_pipeline(
    spec: PipelineSpec,
    manifest: ModelPluginManifest,
    *,
    source: str | Path | None = None,
) -> tuple[ResolvedPipeline | None, ValidationReport]:
    issues: list[ValidationIssue] = []
    stage_fingerprints = {}
    for stage in spec.stages:
        run = RunSpec(
            name=f"{spec.name}/{stage.stage_id}",
            seed=spec.seed,
            model=spec.model,
            stage=stage,
            metadata=spec.metadata,
        )
        resolved, report = resolve_run(run, manifest, source=source)
        for issue in report.issues:
            issues.append(
                ValidationIssue(
                    code=issue.code,
                    message=issue.message,
                    severity=issue.severity,
                    path=(
                        f"stages.{stage.stage_id}.{issue.path}"
                        if issue.path
                        else f"stages.{stage.stage_id}"
                    ),
                    source=issue.source,
                    hint=issue.hint,
                )
            )
        if resolved is not None:
            stage_fingerprints[stage.stage_id] = resolved.fingerprint
    report = ValidationReport(tuple(issues))
    if not report.valid:
        return None, report
    order = topological_order(spec)
    identity = {
        "pipeline": spec,
        "plugin": {
            "id": manifest.plugin_id,
            "version": manifest.plugin_version,
            "api": manifest.api_version,
        },
        "stage_fingerprints": stage_fingerprints,
    }
    digest = canonical_fingerprint(identity)
    return (
        ResolvedPipeline(
            spec=spec,
            order=order,
            stage_fingerprints=stage_fingerprints,
            fingerprint=digest,
            source=Path(source).resolve() if source else None,
        ),
        report,
    )


@dataclass(frozen=True, slots=True)
class PipelineRuntimeState:
    statuses: Mapping[str, str]

    @classmethod
    def initial(cls, spec: PipelineSpec) -> PipelineRuntimeState:
        return cls({stage.stage_id: "pending" for stage in spec.stages})

    def ready_stages(self, spec: PipelineSpec) -> tuple[str, ...]:
        dependencies: dict[str, set[str]] = {
            stage.stage_id: set() for stage in spec.stages
        }
        for edge in spec.edges:
            dependencies[edge.to_stage].add(edge.from_stage)
        return tuple(
            stage_id
            for stage_id in topological_order(spec)
            if self.statuses.get(stage_id) == "pending"
            and all(self.statuses.get(parent) == "succeeded" for parent in dependencies[stage_id])
        )

    def transition(self, stage_id: str, status: str) -> PipelineRuntimeState:
        allowed = {
            "pending": {"running", "skipped"},
            "running": {"succeeded", "failed"},
            "failed": {"running"},
            "succeeded": set(),
            "skipped": set(),
        }
        current = self.statuses.get(stage_id)
        if current is None:
            raise ValueError(f"unknown stage {stage_id!r}")
        if status not in allowed[current]:
            raise ValueError(f"invalid stage transition {current!r} -> {status!r}")
        updated = dict(self.statuses)
        updated[stage_id] = status
        return PipelineRuntimeState(updated)
