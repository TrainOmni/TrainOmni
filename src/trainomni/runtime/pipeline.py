"""Sequential DAG executor with durable state, lineage, and explicit gates."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainomni.config import RunSpec, resolve_run
from trainomni.contracts import ArtifactManifest, ArtifactRef
from trainomni.data import ImporterRegistry, ReaderRegistry
from trainomni.recipes import (
    ArtifactCatalog,
    PipelineRuntimeState,
    ResolvedPipeline,
    evaluate_gate,
)

from .stage import StageRunRequest, execute_stage

PIPELINE_RUN_STATE_VERSION = "trainomni.pipeline-run-state.v1"


class PipelineExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    name: str
    fingerprint: str
    statuses: Mapping[str, str]
    outputs: Mapping[str, Mapping[str, ArtifactRef]]
    metrics: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fingerprint": self.fingerprint,
            "statuses": dict(self.statuses),
            "outputs": {
                stage: {
                    slot: {"ref": str(ref), "uri": ref.uri}
                    for slot, ref in values.items()
                }
                for stage, values in self.outputs.items()
            },
            "metrics": {
                stage: dict(values) for stage, values in self.metrics.items()
            },
        }


class PipelineExecutor:
    def __init__(
        self,
        resolved: ResolvedPipeline,
        *,
        plugin: Any,
        output_dir: Path,
        readers: ReaderRegistry | None = None,
        importers: ImporterRegistry | None = None,
    ) -> None:
        self.resolved = resolved
        self.plugin = plugin
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "pipeline-state.json"
        self.catalog = ArtifactCatalog()
        self.readers = readers
        self.importers = importers

    def run(
        self, *, resume: bool = False, trusted_resume: bool = False
    ) -> PipelineRunResult:
        # A PipelineExecutor can be reused after success or failure. Rebuild the
        # in-memory catalog from durable state so registration stays idempotent.
        self.catalog = ArtifactCatalog()
        runtime, outputs, metrics = self._load_state() if resume else (
            PipelineRuntimeState.initial(self.resolved.spec),
            {},
            {},
        )
        stage_by_id = {stage.stage_id: stage for stage in self.resolved.spec.stages}
        trusted_ids = (
            {ref.artifact_id for values in outputs.values() for ref in values.values()}
            if resume and trusted_resume
            else set()
        )
        while True:
            ready = runtime.ready_stages(self.resolved.spec)
            if not ready:
                break
            stage_id = ready[0]
            stage = stage_by_id[stage_id]
            runtime = runtime.transition(stage_id, "running")
            self._save_state(runtime, outputs, metrics)
            input_artifacts = self._stage_inputs(stage_id, stage.inputs, outputs)
            run = RunSpec(
                name=f"{self.resolved.spec.name}/{stage_id}",
                seed=self.resolved.spec.seed,
                model=self.resolved.spec.model,
                stage=stage,
                metadata=self.resolved.spec.metadata,
            )
            resolved_stage, report = resolve_run(
                run,
                self.plugin.manifest,
                source=self.resolved.source,
            )
            if resolved_stage is None:
                details = "; ".join(item.message for item in report.issues)
                raise PipelineExecutionError(
                    f"stage {stage_id!r} no longer resolves: {details}"
                )
            try:
                result = execute_stage(
                    StageRunRequest(
                        resolved=resolved_stage,
                        plugin=self.plugin,
                        output_dir=self.output_dir / stage_id,
                        input_artifacts=input_artifacts,
                        trusted_input_artifacts=all(
                            ref.artifact_id in trusted_ids
                            for ref in input_artifacts.values()
                        ),
                        readers=self.readers,
                        importers=self.importers,
                    )
                )
                outputs[stage_id] = dict(result.outputs)
                trusted_ids.update(ref.artifact_id for ref in result.outputs.values())
                metrics[stage_id] = dict(result.metrics)
                self._register_outputs(
                    stage_id,
                    resolved_stage.fingerprint,
                    stage.checkpoint.resume_level,
                    input_artifacts,
                    result.outputs,
                )
                gate_results = [
                    evaluate_gate(
                        gate,
                        metrics=result.metrics,
                        artifacts={ref.artifact_id for ref in result.outputs.values()},
                    )
                    for gate in stage.gates
                ]
                failed = [item.message for item in gate_results if not item.passed]
                if failed:
                    raise PipelineExecutionError(
                        f"stage {stage_id!r} failed gates: {failed}"
                    )
                runtime = runtime.transition(stage_id, "succeeded")
            except Exception:
                runtime = runtime.transition(stage_id, "failed")
                self._save_state(runtime, outputs, metrics)
                raise
            self._save_state(runtime, outputs, metrics)
        unfinished = {
            stage: status
            for stage, status in runtime.statuses.items()
            if status not in {"succeeded", "skipped"}
        }
        if unfinished:
            raise PipelineExecutionError(f"pipeline stopped with unfinished stages: {unfinished}")
        return PipelineRunResult(
            name=self.resolved.spec.name,
            fingerprint=self.resolved.fingerprint,
            statuses=dict(runtime.statuses),
            outputs=outputs,
            metrics=metrics,
        )

    def _stage_inputs(
        self,
        stage_id: str,
        declared: Mapping[str, str],
        outputs: Mapping[str, Mapping[str, ArtifactRef]],
    ) -> dict[str, ArtifactRef]:
        values = {slot: _parse_artifact_ref(value) for slot, value in declared.items()}
        for edge in self.resolved.spec.edges:
            if edge.to_stage != stage_id:
                continue
            source = outputs.get(edge.from_stage, {})
            if not source:
                raise PipelineExecutionError(
                    f"stage {stage_id!r} dependency {edge.from_stage!r} has no outputs"
                )
            reference = source.get("checkpoint") or next(iter(source.values()))
            if edge.input_slot in values and values[edge.input_slot] != reference:
                raise PipelineExecutionError(
                    f"stage input slot {edge.input_slot!r} has conflicting sources"
                )
            values[edge.input_slot] = ArtifactRef(
                reference.artifact_id,
                selector=edge.selector,
                uri=reference.uri,
            )
        return values

    def _register_outputs(
        self,
        stage_id: str,
        fingerprint: str,
        resume_level: str,
        inputs: Mapping[str, ArtifactRef],
        outputs: Mapping[str, ArtifactRef],
    ) -> None:
        for reference in inputs.values():
            if reference.artifact_id not in self.catalog.ids():
                self.catalog.register(
                    ArtifactManifest(
                        artifact_id=reference.artifact_id,
                        artifact_type="external_input",
                        run_id=self.resolved.spec.name,
                        stage_id="external",
                        fingerprint="external",
                        resume_level="transfer",
                        metadata={"selector": reference.selector},
                    )
                )
        for reference in outputs.values():
            self.catalog.register(
                ArtifactManifest(
                    artifact_id=reference.artifact_id,
                    artifact_type="checkpoint",
                    run_id=self.resolved.spec.name,
                    stage_id=stage_id,
                    fingerprint=fingerprint,
                    resume_level=resume_level,
                    parents=tuple(inputs.values()),
                    metadata={"selector": reference.selector},
                )
            )

    def _save_state(
        self,
        runtime: PipelineRuntimeState,
        outputs: Mapping[str, Mapping[str, ArtifactRef]],
        metrics: Mapping[str, Mapping[str, float]],
    ) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return
        payload = {
            "schema_version": PIPELINE_RUN_STATE_VERSION,
            "pipeline_fingerprint": self.resolved.fingerprint,
            "statuses": dict(runtime.statuses),
            "outputs": {
                stage: {
                    slot: {
                        "artifact_id": ref.artifact_id,
                        "selector": ref.selector,
                        "uri": ref.uri,
                    }
                    for slot, ref in values.items()
                }
                for stage, values in outputs.items()
            },
            "metrics": {stage: dict(values) for stage, values in metrics.items()},
        }
        temporary = self.state_path.with_suffix(f".tmp-{os.getpid()}.json")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _load_state(
        self,
    ) -> tuple[
        PipelineRuntimeState,
        dict[str, dict[str, ArtifactRef]],
        dict[str, dict[str, float]],
    ]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineExecutionError(
                f"cannot load pipeline state {self.state_path}"
            ) from exc
        if payload.get("schema_version") != PIPELINE_RUN_STATE_VERSION:
            raise PipelineExecutionError("pipeline state version mismatch")
        if payload.get("pipeline_fingerprint") != self.resolved.fingerprint:
            raise PipelineExecutionError("pipeline state fingerprint mismatch")
        statuses = payload.get("statuses")
        if not isinstance(statuses, Mapping):
            raise PipelineExecutionError("pipeline statuses are invalid")
        expected = {stage.stage_id for stage in self.resolved.spec.stages}
        if set(statuses) != expected:
            raise PipelineExecutionError("pipeline state stage set mismatch")
        normalized_statuses = dict(statuses)
        # A process may have died after persisting "running". Retry that stage.
        for stage, status in normalized_statuses.items():
            if status == "running" or status == "failed":
                normalized_statuses[stage] = "pending"
        raw_outputs = payload.get("outputs", {})
        outputs = {
            stage: {
                slot: ArtifactRef(
                    value["artifact_id"],
                    value.get("selector", "last"),
                    value.get("uri"),
                )
                for slot, value in values.items()
            }
            for stage, values in raw_outputs.items()
        }
        metrics = {
            stage: {key: float(value) for key, value in values.items()}
            for stage, values in payload.get("metrics", {}).items()
        }
        # Recreate lineage for already completed stages in topological order.
        stage_by_id = {stage.stage_id: stage for stage in self.resolved.spec.stages}
        for stage_id in self.resolved.order:
            if normalized_statuses.get(stage_id) != "succeeded":
                continue
            inputs = self._stage_inputs(stage_id, stage_by_id[stage_id].inputs, outputs)
            self._register_outputs(
                stage_id,
                self.resolved.stage_fingerprints[stage_id],
                stage_by_id[stage_id].checkpoint.resume_level,
                inputs,
                outputs.get(stage_id, {}),
            )
        return PipelineRuntimeState(normalized_statuses), outputs, metrics


def _parse_artifact_ref(value: str) -> ArtifactRef:
    if value.startswith("artifact://"):
        body = value[len("artifact://") :]
        artifact_id, separator, selector = body.rpartition("/")
        if separator and artifact_id:
            return ArtifactRef(artifact_id, selector)
    return ArtifactRef(value)
