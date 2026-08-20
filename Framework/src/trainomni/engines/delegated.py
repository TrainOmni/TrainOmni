"""Shell-free delegated adapter for VeOmni, TRL, NeMo, veRL, and custom launchers."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainomni.contracts import ArtifactRef, ValidationIssue, ValidationReport

from .protocol import (
    EngineCapabilities,
    EngineKind,
    EngineManifest,
    PreparedStage,
    StageResult,
)


class DelegatedEngineError(RuntimeError):
    pass


VEOMNI_BRIDGE_API_VERSION = "trainomni.veomni-bridge.v1"


@dataclass(frozen=True, slots=True)
class DelegatedStageContext:
    stage_id: str
    output_dir: Path
    config: Mapping[str, Any]
    request_payload: Mapping[str, Any]


_ALL_STAGE_TYPES = frozenset(
    {
        "vision_preparation",
        "modality_alignment",
        "multimodal_pretraining",
        "capability_curriculum",
        "instruction_sft",
        "reasoning_distillation",
        "reward_verifier",
        "offline_preference",
        "online_rl",
        "agentic_rl",
        "evaluate_export",
    }
)
_OBJECTIVES = frozenset(
    {"masked-causal-lm", "dpo", "distillation", "grpo", "ppo"}
)


class DelegatedCommandEngine:
    def __init__(self, engine_id: str = "delegated") -> None:
        self.manifest = EngineManifest(
            engine_id=engine_id,
            engine_version="1.0.0",
            kind=EngineKind.DELEGATED_STAGE,
            capabilities=EngineCapabilities(
                stage_types=_ALL_STAGE_TYPES,
                objectives=_OBJECTIVES,
                parallelism=frozenset(
                    {"single", "ddp", "fsdp2", "tensor_parallel", "pipeline_parallel"}
                ),
                precisions=frozenset({"fp32", "tf32", "fp16", "bf16", "fp8"}),
                resume_levels=frozenset(
                    {"exact", "stage_boundary", "weights_only", "transfer"}
                ),
                supports_generation=True,
                supports_multiple_models=True,
                supports_rollout=True,
            ),
        )

    def validate(self, stage: Any, model: Any) -> ValidationReport:
        issues = []
        config = stage.engine.config
        argv = config.get("argv")
        if not isinstance(argv, (list, tuple)) or not argv or not all(
            isinstance(item, str) and item for item in argv
        ):
            issues.append(
                ValidationIssue(
                    code="engine.delegated.argv",
                    message="delegated engine requires a non-empty argv list",
                    path="stage.engine.config.argv",
                )
            )
        if config.get("allow_external_command") is not True:
            issues.append(
                ValidationIssue(
                    code="engine.delegated.trust",
                    message="delegated command was not explicitly allowed",
                    path="stage.engine.config.allow_external_command",
                )
            )
        return ValidationReport(tuple(issues))

    def prepare(self, context: DelegatedStageContext) -> PreparedStage:
        if not isinstance(context, DelegatedStageContext):
            raise DelegatedEngineError("delegated engine expects DelegatedStageContext")
        context.output_dir.mkdir(parents=True, exist_ok=True)
        request_path = context.output_dir / "delegated-request.json"
        request_path.write_text(
            json.dumps(
                context.request_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        return PreparedStage(context.stage_id, context)

    def run(self, prepared: PreparedStage) -> StageResult:
        context = _context(prepared)
        config = context.config
        argv = config["argv"]
        environment = os.environ.copy()
        extra_environment = config.get("environment", {})
        if not isinstance(extra_environment, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in extra_environment.items()
        ):
            raise DelegatedEngineError("delegated environment must map strings to strings")
        environment.update(extra_environment)
        environment["TRAINOMNI_STAGE_REQUEST"] = str(
            context.output_dir / "delegated-request.json"
        )
        completed = subprocess.run(
            list(argv),
            cwd=context.output_dir,
            env=environment,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.get("timeout_seconds", 86400),
        )
        (context.output_dir / "delegated.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (context.output_dir / "delegated.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            raise DelegatedEngineError(
                f"delegated stage exited with code {completed.returncode}"
            )
        result_path = context.output_dir / config.get(
            "result_json", "stage-result.json"
        )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DelegatedEngineError(
                f"delegated stage did not produce valid result JSON: {result_path}"
            ) from exc
        metrics = payload.get("metrics", {})
        outputs = payload.get("outputs", {})
        if not isinstance(metrics, Mapping) or not isinstance(outputs, Mapping):
            raise DelegatedEngineError("delegated result metrics/outputs must be mappings")
        references = {}
        for slot, value in outputs.items():
            if isinstance(value, str):
                references[str(slot)] = ArtifactRef(value)
            elif isinstance(value, Mapping):
                references[str(slot)] = ArtifactRef(
                    str(value["artifact_id"]),
                    str(value.get("selector", "last")),
                    str(value["uri"]) if value.get("uri") else None,
                )
            else:
                raise DelegatedEngineError("delegated output must be a string or mapping")
        return StageResult(
            stage_id=prepared.stage_id,
            status=str(payload.get("status", "succeeded")),
            outputs=references,
            metrics={str(key): float(value) for key, value in metrics.items()},
        )

    def checkpoint(self, prepared: PreparedStage, reason: str) -> ArtifactRef:
        context = _context(prepared)
        return ArtifactRef(
            f"{prepared.stage_id}/delegated",
            uri=str(context.output_dir.resolve()),
        )

    def collect(self, result: StageResult) -> StageResult:
        return result


class VeOmniCommandEngine(DelegatedCommandEngine):
    """Pinned, VLM-first command bridge to an external VeOmni environment.

    The bridge deliberately does not import VeOmni into the core environment.
    A versioned external launcher consumes ``TRAINOMNI_STAGE_REQUEST`` and must
    publish the ordinary delegated result contract.
    """

    def __init__(self) -> None:
        super().__init__("veomni")
        self.manifest = EngineManifest(
            engine_id="veomni",
            engine_version="1.0.0",
            kind=EngineKind.DELEGATED_STAGE,
            capabilities=EngineCapabilities(
                stage_types=frozenset(
                    {
                        "vision_preparation",
                        "modality_alignment",
                        "multimodal_pretraining",
                        "capability_curriculum",
                        "instruction_sft",
                        "reasoning_distillation",
                        "offline_preference",
                        "evaluate_export",
                    }
                ),
                objectives=frozenset(
                    {"masked-causal-lm", "dpo", "distillation"}
                ),
                parallelism=frozenset(
                    {
                        "single",
                        "ddp",
                        "fsdp2",
                        "tensor_parallel",
                        "pipeline_parallel",
                        "sequence_parallel",
                        "expert_parallel",
                    }
                ),
                precisions=frozenset({"fp32", "tf32", "fp16", "bf16", "fp8"}),
                # Exact runtime resume remains unclaimed until the bridge passes
                # TrainOmni data/RNG/topology conformance.
                resume_levels=frozenset(
                    {"stage_boundary", "weights_only", "transfer"}
                ),
                supports_multiple_models=True,
            ),
            dependency_constraints=("veomni==0.1.11",),
        )

    def validate(self, stage: Any, model: Any) -> ValidationReport:
        base = super().validate(stage, model)
        issues = list(base.issues)
        config = stage.engine.config
        revision = config.get("backend_revision")
        if not isinstance(revision, str) or not revision.strip():
            issues.append(
                ValidationIssue(
                    code="engine.veomni.revision",
                    message="VeOmni bridge requires a pinned backend_revision",
                    path="stage.engine.config.backend_revision",
                )
            )
        elif revision.strip().lower() in {"main", "master", "latest", "head"}:
            issues.append(
                ValidationIssue(
                    code="engine.veomni.unpinned_revision",
                    message="VeOmni backend_revision must be an immutable tag or commit",
                    path="stage.engine.config.backend_revision",
                )
            )
        if config.get("bridge_api") != VEOMNI_BRIDGE_API_VERSION:
            issues.append(
                ValidationIssue(
                    code="engine.veomni.bridge_api",
                    message=(
                        "VeOmni bridge_api must equal "
                        f"{VEOMNI_BRIDGE_API_VERSION!r}"
                    ),
                    path="stage.engine.config.bridge_api",
                )
            )
        return ValidationReport(tuple(issues))

    def prepare(self, context: DelegatedStageContext) -> PreparedStage:
        if not isinstance(context, DelegatedStageContext):
            raise DelegatedEngineError("VeOmni engine expects DelegatedStageContext")
        payload = dict(context.request_payload)
        payload["backend_contract"] = {
            "schema_version": VEOMNI_BRIDGE_API_VERSION,
            "engine": "veomni",
            "backend_revision": context.config["backend_revision"],
            "result_contract": "trainomni.delegated-result.v1",
        }
        enriched = DelegatedStageContext(
            stage_id=context.stage_id,
            output_dir=context.output_dir,
            config=context.config,
            request_payload=payload,
        )
        return super().prepare(enriched)


def _context(prepared: PreparedStage) -> DelegatedStageContext:
    if not isinstance(prepared.state, DelegatedStageContext):
        raise DelegatedEngineError("prepared stage does not belong to delegated engine")
    return prepared.state
