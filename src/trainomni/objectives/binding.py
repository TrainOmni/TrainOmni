"""Resolve semantic sample objectives to concrete loss/algorithm plugins."""

from __future__ import annotations

from dataclasses import dataclass

from trainomni.config import StageSpec

from .protocol import Objective
from .registry import ObjectiveRegistry, ObjectiveRegistryError

DEFAULT_OBJECTIVE_IMPLEMENTATIONS = {
    "cpt": "masked-causal-lm",
    "sft": "masked-causal-lm",
}


@dataclass(frozen=True, slots=True)
class ObjectiveBinding:
    sample_objective: str
    implementation_id: str
    objective: Objective


def resolve_objective(
    stage: StageSpec, registry: ObjectiveRegistry
) -> ObjectiveBinding:
    implementation_id = stage.objective_impl or DEFAULT_OBJECTIVE_IMPLEMENTATIONS.get(
        stage.objective
    )
    if implementation_id is None:
        raise ObjectiveRegistryError(
            f"sample objective {stage.objective!r} requires an explicit objective_impl"
        )
    objective = registry.get(implementation_id)
    requirements = objective.manifest.requirements
    if stage.objective not in requirements.sample_objectives:
        raise ObjectiveRegistryError(
            f"objective implementation {implementation_id!r} does not support "
            f"sample objective {stage.objective!r}"
        )
    if stage.engine.backend not in objective.manifest.supported_engines:
        raise ObjectiveRegistryError(
            f"objective implementation {implementation_id!r} does not support "
            f"engine {stage.engine.backend!r}"
        )
    missing_modalities = requirements.required_modalities - stage.data.modalities
    missing_blocks = requirements.required_content_blocks - stage.data.content_blocks
    if missing_modalities or missing_blocks:
        raise ObjectiveRegistryError(
            f"objective implementation {implementation_id!r} requirements are unmet: "
            f"modalities={sorted(missing_modalities)}, "
            f"content_blocks={sorted(missing_blocks)}"
        )
    return ObjectiveBinding(stage.objective, implementation_id, objective)
