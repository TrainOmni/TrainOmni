"""Objective registry with built-in masked causal language modeling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainomni.models import ModelBatch

from .protocol import (
    LossOutput,
    LossTerm,
    Objective,
    ObjectiveManifest,
    ObjectiveRequirements,
)


class ObjectiveRegistryError(ValueError):
    pass


class ObjectiveRegistry:
    def __init__(self, *, include_builtins: bool = True) -> None:
        self._objectives: dict[str, Objective] = {}
        if include_builtins:
            self.register(MaskedCausalLMObjective())
            self.register(
                DelegatedObjective(
                    "dpo",
                    ObjectiveRequirements(
                        sample_objectives=frozenset({"preference"}),
                        requires_reference_model=True,
                    ),
                    supported_engines=frozenset(
                        {"delegated", "trl", "verl", "nemo", "veomni"}
                    ),
                )
            )
            self.register(
                DelegatedObjective(
                    "distillation",
                    ObjectiveRequirements(
                        sample_objectives=frozenset({"sft"}),
                        requires_teacher_model=True,
                    ),
                    supported_engines=frozenset(
                        {"delegated", "trl", "nemo", "veomni"}
                    ),
                )
            )
            self.register(
                DelegatedObjective(
                    "grpo",
                    ObjectiveRequirements(
                        sample_objectives=frozenset({"prompt_only"}),
                        requires_reward_provider=True,
                        requires_rollout=True,
                    ),
                    supported_engines=frozenset(
                        {"delegated", "trl", "verl", "nemo"}
                    ),
                )
            )
            self.register(
                DelegatedObjective(
                    "ppo",
                    ObjectiveRequirements(
                        sample_objectives=frozenset({"prompt_only"}),
                        requires_reference_model=True,
                        requires_reward_provider=True,
                        requires_rollout=True,
                    ),
                    supported_engines=frozenset(
                        {"delegated", "trl", "verl", "nemo"}
                    ),
                )
            )

    def register(self, objective: Objective) -> None:
        manifest = getattr(objective, "manifest", None)
        if not isinstance(manifest, ObjectiveManifest):
            raise ObjectiveRegistryError("objective must define ObjectiveManifest")
        if not callable(getattr(objective, "prepare", None)) or not callable(
            getattr(objective, "compute", None)
        ):
            raise ObjectiveRegistryError("objective must implement prepare()/compute()")
        if manifest.objective_id in self._objectives:
            raise ObjectiveRegistryError(
                f"objective {manifest.objective_id!r} is already registered"
            )
        self._objectives[manifest.objective_id] = objective

    def get(self, objective_id: str) -> Objective:
        try:
            return self._objectives[objective_id]
        except KeyError as exc:
            raise ObjectiveRegistryError(
                f"unknown objective {objective_id!r}; available: {sorted(self._objectives)}"
            ) from exc

    def manifests(self) -> tuple[ObjectiveManifest, ...]:
        return tuple(
            self._objectives[key].manifest for key in sorted(self._objectives)
        )


class MaskedCausalLMObjective:
    manifest = ObjectiveManifest(
        objective_id="masked-causal-lm",
        objective_version="1.0.0",
        requirements=ObjectiveRequirements(
            sample_objectives=frozenset({"cpt", "sft"})
        ),
        supported_engines=frozenset(
            {"torch", "delegated", "trl", "nemo", "veomni"}
        ),
    )

    def prepare(self, batch: ModelBatch, context: Any) -> ModelBatch:
        if "labels" not in batch.model_inputs:
            raise ObjectiveRegistryError(
                "masked causal LM requires model batch field 'labels'"
            )
        return batch

    def compute(self, models: Any, batch: ModelBatch) -> LossOutput:
        model = models.get("model") if isinstance(models, Mapping) else models
        if model is None or not callable(model):
            raise ObjectiveRegistryError("masked causal LM requires a callable model")
        output = model(**dict(batch.model_inputs))
        loss = getattr(output, "loss", None)
        if loss is None and isinstance(output, Mapping):
            loss = output.get("loss")
        if loss is None:
            raise ObjectiveRegistryError(
                "model output must expose pre-normalized .loss or ['loss']"
            )
        denominator = _loss_denominator(batch.model_inputs["labels"])
        scalar = _safe_scalar(loss)
        return LossOutput(
            total=loss,
            terms={"token_ce": LossTerm(value=loss, denominator=denominator)},
            metrics={"loss": scalar} if scalar is not None else {},
            counts={"loss_tokens": denominator},
        )


class DelegatedObjective:
    """Static algorithm requirements for a backend-owned training stage."""

    def __init__(
        self,
        objective_id: str,
        requirements: ObjectiveRequirements,
        *,
        supported_engines: frozenset[str],
    ) -> None:
        self.manifest = ObjectiveManifest(
            objective_id=objective_id,
            objective_version="1.0.0",
            requirements=requirements,
            supported_engines=supported_engines,
        )

    def prepare(self, batch: Any, context: Any) -> Any:
        raise ObjectiveRegistryError(
            f"objective {self.manifest.objective_id!r} is owned by its delegated engine"
        )

    def compute(self, models: Any, batch: Any) -> LossOutput:
        raise ObjectiveRegistryError(
            f"objective {self.manifest.objective_id!r} is owned by its delegated engine"
        )


def _loss_denominator(labels: Any) -> int:
    """Count labels not equal to the conventional ignore index (-100)."""

    if hasattr(labels, "ne") and callable(labels.ne):
        count = labels.ne(-100).sum()
        if hasattr(count, "item"):
            return max(1, int(count.item()))
    if isinstance(labels, (list, tuple)):
        count = 0
        stack = list(labels)
        while stack:
            value = stack.pop()
            if isinstance(value, (list, tuple)):
                stack.extend(value)
            elif value != -100:
                count += 1
        return max(1, count)
    return 1


def _safe_scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    detached = value.detach() if hasattr(value, "detach") else value
    if hasattr(detached, "item"):
        try:
            return float(detached.item())
        except (TypeError, ValueError):
            return None
    return None
