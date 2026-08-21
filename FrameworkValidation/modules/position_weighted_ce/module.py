"""A task-local objective proving that loss semantics are externally replaceable."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.forward import (
    ForwardPlan,
    ForwardRequest,
    ForwardResult,
    OutputRequirements,
)
from trainomni.contracts.loss import LossBundle, LossTerm
from trainomni.core.capability import CapabilitySet
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import ObjectiveError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.objectives.ops import causal_shift, token_cross_entropy
from trainomni.modules.objectives.protocol import ObjectiveRequirements

from .config import PositionWeightedCEConfig


class PositionWeightedCEObjective:
    """FP32 causal CE with deterministic absolute-position weights."""

    def __init__(self, config: PositionWeightedCEConfig) -> None:
        self.config = config

    def requirements(self) -> ObjectiveRequirements:
        return ObjectiveRequirements(outputs=OutputRequirements(logits=True))

    def plan(self, batch: OmniBatch, context: ObjectiveContext) -> ForwardPlan:
        return ForwardPlan.single(
            ForwardRequest(
                name="policy",
                inputs=batch.model_inputs,
                outputs=self.requirements().outputs,
                requires_grad=context.training,
            )
        )

    def compute(
        self,
        batch: OmniBatch,
        outputs: Mapping[str, ForwardResult],
        context: ObjectiveContext,
    ) -> LossBundle:
        del context
        if "policy" not in outputs:
            raise ObjectiveError("position-weighted CE requires the 'policy' forward")
        logits = outputs["policy"].require("logits")
        shifted_logits, shifted_labels = causal_shift(logits, batch.labels)
        mask = shifted_labels.ne(self.config.ignore_index)
        supervised_tokens = mask.sum()
        if supervised_tokens.item() == 0:
            raise ObjectiveError("position-weighted CE requires supervised target tokens")

        token_losses = token_cross_entropy(
            shifted_logits,
            shifted_labels,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.label_smoothing,
        )
        position_weights = torch.linspace(
            1.0,
            self.config.final_token_weight,
            shifted_labels.shape[1],
            dtype=torch.float32,
            device=shifted_labels.device,
        ).unsqueeze(0)
        effective_weights = position_weights.expand_as(token_losses) * mask.float()
        numerator = (token_losses.float() * effective_weights).sum()
        denominator = effective_weights.sum()
        total = numerator / denominator
        return LossBundle(
            total=total,
            terms={
                "position_weighted_ce": LossTerm(
                    value=total,
                    weight=1.0,
                    numerator=numerator,
                    denominator=denominator,
                )
            },
            metrics={
                "supervised_tokens": supervised_tokens.detach(),
                "effective_token_weight": (
                    denominator / supervised_tokens.float()
                ).detach(),
            },
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ObjectiveError("position-weighted CE objective has no mutable state")


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("objective:validation/position_weighted_ce@1"),
        config_type=PositionWeightedCEConfig,
        factory=lambda config, context: PositionWeightedCEObjective(config),
        provides=CapabilitySet.of(
            {"objective.position_weighted_ce", "objective.loss_bundle"}
        ),
        requires=CapabilitySet.of({"model.output.logits", "batch.labels"}),
    )
