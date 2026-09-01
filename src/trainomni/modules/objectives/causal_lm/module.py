"""Causal next-token objective."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.forward import (
    ForwardPlan,
    ForwardRequest,
    ForwardResult,
    OutputRequirements,
)
from trainomni.contracts.loss import LossBundle, LossTerm, ObjectiveMetric
from trainomni.core.capability import CapabilitySet
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import ObjectiveError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .._ops.causal_shift import causal_shift
from .._ops.reductions import reduce_token_losses
from .._ops.token_ce import token_cross_entropy
from ..protocol import ObjectiveRequirements
from .config import CausalLMConfig


class CausalLMObjective:
    def __init__(self, config: CausalLMConfig) -> None:
        self.config = config

    def requirements(self) -> ObjectiveRequirements:
        return ObjectiveRequirements(
            outputs=OutputRequirements(logits=True),
            metric_aggregations=(("supervised_tokens", "sum"),),
        )

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
            raise ObjectiveError("causal LM objective requires the 'policy' forward")
        logits = outputs["policy"].require("logits")
        shifted_logits, shifted_labels = causal_shift(logits, batch.labels)
        mask = shifted_labels.ne(self.config.ignore_index)
        token_losses = token_cross_entropy(
            shifted_logits,
            shifted_labels,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.label_smoothing,
        )
        total, numerator, denominator = reduce_token_losses(
            token_losses, mask, reduction=self.config.reduction
        )
        return LossBundle(
            total=total,
            terms={
                "token_ce": LossTerm(
                    value=total,
                    weight=1.0,
                    numerator=numerator,
                    denominator=denominator,
                )
            },
            metrics={
                "supervised_tokens": ObjectiveMetric.sum(denominator.detach())
            },
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ObjectiveError("causal LM objective has no mutable state")


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("objective:trainomni/causal_lm@1"),
        config_type=CausalLMConfig,
        factory=lambda config, context: CausalLMObjective(config),
        provides=CapabilitySet.of({"objective.causal_lm", "objective.loss_bundle"}),
        requires=CapabilitySet.of({"model.output.logits", "batch.labels"}),
    )
