"""Native offline dense-logit KD objective."""

from __future__ import annotations

from collections.abc import Mapping

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

from .._ops.cache_identity import validate_cache_binding
from .._ops.causal_shift import causal_shift
from .._ops.reductions import reduce_token_losses
from .._ops.token_ce import token_cross_entropy
from .._ops.token_kl import dense_token_kl
from ..protocol import ObjectiveRequirements
from .config import DenseKDConfig


class DenseKDObjective:
    def __init__(self, config: DenseKDConfig) -> None:
        self.config = config

    def requirements(self) -> ObjectiveRequirements:
        prefix = f"__cache_identity__{self.config.teacher_logits_field}__"
        return ObjectiveRequirements(
            outputs=OutputRequirements(logits=True),
            supervision_fields=frozenset(
                {
                    self.config.teacher_logits_field,
                    prefix + "input_ids_sha256",
                    prefix + "supervised_positions_sha256",
                    prefix + "target_token_ids_sha256",
                    prefix + "producer_identity_sha256",
                    prefix + "branch",
                }
            ),
        )

    def plan(self, batch: OmniBatch, context: ObjectiveContext) -> ForwardPlan:
        teacher_logits = batch.supervision.get(self.config.teacher_logits_field)
        if not isinstance(teacher_logits, torch.Tensor) or teacher_logits.ndim != 3:
            raise ObjectiveError(
                "dense KD teacher logits must be [batch, positions, vocab]"
            )
        if batch.labels.ndim != 2 or teacher_logits.shape[0] != batch.labels.shape[0]:
            raise ObjectiveError("dense KD teacher logits batch does not align with labels")
        if teacher_logits.shape[1] not in {
            batch.labels.shape[1],
            batch.labels.shape[1] - 1,
        }:
            raise ObjectiveError(
                "dense KD teacher positions must align with full or causal-shifted labels"
            )
        validate_cache_binding(
            batch=batch,
            cache_field=self.config.teacher_logits_field,
            inputs=batch.model_inputs,
            labels=batch.labels,
            ignore_index=self.config.ignore_index,
            branch_code=0,
            producer_identity_sha256=self.config.producer_identity_sha256,
        )
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
            raise ObjectiveError("dense KD requires the 'policy' forward")
        teacher_logits = batch.supervision.get(self.config.teacher_logits_field)
        if teacher_logits is None:
            raise ObjectiveError(
                f"dense KD is missing supervision {self.config.teacher_logits_field!r}"
            )
        if not isinstance(teacher_logits, torch.Tensor):
            raise ObjectiveError("dense KD teacher logits must be a tensor")
        student_logits, labels = causal_shift(
            outputs["policy"].require("logits"), batch.labels
        )
        if teacher_logits.shape == outputs["policy"].require("logits").shape:
            teacher_logits = teacher_logits[:, :-1, :]
        if teacher_logits.shape != student_logits.shape:
            raise ObjectiveError(
                "teacher logits must align with causal prediction positions: "
                f"{tuple(teacher_logits.shape)} vs {tuple(student_logits.shape)}"
            )
        mask = labels.ne(self.config.ignore_index)
        ce_tokens = token_cross_entropy(
            student_logits,
            labels,
            ignore_index=self.config.ignore_index,
            label_smoothing=0.0,
        )
        kd_tokens = dense_token_kl(
            student_logits,
            teacher_logits,
            temperature=self.config.temperature,
        )
        ce, ce_numerator, denominator = reduce_token_losses(
            ce_tokens, mask, reduction=self.config.reduction
        )
        kd, kd_numerator, kd_denominator = reduce_token_losses(
            kd_tokens, mask, reduction=self.config.reduction
        )
        total = (
            self.config.ce_weight * ce
            + self.config.kd_weight * self.config.temperature**2 * kd
        )
        return LossBundle(
            total=total,
            terms={
                "token_ce": LossTerm(
                    value=ce,
                    weight=self.config.ce_weight,
                    numerator=ce_numerator,
                    denominator=denominator,
                ),
                "dense_kl": LossTerm(
                    value=kd,
                    weight=self.config.kd_weight * self.config.temperature**2,
                    numerator=kd_numerator,
                    denominator=kd_denominator,
                ),
            },
            metrics={"supervised_tokens": denominator.detach()},
        )

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ObjectiveError("dense KD objective has no mutable state")


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("objective:trainomni/dense_kd@1"),
        config_type=DenseKDConfig,
        factory=lambda config, context: DenseKDObjective(config),
        provides=CapabilitySet.of({"objective.dense_kd", "objective.loss_bundle"}),
        requires=CapabilitySet.of(
            {"model.output.logits", "batch.labels", "batch.teacher_logits"}
        ),
    )
