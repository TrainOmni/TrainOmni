"""Native paired-policy DPO with offline per-token reference log-probs."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional

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

from .._ops.sequence_logp import causal_sequence_logp
from ..protocol import ObjectiveRequirements
from .config import DPOConfig


class DPOObjective:
    def __init__(self, config: DPOConfig) -> None:
        self.config = config

    def requirements(self) -> ObjectiveRequirements:
        return ObjectiveRequirements(
            outputs=OutputRequirements(logits=True),
            supervision_fields=frozenset(
                {
                    self.config.chosen_inputs_field,
                    self.config.rejected_inputs_field,
                    self.config.chosen_labels_field,
                    self.config.rejected_labels_field,
                    self.config.chosen_reference_logps_field,
                    self.config.rejected_reference_logps_field,
                }
            ),
        )

    def plan(self, batch: OmniBatch, context: ObjectiveContext) -> ForwardPlan:
        chosen_inputs = self._field(batch, self.config.chosen_inputs_field)
        rejected_inputs = self._field(batch, self.config.rejected_inputs_field)
        if not isinstance(chosen_inputs, Mapping) or not isinstance(
            rejected_inputs, Mapping
        ):
            raise ObjectiveError("DPO chosen/rejected inputs must be mappings")
        self._preflight_branch(
            batch,
            inputs=chosen_inputs,
            labels_field=self.config.chosen_labels_field,
            reference_field=self.config.chosen_reference_logps_field,
            branch="chosen",
        )
        self._preflight_branch(
            batch,
            inputs=rejected_inputs,
            labels_field=self.config.rejected_labels_field,
            reference_field=self.config.rejected_reference_logps_field,
            branch="rejected",
        )
        outputs = self.requirements().outputs
        return ForwardPlan(
            (
                ForwardRequest(
                    "chosen_policy",
                    chosen_inputs,
                    outputs,
                    requires_grad=context.training,
                ),
                ForwardRequest(
                    "rejected_policy",
                    rejected_inputs,
                    outputs,
                    requires_grad=context.training,
                ),
            )
        )

    def compute(
        self,
        batch: OmniBatch,
        outputs: Mapping[str, ForwardResult],
        context: ObjectiveContext,
    ) -> LossBundle:
        del context
        if set(outputs) != {"chosen_policy", "rejected_policy"}:
            raise ObjectiveError("DPO requires exactly chosen_policy and rejected_policy")
        chosen_labels = self._tensor_field(batch, self.config.chosen_labels_field)
        rejected_labels = self._tensor_field(batch, self.config.rejected_labels_field)
        chosen_policy, chosen_mask = causal_sequence_logp(
            outputs["chosen_policy"].require("logits"),
            chosen_labels,
            ignore_index=self.config.ignore_index,
        )
        rejected_policy, rejected_mask = causal_sequence_logp(
            outputs["rejected_policy"].require("logits"),
            rejected_labels,
            ignore_index=self.config.ignore_index,
        )
        chosen_reference = self._reference_sequence_logp(
            batch,
            self.config.chosen_reference_logps_field,
            chosen_mask,
            chosen_labels,
        )
        rejected_reference = self._reference_sequence_logp(
            batch,
            self.config.rejected_reference_logps_field,
            rejected_mask,
            rejected_labels,
        )
        policy_ratio = chosen_policy - rejected_policy
        reference_ratio = chosen_reference - rejected_reference
        delta = policy_ratio - reference_ratio
        dpo_logit = self.config.beta * delta
        per_pair_loss = -functional.logsigmoid(dpo_logit)
        loss = per_pair_loss.mean(dtype=torch.float32)
        pair_count = torch.tensor(
            per_pair_loss.shape[0],
            device=loss.device,
            dtype=torch.long,
        )
        chosen_reward = self.config.beta * (chosen_policy - chosen_reference)
        rejected_reward = self.config.beta * (rejected_policy - rejected_reference)
        margin = chosen_reward - rejected_reward
        return LossBundle(
            total=loss,
            terms={
                "dpo": LossTerm(
                    value=loss,
                    weight=1.0,
                    numerator=per_pair_loss.sum(dtype=torch.float32),
                    denominator=pair_count,
                )
            },
            metrics={
                "chosen_policy_logp": chosen_policy.mean().detach(),
                "rejected_policy_logp": rejected_policy.mean().detach(),
                "chosen_reference_logp": chosen_reference.mean().detach(),
                "rejected_reference_logp": rejected_reference.mean().detach(),
                "policy_ratio": policy_ratio.mean().detach(),
                "reference_ratio": reference_ratio.mean().detach(),
                "delta": delta.mean().detach(),
                "dpo_logit": dpo_logit.mean().detach(),
                "chosen_reward": chosen_reward.mean().detach(),
                "rejected_reward": rejected_reward.mean().detach(),
                "reward_margin": margin.mean().detach(),
                "accuracy": margin.gt(0).float().mean().detach(),
                "preference_pairs": pair_count.detach(),
                "chosen_tokens": chosen_mask.sum().detach(),
                "rejected_tokens": rejected_mask.sum().detach(),
            },
        )

    def _field(self, batch: OmniBatch, name: str):
        if name not in batch.supervision:
            raise ObjectiveError(f"DPO is missing supervision field {name!r}")
        return batch.supervision[name]

    def _tensor_field(self, batch: OmniBatch, name: str):
        value = self._field(batch, name)
        if not isinstance(value, torch.Tensor):
            raise ObjectiveError(f"DPO field {name!r} must be a tensor")
        return value

    def _reference_sequence_logp(self, batch, name, mask, labels):
        token_logps = self._tensor_field(batch, name)
        if token_logps.dtype != torch.float32:
            raise ObjectiveError(f"DPO reference field {name!r} must be FP32")
        if token_logps.shape == labels.shape:
            token_logps = token_logps[:, 1:]
        if token_logps.shape != mask.shape:
            raise ObjectiveError(
                f"DPO reference field {name!r} does not align with labels"
            )
        return (token_logps * mask).sum(dim=-1, dtype=torch.float32)

    def _preflight_branch(
        self,
        batch,
        *,
        inputs,
        labels_field,
        reference_field,
        branch,
    ):
        input_ids = inputs.get("input_ids")
        labels = self._tensor_field(batch, labels_field)
        reference = self._tensor_field(batch, reference_field)
        if not isinstance(input_ids, torch.Tensor) or input_ids.shape != labels.shape:
            raise ObjectiveError(
                f"DPO {branch} input_ids must be a tensor aligned with labels"
            )
        if labels.ndim != 2:
            raise ObjectiveError(f"DPO {branch} labels must be [batch, sequence]")
        if reference.dtype != torch.float32:
            raise ObjectiveError(f"DPO reference field {reference_field!r} must be FP32")
        if reference.shape not in {
            labels.shape,
            labels[:, 1:].shape,
        }:
            raise ObjectiveError(
                f"DPO {branch} reference log-probs do not align with labels"
            )

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ObjectiveError("DPO objective has no mutable state")


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("objective:trainomni/offline_reference_dpo@1"),
        config_type=DPOConfig,
        factory=lambda config, context: DPOObjective(config),
        provides=CapabilitySet.of({"objective.dpo", "objective.loss_bundle"}),
        requires=CapabilitySet.of(
            {"model.output.logits", "batch.preference", "batch.reference_logps"}
        ),
    )
