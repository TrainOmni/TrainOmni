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

from .._ops.cache_identity import validate_cache_binding
from .._ops.sequence_logp import causal_sequence_logp
from ..protocol import ObjectiveRequirements
from .config import DPOConfig


class DPOObjective:
    def __init__(self, config: DPOConfig) -> None:
        self.config = config

    def requirements(self) -> ObjectiveRequirements:
        cache_identity_fields = set()
        for field in (
            self.config.chosen_reference_logps_field,
            self.config.rejected_reference_logps_field,
        ):
            prefix = f"__cache_identity__{field}__"
            cache_identity_fields.update(
                {
                    prefix + "input_ids_sha256",
                    prefix + "supervised_positions_sha256",
                    prefix + "target_token_ids_sha256",
                    prefix + "producer_identity_sha256",
                    prefix + "branch",
                }
            )
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
                    *cache_identity_fields,
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
        self._preflight_pair_alignment(
            chosen_inputs=chosen_inputs,
            rejected_inputs=rejected_inputs,
            chosen_labels=self._tensor_field(batch, self.config.chosen_labels_field),
            rejected_labels=self._tensor_field(batch, self.config.rejected_labels_field),
        )
        validate_cache_binding(
            batch=batch,
            cache_field=self.config.chosen_reference_logps_field,
            inputs=chosen_inputs,
            labels=self._tensor_field(batch, self.config.chosen_labels_field),
            ignore_index=self.config.ignore_index,
            branch_code=1,
            producer_identity_sha256=self.config.reference_producer_identity_sha256,
        )
        validate_cache_binding(
            batch=batch,
            cache_field=self.config.rejected_reference_logps_field,
            inputs=rejected_inputs,
            labels=self._tensor_field(batch, self.config.rejected_labels_field),
            ignore_index=self.config.ignore_index,
            branch_code=2,
            producer_identity_sha256=self.config.reference_producer_identity_sha256,
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

    @staticmethod
    def _equal_value(left, right) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and left.shape == right.shape
                and left.dtype == right.dtype
                and torch.equal(left, right)
            )
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            return (
                isinstance(left, Mapping)
                and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(DPOObjective._equal_value(left[key], right[key]) for key in left)
            )
        if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
            return (
                isinstance(left, (tuple, list))
                and isinstance(right, (tuple, list))
                and len(left) == len(right)
                and all(
                    DPOObjective._equal_value(a, b)
                    for a, b in zip(left, right, strict=True)
                )
            )
        return left == right

    def _prompt_rows(self, inputs, labels, *, branch: str):
        input_ids = inputs["input_ids"]
        attention = inputs.get("attention_mask")
        if attention is not None and (
            not isinstance(attention, torch.Tensor) or attention.shape != labels.shape
        ):
            raise ObjectiveError(f"DPO {branch} attention_mask must align with labels")
        prompts = []
        for index in range(labels.shape[0]):
            valid = (
                torch.ones_like(labels[index], dtype=torch.bool)
                if attention is None
                else attention[index].bool()
            )
            ids = input_ids[index][valid]
            row_labels = labels[index][valid]
            supervised = torch.nonzero(
                row_labels.ne(self.config.ignore_index), as_tuple=False
            ).flatten()
            if supervised.numel() == 0:
                raise ObjectiveError(f"DPO {branch} sample has no supervised response")
            boundary = int(supervised[0].item())
            if boundary <= 0 or bool(
                row_labels[:boundary].ne(self.config.ignore_index).any().item()
            ):
                raise ObjectiveError(
                    f"DPO {branch} labels do not define one masked common prompt prefix"
                )
            prompts.append(ids[:boundary].detach().cpu())
        return tuple(prompts)

    def _preflight_pair_alignment(
        self, *, chosen_inputs, rejected_inputs, chosen_labels, rejected_labels
    ) -> None:
        if set(chosen_inputs) != set(rejected_inputs):
            raise ObjectiveError("DPO chosen/rejected model-input keys differ")
        chosen_prompts = self._prompt_rows(chosen_inputs, chosen_labels, branch="chosen")
        rejected_prompts = self._prompt_rows(
            rejected_inputs, rejected_labels, branch="rejected"
        )
        if len(chosen_prompts) != len(rejected_prompts) or any(
            not torch.equal(chosen, rejected)
            for chosen, rejected in zip(chosen_prompts, rejected_prompts, strict=True)
        ):
            raise ObjectiveError("DPO chosen/rejected common prompt tokens differ")
        varying = set(self.config.branch_sequence_fields)
        for field in sorted(set(chosen_inputs) - varying):
            if not self._equal_value(chosen_inputs[field], rejected_inputs[field]):
                raise ObjectiveError(
                    f"DPO chosen/rejected common input {field!r} differs"
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
