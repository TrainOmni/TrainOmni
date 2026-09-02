"""Construct labels while retaining cached dense teacher logits."""

import torch

from trainomni.contracts.batch import SupervisedExample
from trainomni.contracts.cache import (
    current_model_inputs_field,
    digest_tensor,
    model_inputs_digest,
)
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.data._tensors import binary_mask

from .config import DenseKDSupervisionConfig


class DenseKDSupervision:
    def __init__(self, config: DenseKDSupervisionConfig) -> None:
        self.config = config

    def annotate(self, sample):
        input_ids = sample.model_inputs.get(self.config.input_ids_field)
        teacher_logits = sample.supervision.get(self.config.teacher_logits_field)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            raise SpecError("dense KD supervision requires one-dimensional input_ids")
        if not isinstance(teacher_logits, torch.Tensor) or teacher_logits.ndim != 2:
            raise SpecError("dense KD teacher_logits must be [positions, vocab]")
        labels = input_ids.detach().clone()
        loss_mask = sample.supervision.get(self.config.loss_mask_field)
        if loss_mask is not None:
            loss_mask = binary_mask(
                loss_mask,
                field="dense KD loss_mask",
                shape=labels.shape,
            )
            labels = labels.masked_fill(~loss_mask, self.config.ignore_index)
        supervision = dict(sample.supervision)
        current_field = current_model_inputs_field(self.config.teacher_logits_field)
        if current_field in supervision:
            raise SpecError(
                f"dense KD supervision reserves current-input field {current_field!r}"
            )
        try:
            current_digest = model_inputs_digest(sample.model_inputs)
        except ValueError as exc:
            raise SpecError(f"cannot bind dense KD model inputs: {exc}") from exc
        supervision[current_field] = digest_tensor(current_digest)
        return SupervisedExample(
            sample_id=sample.sample_id,
            model_inputs=sample.model_inputs,
            labels=labels,
            supervision=supervision,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("supervision:trainomni/dense_kd@1"),
        config_type=DenseKDSupervisionConfig,
        factory=lambda config, context: DenseKDSupervision(config),
        provides=CapabilitySet.of(
            {"data.supervised", "batch.labels", "batch.teacher_logits"}
        ),
        requires=CapabilitySet.of({"data.encoded"}),
    )
