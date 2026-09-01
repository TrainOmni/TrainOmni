"""Construct labels while retaining cached dense teacher logits."""

import torch

from trainomni.contracts.batch import SupervisedExample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

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
            if not isinstance(loss_mask, torch.Tensor) or loss_mask.shape != labels.shape:
                raise SpecError("dense KD loss_mask must align with input_ids")
            labels = labels.masked_fill(~loss_mask.bool(), self.config.ignore_index)
        return SupervisedExample(
            sample_id=sample.sample_id,
            model_inputs=sample.model_inputs,
            labels=labels,
            supervision=sample.supervision,
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
