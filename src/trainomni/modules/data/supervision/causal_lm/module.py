"""Construct causal labels from encoded model inputs."""

from __future__ import annotations

import torch

from trainomni.contracts.batch import EncodedSample, SupervisedExample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.data._tensors import binary_mask

from .config import CausalSupervisionConfig


class CausalSupervision:
    def __init__(self, config: CausalSupervisionConfig) -> None:
        self.config = config

    def annotate(self, sample: EncodedSample) -> SupervisedExample:
        input_ids = sample.model_inputs.get(self.config.input_ids_field)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            raise SpecError("causal supervision requires one-dimensional input_ids")
        labels = input_ids.detach().clone()
        loss_mask = sample.supervision.get(self.config.loss_mask_field)
        if loss_mask is not None:
            loss_mask = binary_mask(
                loss_mask,
                field="loss_mask",
                shape=labels.shape,
            )
            labels = labels.masked_fill(~loss_mask, self.config.ignore_index)
        return SupervisedExample(
            sample_id=sample.sample_id,
            model_inputs=sample.model_inputs,
            labels=labels,
            supervision=sample.supervision,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("supervision:trainomni/causal_lm@1"),
        config_type=CausalSupervisionConfig,
        factory=lambda config, context: CausalSupervision(config),
        provides=CapabilitySet.of({"data.supervised", "batch.labels"}),
        requires=CapabilitySet.of({"data.encoded"}),
    )
