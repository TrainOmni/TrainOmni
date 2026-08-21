"""Validate and retain paired policy/reference preference tensors."""

from collections.abc import Mapping

import torch

from trainomni.contracts.batch import SupervisedExample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import PreferenceSupervisionConfig


class PreferenceSupervision:
    def __init__(self, config: PreferenceSupervisionConfig) -> None:
        self.config = config

    def annotate(self, sample):
        values = {}
        for name in (
            self.config.chosen_inputs_field,
            self.config.rejected_inputs_field,
            self.config.chosen_labels_field,
            self.config.rejected_labels_field,
            self.config.chosen_reference_logps_field,
            self.config.rejected_reference_logps_field,
        ):
            if name not in sample.supervision:
                raise SpecError(f"preference sample is missing {name!r}")
            values[name] = sample.supervision[name]
        chosen_inputs = values[self.config.chosen_inputs_field]
        rejected_inputs = values[self.config.rejected_inputs_field]
        if not isinstance(chosen_inputs, Mapping) or not isinstance(
            rejected_inputs, Mapping
        ):
            raise SpecError("chosen/rejected inputs must be mappings")
        chosen_labels = values[self.config.chosen_labels_field]
        rejected_labels = values[self.config.rejected_labels_field]
        if (
            not isinstance(chosen_labels, torch.Tensor)
            or not isinstance(rejected_labels, torch.Tensor)
            or chosen_labels.ndim != 1
            or rejected_labels.ndim != 1
        ):
            raise SpecError("chosen/rejected labels must be one-dimensional tensors")
        for branch, inputs, labels in (
            ("chosen", chosen_inputs, chosen_labels),
            ("rejected", rejected_inputs, rejected_labels),
        ):
            input_ids = inputs.get("input_ids")
            if not isinstance(input_ids, torch.Tensor) or input_ids.shape != labels.shape:
                raise SpecError(f"{branch} input_ids must align with {branch} labels")
        for name, labels in (
            (self.config.chosen_reference_logps_field, chosen_labels),
            (self.config.rejected_reference_logps_field, rejected_labels),
        ):
            logps = values[name]
            if (
                not isinstance(logps, torch.Tensor)
                or logps.dtype != torch.float32
                or logps.shape not in {labels.shape, labels[1:].shape}
            ):
                raise SpecError(
                    f"{name} must be FP32 and align with full or causal-shifted labels"
                )
        return SupervisedExample(
            sample_id=sample.sample_id,
            model_inputs=chosen_inputs,
            labels=chosen_labels,
            supervision=values,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("supervision:trainomni/preference@1"),
        config_type=PreferenceSupervisionConfig,
        factory=lambda config, context: PreferenceSupervision(config),
        provides=CapabilitySet.of(
            {
                "data.supervised",
                "batch.labels",
                "batch.preference",
                "batch.reference_logps",
            }
        ),
        requires=CapabilitySet.of({"data.encoded"}),
    )
