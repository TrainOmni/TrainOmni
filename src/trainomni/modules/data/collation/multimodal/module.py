"""Strict fixed-shape collation for the first vertical slice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from trainomni.contracts.batch import OmniBatch, SupervisedExample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import MultimodalCollatorConfig


class MultimodalCollator:
    def __init__(self, config: MultimodalCollatorConfig) -> None:
        self.config = config

    def _pad_value(self, field: str):
        configured = self.config.field_pad_values.get(
            field,
            self.config.field_pad_values.get(field.rsplit(".", 1)[-1]),
        )
        if configured is not None:
            return configured
        leaf = field.rsplit(".", 1)[-1]
        if leaf in {"labels", "chosen_labels", "rejected_labels"}:
            return self.config.label_pad_id
        if leaf in {"attention_mask", "chosen_reference_logps", "rejected_reference_logps"}:
            return 0
        if leaf in {"modal_positions", "packed_segment_ids"}:
            return -1
        if leaf == "packed_lengths":
            return 0
        return self.config.pad_token_id

    def _mode(self, field: str) -> str:
        return self.config.field_modes.get(
            field,
            self.config.field_modes.get(field.rsplit(".", 1)[-1], "auto"),
        )

    def _pad_tensors(self, values, *, field: str):
        if any(value.ndim == 0 for value in values):
            raise SpecError(f"collator cannot pad scalar tensors for {field}")
        ndim = values[0].ndim
        tail = tuple(values[0].shape[1:])
        if any(value.ndim != ndim or tuple(value.shape[1:]) != tail for value in values):
            raise SpecError(
                f"collator pad requires equal trailing shapes for {field}: "
                f"{sorted({tuple(value.shape) for value in values})}"
            )
        target = max(value.shape[0] for value in values)
        multiple = self.config.pad_to_multiple_of
        if multiple is not None:
            target = ((target + multiple - 1) // multiple) * multiple
        output = values[0].new_full(
            (len(values), target, *tail),
            self._pad_value(field),
        )
        for index, value in enumerate(values):
            if self.config.padding_side == "right":
                output[index, : value.shape[0]] = value
            else:
                output[index, target - value.shape[0] :] = value
        return output

    @staticmethod
    def _concat_tensors(values, *, field: str):
        if any(value.ndim == 0 for value in values):
            raise SpecError(f"collator cannot concatenate scalar tensors for {field}")
        ndim = values[0].ndim
        tail = tuple(values[0].shape[1:])
        if any(value.ndim != ndim or tuple(value.shape[1:]) != tail for value in values):
            raise SpecError(
                f"collator concat requires equal trailing shapes for {field}: "
                f"{sorted({tuple(value.shape) for value in values})}"
            )
        return torch.cat(tuple(values), dim=0)

    def _stack(self, values, *, field: str):
        mode = self._mode(field)
        if mode == "list":
            return tuple(values)
        if all(isinstance(value, torch.Tensor) for value in values):
            shapes = {tuple(value.shape) for value in values}
            if mode == "stack":
                if len(shapes) != 1:
                    raise SpecError(
                        f"collator stack requires equal shapes for {field}: {shapes}"
                    )
                return torch.stack(tuple(values))
            if mode == "pad":
                return self._pad_tensors(values, field=field)
            if mode == "concat":
                return self._concat_tensors(values, field=field)
            if len(shapes) == 1:
                return torch.stack(tuple(values))
            if any(value.ndim != 1 for value in values):
                raise SpecError(
                    f"collator cannot pad non-sequence tensors for {field}: {shapes}"
                )
            return self._pad_tensors(values, field=field)
        if all(isinstance(value, Mapping) for value in values):
            if mode != "auto":
                raise SpecError(
                    f"collator mode {mode!r} cannot be applied to mapping field {field}"
                )
            keys = set(values[0])
            if any(set(value) != keys for value in values[1:]):
                raise SpecError(f"collator mapping keys differ for {field}")
            return {
                key: self._stack(
                    [value[key] for value in values],
                    field=f"{field}.{key}",
                )
                for key in sorted(keys)
            }
        raise SpecError(
            f"collator only supports tensors, nested mappings, or configured list "
            f"fields for {field}"
        )

    def collate(self, examples: Sequence[SupervisedExample]) -> OmniBatch:
        if not examples:
            raise SpecError("cannot collate an empty example sequence")
        input_keys = set(examples[0].model_inputs)
        if any(set(example.model_inputs) != input_keys for example in examples[1:]):
            raise SpecError("model input keys differ within a batch")
        model_inputs = {
            key: self._stack(
                [example.model_inputs[key] for example in examples],
                field=f"model_inputs.{key}",
            )
            for key in sorted(input_keys)
        }
        supervision_keys = set(examples[0].supervision)
        if any(set(example.supervision) != supervision_keys for example in examples[1:]):
            raise SpecError("supervision keys differ within a batch")
        supervision = {
            key: self._stack(
                [example.supervision[key] for example in examples],
                field=f"supervision.{key}",
            )
            for key in sorted(supervision_keys)
        }
        return OmniBatch(
            sample_ids=tuple(example.sample_id for example in examples),
            model_inputs=model_inputs,
            labels=self._stack(
                [example.labels for example in examples], field="labels"
            ),
            supervision=supervision,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("collator:trainomni/multimodal@1"),
        config_type=MultimodalCollatorConfig,
        factory=lambda config, context: MultimodalCollator(config),
        provides=CapabilitySet.of({"batch.omni"}),
        requires=CapabilitySet.of({"data.packed", "batch.labels"}),
    )
