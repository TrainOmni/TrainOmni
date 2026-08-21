"""Fixed-length multimodal-aware packing with explicit attention isolation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch

from trainomni.contracts.batch import SupervisedExample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import CheckpointError, SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import SequencePackerConfig


class SequencePacker:
    def __init__(self, config: SequencePackerConfig) -> None:
        self.config = config
        self._buffer: list[SupervisedExample] = []
        self._tokens = 0

    def _length(self, sample: SupervisedExample) -> int:
        input_ids = sample.model_inputs.get(self.config.input_ids_field)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            raise SpecError("sequence packing requires one-dimensional input_ids")
        labels = sample.labels
        if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
            raise SpecError("sequence packing requires labels aligned with input_ids")
        length = int(input_ids.shape[0])
        if length <= 0 or length > self.config.max_length:
            raise SpecError(
                f"sample token length {length} is outside [1, {self.config.max_length}]"
            )
        attention = sample.model_inputs.get(self.config.attention_mask_field)
        if attention is not None:
            if not isinstance(attention, torch.Tensor) or attention.shape != input_ids.shape:
                raise SpecError("pre-pack attention_mask must align with input_ids")
            if not bool(attention.bool().all().item()):
                raise SpecError("pre-pack examples must not contain padded attention tokens")
        return length

    @staticmethod
    def _validate_tensor_family(values: Sequence[torch.Tensor], *, field: str) -> None:
        first = values[0]
        if any(
            value.ndim != first.ndim
            or tuple(value.shape[1:]) != tuple(first.shape[1:])
            or value.dtype != first.dtype
            or value.device != first.device
            for value in values[1:]
        ):
            raise SpecError(
                f"packed field {field!r} has incompatible rank, trailing shape, "
                "dtype, or device"
            )

    @staticmethod
    def _pad_first_axis(
        value: torch.Tensor,
        *,
        target: int,
        pad_value: float,
    ) -> torch.Tensor:
        if value.ndim == 0 or value.shape[0] > target:
            raise SpecError("cannot pad this packed tensor along dimension zero")
        result = value.new_full((target, *value.shape[1:]), pad_value)
        result[: value.shape[0]] = value
        return result

    def _pack_model_inputs(
        self,
        samples: Sequence[SupervisedExample],
        lengths: Sequence[int],
    ) -> dict[str, object]:
        keys = set(samples[0].model_inputs)
        if any(set(sample.model_inputs) != keys for sample in samples[1:]):
            raise SpecError("packed samples must have identical model-input keys")
        reserved = {
            self.config.input_ids_field,
            self.config.attention_mask_field,
            self.config.position_ids_field,
            self.config.segment_ids_field,
            self.config.block_attention_field,
        }
        known = (
            reserved
            | set(self.config.sequence_fields)
            | set(self.config.concat_fields)
            | set(self.config.offset_fields)
            | set(self.config.list_fields)
        )
        unknown = sorted(keys - known)
        if unknown:
            raise SpecError(
                "sequence packer has no field policy for: " + ", ".join(unknown)
            )

        max_length = self.config.max_length
        input_values = [
            sample.model_inputs[self.config.input_ids_field] for sample in samples
        ]
        packed_input_ids = self._pad_first_axis(
            torch.cat(tuple(input_values), dim=0),
            target=max_length,
            pad_value=self.config.pad_token_id,
        )
        valid_tokens = sum(lengths)
        attention_mask = packed_input_ids.new_zeros(max_length)
        attention_mask[:valid_tokens] = 1
        positions = packed_input_ids.new_zeros(max_length)
        segments = packed_input_ids.new_full((max_length,), -1)
        block_attention = torch.zeros(
            (1, max_length, max_length),
            dtype=torch.bool,
            device=packed_input_ids.device,
        )
        offset = 0
        for segment, length in enumerate(lengths):
            stop = offset + length
            positions[offset:stop] = torch.arange(
                length,
                dtype=positions.dtype,
                device=positions.device,
            )
            segments[offset:stop] = segment
            block_attention[:, offset:stop, offset:stop] = torch.ones(
                (length, length),
                dtype=torch.bool,
                device=block_attention.device,
            ).tril()
            offset = stop
        output: dict[str, object] = {
            self.config.input_ids_field: packed_input_ids,
            self.config.attention_mask_field: attention_mask,
            self.config.position_ids_field: positions,
            self.config.segment_ids_field: segments,
            self.config.block_attention_field: block_attention,
        }

        for field in self.config.sequence_fields:
            values = [sample.model_inputs[field] for sample in samples]
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise SpecError(f"packed sequence field {field!r} must contain tensors")
            self._validate_tensor_family(values, field=field)
            if any(
                value.shape[0] != length
                for value, length in zip(values, lengths, strict=True)
            ):
                raise SpecError(f"packed sequence field {field!r} must align with input_ids")
            output[field] = self._pad_first_axis(
                torch.cat(tuple(values), dim=0),
                target=max_length,
                pad_value=self.config.field_pad_values.get(field, 0),
            )
        for field in self.config.concat_fields:
            values = [sample.model_inputs[field] for sample in samples]
            if not all(
                isinstance(value, torch.Tensor) and value.ndim > 0 for value in values
            ):
                raise SpecError(f"packed concat field {field!r} must contain tensors")
            self._validate_tensor_family(values, field=field)
            output[field] = torch.cat(tuple(values), dim=0)
        for field in self.config.offset_fields:
            values = [sample.model_inputs[field] for sample in samples]
            if not all(
                isinstance(value, torch.Tensor)
                and value.ndim == 1
                and not value.is_floating_point()
                for value in values
            ):
                raise SpecError(
                    f"packed offset field {field!r} must contain integer vectors"
                )
            adjusted = []
            offset = 0
            for value, length in zip(values, lengths, strict=True):
                if bool((value < 0).any().item()) or bool((value >= length).any().item()):
                    raise SpecError(
                        f"packed offset field {field!r} contains invalid positions"
                    )
                adjusted.append(value + offset)
                offset += length
            output[field] = torch.cat(tuple(adjusted), dim=0)
        for field in self.config.list_fields:
            output[field] = tuple(sample.model_inputs[field] for sample in samples)
        return output

    def _pack_supervision(
        self,
        samples: Sequence[SupervisedExample],
        lengths: Sequence[int],
    ) -> Mapping[str, object]:
        keys = set(samples[0].supervision)
        if any(set(sample.supervision) != keys for sample in samples[1:]):
            raise SpecError("packed samples must have identical supervision keys")
        output = {}
        for field in sorted(keys):
            values = [sample.supervision[field] for sample in samples]
            if not all(
                isinstance(value, torch.Tensor) and value.ndim > 0 for value in values
            ):
                raise SpecError(
                    f"packed supervision field {field!r} must be a token-aligned tensor"
                )
            self._validate_tensor_family(values, field=f"supervision.{field}")
            if any(
                value.shape[0] != length
                for value, length in zip(values, lengths, strict=True)
            ):
                raise SpecError(
                    f"packed supervision field {field!r} must align with input_ids"
                )
            output[field] = self._pad_first_axis(
                torch.cat(tuple(values), dim=0),
                target=self.config.max_length,
                pad_value=0,
            )
        output["packed_lengths"] = torch.tensor(
            lengths,
            dtype=torch.int64,
            device=samples[0].labels.device,
        )
        return MappingProxyType(output)

    def _emit(self) -> SupervisedExample:
        if not self._buffer:
            raise SpecError("cannot emit an empty sequence pack")
        samples = tuple(self._buffer)
        lengths = tuple(self._length(sample) for sample in samples)
        labels = torch.cat(tuple(sample.labels for sample in samples), dim=0)
        offset = 0
        for index, length in enumerate(lengths):
            if index > 0:
                labels[offset] = self.config.ignore_index
            offset += length
        labels = self._pad_first_axis(
            labels,
            target=self.config.max_length,
            pad_value=self.config.ignore_index,
        )
        digest = hashlib.sha256(
            "\0".join(sample.sample_id for sample in samples).encode("utf-8")
        ).hexdigest()[:20]
        result = SupervisedExample(
            sample_id=f"pack:{digest}",
            model_inputs=MappingProxyType(self._pack_model_inputs(samples, lengths)),
            labels=labels,
            supervision=self._pack_supervision(samples, lengths),
        )
        self._buffer = []
        self._tokens = 0
        return result

    def add(self, sample: SupervisedExample) -> tuple[SupervisedExample, ...]:
        length = self._length(sample)
        emitted = []
        if self._buffer and self._tokens + length > self.config.max_length:
            emitted.append(self._emit())
        self._buffer.append(sample)
        self._tokens += length
        sample_limit = self.config.max_samples_per_pack
        if self._tokens == self.config.max_length or (
            sample_limit is not None and len(self._buffer) >= sample_limit
        ):
            emitted.append(self._emit())
        return tuple(emitted)

    def state_dict(self) -> Mapping[str, object]:
        return {"buffer": tuple(self._buffer), "tokens": self._tokens}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"buffer", "tokens"}:
            raise CheckpointError("invalid sequence-packer state keys")
        buffer = state["buffer"]
        tokens = int(state["tokens"])
        if not isinstance(buffer, (tuple, list)) or any(
            not isinstance(sample, SupervisedExample) for sample in buffer
        ):
            raise CheckpointError("sequence-packer buffer is invalid")
        try:
            actual = sum(self._length(sample) for sample in buffer)
        except SpecError as exc:
            raise CheckpointError(f"sequence-packer buffer is incompatible: {exc}") from exc
        if tokens != actual or not 0 <= tokens < self.config.max_length:
            raise CheckpointError("sequence-packer token cursor is inconsistent")
        self._buffer = list(buffer)
        self._tokens = tokens


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("packer:trainomni/sequence@1"),
        config_type=SequencePackerConfig,
        factory=lambda config, context: SequencePacker(config),
        provides=CapabilitySet.of(
            {"data.packed", "batch.packed_attention", "batch.segment_ids"}
        ),
        requires=CapabilitySet.of({"data.supervised", "model.attention.packed"}),
    )
