"""Validate and translate packed block-diagonal causal masks."""

from __future__ import annotations

import torch

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.data._tensors import binary_mask

from ..protocol import AttentionInputs
from .config import PackedAttentionConfig


def _describe(value):
    if isinstance(value, torch.Tensor):
        return f"shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
    return f"type={type(value).__name__}"


class PackedAttentionPolicy:
    def __init__(self, config: PackedAttentionConfig) -> None:
        self.config = config

    def apply(
        self,
        *,
        input_ids,
        attention_mask,
        modal_positions,
        model_inputs,
    ) -> AttentionInputs:
        del modal_positions
        block = model_inputs.get(self.config.block_attention_field)
        segments = model_inputs.get(self.config.segment_ids_field)
        if not isinstance(block, torch.Tensor) or block.dtype is not torch.bool:
            raise SpecError(f"packed attention mask must be a boolean tensor; got {_describe(block)}")
        if block.shape != (
            input_ids.shape[0],
            1,
            input_ids.shape[1],
            input_ids.shape[1],
        ):
            raise SpecError(
                "packed attention mask must be [batch, 1, sequence, sequence] "
                f"= {(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])}; "
                f"got {_describe(block)}. The singleton axis is the head axis, not batch."
            )
        if not isinstance(segments, torch.Tensor) or segments.shape != input_ids.shape:
            raise SpecError(
                f"packed segment ids must align with input_ids {tuple(input_ids.shape)}; "
                f"got {_describe(segments)}"
            )
        if segments.dtype is torch.bool or segments.is_floating_point() or segments.is_complex():
            raise SpecError(f"packed segment ids must use an integer dtype; got {_describe(segments)}")
        if block.device != input_ids.device or segments.device != input_ids.device:
            raise SpecError("packed attention tensors must share the input_ids device")
        valid = binary_mask(
            attention_mask,
            field="packed token-validity attention_mask",
            shape=input_ids.shape,
        )
        if valid.device != input_ids.device:
            raise SpecError("packed attention tensors must share the input_ids device")
        expected = (
            valid[:, None, :, None]
            & valid[:, None, None, :]
            & (segments[:, None, :, None] == segments[:, None, None, :])
        )
        causal = torch.ones(
            (input_ids.shape[1], input_ids.shape[1]),
            dtype=torch.bool,
            device=input_ids.device,
        ).tril()
        expected &= causal[None, None]
        if not torch.equal(block, expected):
            raise SpecError(
                "packed attention mask disagrees with validity, segments, or causal order"
            )
        output = block
        if self.config.output_format == "additive_4d":
            output = torch.zeros(
                block.shape,
                dtype=torch.float32,
                device=block.device,
            ).masked_fill(~block, torch.finfo(torch.float32).min)
        return AttentionInputs(
            attention_mask=output,
            consumed_model_inputs=(
                self.config.block_attention_field,
                self.config.segment_ids_field,
            ),
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("attention_policy:trainomni/packed_block_diagonal@1"),
        config_type=PackedAttentionConfig,
        factory=lambda config, context: PackedAttentionPolicy(config),
        provides=CapabilitySet.of({"model.attention.packed"}),
        requires=CapabilitySet.of(
            {"model.attention.semantic", "fusion.sequence_length_preserving"}
        ),
    )
