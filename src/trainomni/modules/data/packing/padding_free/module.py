"""Reuse field routing and loss masking, without padding or quadratic masks."""

import torch

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.data.packing.sequence.config import SequencePackerConfig
from trainomni.modules.data.packing.sequence.module import SequencePacker

CU_SEQLENS = "packed_cu_seqlens"


class PaddingFreePacker(SequencePacker):
    def __init__(self, config: SequencePackerConfig) -> None:
        super().__init__(config)
        names = (
            config.input_ids_field,
            config.attention_mask_field,
            config.position_ids_field,
            config.segment_ids_field,
            config.block_attention_field,
            *config.sequence_fields,
            *config.concat_fields,
            *config.offset_fields,
            *config.list_fields,
        )
        if CU_SEQLENS in names:
            raise SpecError(f"{CU_SEQLENS} is reserved by the padding-free packer")

    def _target_length(self, lengths) -> int:
        return sum(lengths)

    def _attention_inputs(self, lengths, input_ids):
        if sum(lengths) > torch.iinfo(torch.int32).max:
            raise SpecError("padding-free token count exceeds int32 capacity")
        return {
            CU_SEQLENS: torch.tensor(
                [0, *lengths], dtype=torch.int32, device=input_ids.device
            ).cumsum(0, dtype=torch.int32)
        }


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("packer:trainomni/padding_free@1"),
        config_type=SequencePackerConfig,
        factory=lambda config, context: PaddingFreePacker(config),
        provides=CapabilitySet.of({"batch.padding_free", "batch.segment_ids"}),
        requires=CapabilitySet.of({"data.supervised", "model.attention.padding_free"}),
    )
