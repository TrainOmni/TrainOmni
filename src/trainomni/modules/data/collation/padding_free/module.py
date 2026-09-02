"""A pack already contains multiple samples; do not pad multiple packs together."""

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.data.collation.multimodal.config import MultimodalCollatorConfig
from trainomni.modules.data.collation.multimodal.module import MultimodalCollator
from trainomni.modules.data.packing.padding_free.module import CU_SEQLENS


class PaddingFreeCollator(MultimodalCollator):
    def __init__(self, config: MultimodalCollatorConfig) -> None:
        super().__init__(config)
        if config.padding_side != "right" or config.pad_to_multiple_of is not None:
            raise SpecError("padding-free collation cannot left-pad or pad to a multiple")
        if any(mode == "pad" for mode in config.field_modes.values()):
            raise SpecError("padding-free collation rejects pad field modes")
        for field in (
            "labels",
            "model_inputs.input_ids",
            "model_inputs.attention_mask",
            "model_inputs.position_ids",
            "model_inputs.packed_segment_ids",
            f"model_inputs.{CU_SEQLENS}",
        ):
            if self._mode(field) not in {"auto", "stack"}:
                raise SpecError(f"padding-free {field} must use auto/stack collation")

    def collate(self, examples):
        if len(examples) != 1:
            raise SpecError("padding-free collation requires per_device_batch_size=1 pack")
        if CU_SEQLENS not in examples[0].model_inputs:
            raise SpecError("padding-free collation requires packed_cu_seqlens")
        return super().collate(examples)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("collator:trainomni/padding_free@1"),
        config_type=MultimodalCollatorConfig,
        factory=lambda config, context: PaddingFreeCollator(config),
        provides=CapabilitySet.of({"batch.omni"}),
        requires=CapabilitySet.of({"batch.padding_free", "batch.labels"}),
    )
