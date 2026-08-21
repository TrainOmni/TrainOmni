"""Model-default attention semantics with strict mask validation."""

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from ..protocol import AttentionInputs
from .config import AttentionPolicyConfig


class ModelDefaultAttentionPolicy:
    def __init__(self, config: AttentionPolicyConfig) -> None:
        self.config = config

    def apply(self, *, input_ids, attention_mask, modal_positions, model_inputs):
        del modal_positions, model_inputs
        if self.config.require_attention_mask and attention_mask is None:
            raise SpecError("attention policy requires attention_mask")
        if attention_mask is not None and (
            attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape
        ):
            raise SpecError(
                "model-default attention_mask must align with input_ids [batch, sequence]"
            )
        return AttentionInputs(attention_mask=attention_mask)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("attention_policy:trainomni/model_default@1"),
        config_type=AttentionPolicyConfig,
        factory=lambda config, context: ModelDefaultAttentionPolicy(config),
        provides=CapabilitySet.of({"attention.semantic.model_default"}),
        requires=CapabilitySet.of({"model.attention.semantic"}),
    )
