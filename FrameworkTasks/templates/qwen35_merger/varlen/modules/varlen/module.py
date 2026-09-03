"""Reuse composite fusion; only bridge the final layout to upstream varlen."""

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.model.models.composite.config import CompositeModelConfig
from trainomni.modules.model.models.composite.module import CompositeModel
from trainomni.modules.model.language.transformers_causal_lm.module import TransformersCausalLanguage
from trainomni.runtime.kernels.attention.varlen import VarlenLayout, padding_free_forward


class VarlenLanguage(TransformersCausalLanguage):
    def forward_embeddings(self, embeddings, *, attention_mask=None, trainomni_layout=None, **kwargs):
        if kwargs or trainomni_layout is None:
            raise SpecError("varlen language requires only validated post-fusion layout")
        if attention_mask is None or not attention_mask.bool().all():
            raise SpecError("varlen language rejects padding")
        return padding_free_forward(self.model, inputs_embeds=embeddings, layout=trainomni_layout)


class VarlenComposite(CompositeModel):
    def forward(self, **inputs):
        layout = VarlenLayout.from_packed(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"], segment_ids=inputs["packed_segment_ids"],
            cu_seqlens=inputs["packed_cu_seqlens"],
        )
        if "packed_attention_mask" in inputs:
            raise SpecError("varlen task must not materialize a dense language mask")
        inputs = dict(inputs)
        for key in ("position_ids", "packed_segment_ids", "packed_cu_seqlens"):
            inputs.pop(key)
        # This task expands image placeholders BEFORE packing. Token replacement
        # does not change length; the validated layout is already post-fusion.
        return super().forward(**inputs, trainomni_layout=layout)


def build(config, context):
    components = dict(context.components)
    language = components[config.language]
    if not isinstance(language, TransformersCausalLanguage):
        raise SpecError("this varlen example requires the Transformers language adapter")
    components[config.language] = VarlenLanguage(language.model)
    return VarlenComposite(branches=config.branches, components=components,
                           fusion_name=config.fusion, language_name=config.language)


def descriptor():
    return ModuleDescriptor(
        module_id=ModuleId.parse("model:example/qwen35_merger_varlen@1"),
        config_type=CompositeModelConfig, factory=build,
        provides=CapabilitySet.of({"model.composite", "model.output.logits", "model.parameters",
                                   "model.attention.padding_free"}),
        requires=CapabilitySet.of({"component.encoder", "component.connector", "component.fusion",
                                   "component.language", "fusion.sequence_length_preserving"}),
    )
