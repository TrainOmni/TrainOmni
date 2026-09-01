"""Supply modal features as decoder cross-attention memory."""

from __future__ import annotations

import torch
from torch import nn

from trainomni.contracts.features import ModalFeatureSet
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import CrossAttentionFusionConfig


class CrossAttentionFusion(nn.Module):
    def __init__(self, config: CrossAttentionFusionConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        *,
        language,
        input_ids,
        modal_features,
        attention_mask=None,
        modal_positions=None,
        **kwargs,
    ):
        del modal_positions
        feature_set = ModalFeatureSet.coerce(modal_features)
        if not feature_set.branches:
            raise SpecError("cross-attention fusion requires modal features")
        try:
            merged = feature_set.concatenate()
        except ValueError as exc:
            raise SpecError(f"invalid cross-attention modal features: {exc}") from exc
        text_embeddings = language.embed(input_ids)
        memory = merged.embeddings.to(text_embeddings.dtype)
        memory_mask = merged.mask
        if memory_mask is None:
            memory_mask = torch.ones(
                memory.shape[:2], dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=memory.device,
            )
        reserved = {
            self.config.hidden_states_argument: memory,
            self.config.mask_argument: memory_mask,
        }
        overlap = sorted(set(kwargs) & set(reserved))
        if overlap:
            raise SpecError(
                "cross-attention inputs attempted to overwrite fusion-owned fields: "
                + ", ".join(overlap)
            )
        kwargs.update(reserved)
        return language.forward_embeddings(
            text_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("fusion:trainomni/cross_attention@1"),
        config_type=CrossAttentionFusionConfig,
        factory=lambda config, context: CrossAttentionFusion(config),
        provides=CapabilitySet.of({"component.fusion", "fusion.cross_attention"}),
        requires=CapabilitySet.of(
            {"language.inputs_embeds", "language.cross_attention"}
        ),
    )
