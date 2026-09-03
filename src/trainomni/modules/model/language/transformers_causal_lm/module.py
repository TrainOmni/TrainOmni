"""Narrow Transformers causal-LM embedding/decoder adapter."""

from __future__ import annotations

import torch
from torch import nn

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import TransformersCausalLMConfig


class TransformersCausalLanguage(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def embed(self, input_ids):
        embedding = self.model.get_input_embeddings()
        if embedding is None:
            raise SpecError("causal language model does not expose input embeddings")
        return embedding(input_ids)

    def forward_embeddings(self, embeddings, *, attention_mask=None, **kwargs):
        kwargs.pop("input_ids", None)
        # Semantic policies construct additive masks after device placement.
        # Keep boolean masks boolean; match additive bias to runtime Q/K dtype.
        if attention_mask is not None and attention_mask.is_floating_point():
            bounds = torch.finfo(embeddings.dtype)
            # FP32 finfo.min overflows BF16/FP16. Preserve intentional infinities,
            # but keep finite mask sentinels finite (all-padding query rows).
            attention_mask = torch.where(
                torch.isfinite(attention_mask),
                attention_mask.clamp(min=bounds.min, max=bounds.max),
                attention_mask,
            ).to(dtype=embeddings.dtype)
        return self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            **kwargs,
        )

    def enable_activation_checkpointing(self, *, use_reentrant: bool) -> None:
        hook = getattr(self.model, "gradient_checkpointing_enable", None)
        if not callable(hook):
            raise SpecError("causal language model has no gradient checkpointing hook")
        hook(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})


def _factory(config: TransformersCausalLMConfig, context):
    del context
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SpecError("Transformers causal language module requires transformers") from exc
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    return TransformersCausalLanguage(model)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("language:trainomni/transformers_causal_lm@1"),
        config_type=TransformersCausalLMConfig,
        factory=_factory,
        provides=CapabilitySet.of(
            {"component.language", "language.inputs_embeds", "language.causal_lm"}
        ),
    )
