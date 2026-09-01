"""Prepend modal embeddings while returning text-aligned logits."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import torch
from torch import nn

from trainomni.contracts.features import ModalFeatureSet
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import PrefixFusionConfig


class PrefixFusion(nn.Module):
    def __init__(self, config: PrefixFusionConfig) -> None:
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
        text_embeddings = language.embed(input_ids)
        feature_set = ModalFeatureSet.coerce(modal_features)
        if not feature_set.branches:
            return language.forward_embeddings(
                text_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )
        try:
            merged = feature_set.concatenate()
        except ValueError as exc:
            raise SpecError(f"invalid prefix modal features: {exc}") from exc
        modal_embeddings = merged.embeddings.to(text_embeddings.dtype)
        if modal_embeddings.ndim != 3:
            raise SpecError("prefix modal embeddings must be [batch, tokens, hidden]")
        if modal_embeddings.shape[0] != text_embeddings.shape[0]:
            raise SpecError("modal and text batch sizes differ")
        embeddings = torch.cat((modal_embeddings, text_embeddings), dim=1)
        if attention_mask is not None or merged.mask is not None:
            if attention_mask is None:
                attention_mask = torch.ones(
                    input_ids.shape,
                    device=input_ids.device,
                    dtype=torch.long,
                )
            prefix_mask = (
                merged.mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                if merged.mask is not None
                else torch.ones(
                    modal_embeddings.shape[:2],
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
            )
            attention_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        position_ids = kwargs.pop("position_ids", None)
        unsupported_positions = sorted(
            key
            for key in kwargs
            if key in {"cache_position", "rope_deltas"} or key.endswith("_positions")
        )
        if unsupported_positions:
            raise SpecError(
                "prefix fusion cannot generically rewrite position-dependent fields: "
                + ", ".join(unsupported_positions)
            )
        if position_ids is not None:
            if not isinstance(position_ids, torch.Tensor) or position_ids.shape != input_ids.shape:
                raise SpecError("prefix fusion position_ids must align with input_ids")
            position_dtype = position_ids.dtype
            if attention_mask is None:
                position_ids = torch.arange(
                    embeddings.shape[1],
                    device=position_ids.device,
                    dtype=position_dtype,
                ).unsqueeze(0).expand(input_ids.shape[0], -1)
            else:
                position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
                position_ids.masked_fill_(attention_mask.eq(0), 0)
                position_ids = position_ids.to(
                    device=position_ids.device,
                    dtype=position_dtype,
                )
            kwargs["position_ids"] = position_ids
        output = language.forward_embeddings(
            embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = output.get("logits") if isinstance(output, Mapping) else getattr(
            output, "logits", None
        )
        if logits is None:
            raise SpecError("language component did not return logits")
        prefix_length = modal_embeddings.shape[1]
        text_logits = logits[:, prefix_length : prefix_length + input_ids.shape[1]]
        if isinstance(output, Mapping):
            result = dict(output)
            result["logits"] = text_logits
            return result
        return SimpleNamespace(logits=text_logits, raw_output=output)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("fusion:trainomni/prefix@1"),
        config_type=PrefixFusionConfig,
        factory=lambda config, context: PrefixFusion(config),
        provides=CapabilitySet.of({"component.fusion", "fusion.prefix"}),
        requires=CapabilitySet.of({"language.inputs_embeds"}),
    )
