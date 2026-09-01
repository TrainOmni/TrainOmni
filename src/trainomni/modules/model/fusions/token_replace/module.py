"""Replace selected text embeddings with modal embeddings."""

from __future__ import annotations

import torch
from torch import nn

from trainomni.contracts.features import ModalFeatureSet
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import TokenReplaceConfig


class TokenReplaceFusion(nn.Module):
    def __init__(self, config: TokenReplaceConfig) -> None:
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
        feature_set = ModalFeatureSet.coerce(modal_features)
        if not feature_set.branches:
            if modal_positions is not None and modal_positions.numel() != 0:
                raise SpecError("text-only fusion received modal positions")
            return language.forward_embeddings(
                language.embed(input_ids),
                attention_mask=attention_mask,
                **kwargs,
            )
        try:
            merged = feature_set.concatenate()
        except ValueError as exc:
            raise SpecError(f"invalid token-replacement modal features: {exc}") from exc
        if modal_positions is None:
            raise SpecError("token-replacement fusion requires modal_positions")
        if modal_positions.ndim != 2 or modal_positions.shape[0] != input_ids.shape[0]:
            raise SpecError("modal_positions must have shape [batch, modal_tokens]")
        if modal_positions.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise SpecError("modal_positions must use an integer dtype")
        modal = merged.embeddings
        if modal.ndim != 3 or modal.shape[:2] != modal_positions.shape:
            raise SpecError("modal embeddings and modal_positions are not aligned")
        valid = merged.mask
        if valid is None:
            valid = torch.ones_like(modal_positions, dtype=torch.bool)
        elif not isinstance(valid, torch.Tensor) or valid.shape != modal_positions.shape:
            raise SpecError("modal feature mask must align with modal_positions")
        else:
            valid = valid.bool()
        if bool(modal_positions[~valid].ne(-1).any().item()):
            raise SpecError("padded modal slots require modal_positions=-1")
        valid_positions = modal_positions[valid]
        if bool(
            (valid_positions.lt(0) | valid_positions.ge(input_ids.shape[1])).any().item()
        ):
            raise SpecError("modal_positions contains an out-of-range index")
        if self.config.strict_count:
            for positions, row_valid in zip(modal_positions, valid, strict=True):
                positions = positions[row_valid]
                if positions.unique().numel() != positions.numel():
                    raise SpecError(
                        "strict token-replacement fusion rejects duplicate positions"
                    )
        embeddings = language.embed(input_ids).clone()
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(1)
        batch_indices = batch_indices.expand_as(modal_positions)[valid]
        embeddings[batch_indices, valid_positions] = modal.to(embeddings.dtype)[valid]
        return language.forward_embeddings(
            embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("fusion:trainomni/token_replace@1"),
        config_type=TokenReplaceConfig,
        factory=lambda config, context: TokenReplaceFusion(config),
        provides=CapabilitySet.of(
            {
                "component.fusion",
                "fusion.token_replace",
                "fusion.sequence_length_preserving",
            }
        ),
        requires=CapabilitySet.of({"language.inputs_embeds", "batch.modal_positions"}),
    )
