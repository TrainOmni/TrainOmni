"""Narrow Transformers video encoder adapter."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from trainomni.contracts.features import ModalFeatures
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import TransformersVideoConfig


class TransformersVideoEncoder(nn.Module):
    def __init__(self, model: nn.Module, config: TransformersVideoConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, modal_inputs):
        inputs = (
            dict(modal_inputs)
            if isinstance(modal_inputs, Mapping)
            else {self.config.input_field: modal_inputs}
        )
        grid = (
            None
            if self.config.temporal_grid_field is None
            else inputs.get(self.config.temporal_grid_field)
        )
        output = self.model(**inputs)
        features = (
            output.get(self.config.output_field)
            if isinstance(output, Mapping)
            else getattr(output, self.config.output_field, None)
        )
        if features is None:
            raise SpecError(f"video encoder output has no {self.config.output_field!r}")
        if features.ndim != 3:
            raise SpecError("video encoder features must be [batch, tokens, hidden]")
        return ModalFeatures(embeddings=features, grid=grid)

    def enable_activation_checkpointing(self, *, use_reentrant: bool) -> None:
        hook = getattr(self.model, "gradient_checkpointing_enable", None)
        if not callable(hook):
            raise SpecError("video encoder has no gradient checkpointing hook")
        hook(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})


def _factory(config: TransformersVideoConfig, context):
    del context
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise SpecError("Transformers video encoder requires transformers") from exc
    model = AutoModel.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    return TransformersVideoEncoder(model, config)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("encoder:trainomni/transformers_video@1"),
        config_type=TransformersVideoConfig,
        factory=_factory,
        provides=CapabilitySet.of(
            {"component.encoder", "modal_features.input", "encoder.video"}
        ),
    )
