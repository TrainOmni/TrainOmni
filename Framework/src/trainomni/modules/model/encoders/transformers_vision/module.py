"""Narrow Transformers vision encoder adapter."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from trainomni.contracts.features import ModalFeatures
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import TransformersVisionConfig


class TransformersVisionEncoder(nn.Module):
    def __init__(self, model: nn.Module, config: TransformersVisionConfig) -> None:
        super().__init__()
        self.model = model
        self.output_field = config.output_field
        self.drop_cls_token = config.drop_cls_token

    def forward(self, modal_inputs):
        inputs = modal_inputs if isinstance(modal_inputs, Mapping) else {"pixel_values": modal_inputs}
        output = self.model(**inputs)
        features = (
            output[self.output_field]
            if isinstance(output, Mapping)
            else getattr(output, self.output_field, None)
        )
        if features is None:
            raise SpecError(f"vision encoder output has no {self.output_field!r}")
        if self.drop_cls_token:
            if features.ndim != 3 or features.shape[1] < 2:
                raise SpecError("cannot drop CLS token from this vision output")
            features = features[:, 1:]
        return ModalFeatures(embeddings=features)

    def enable_activation_checkpointing(self, *, use_reentrant: bool) -> None:
        hook = getattr(self.model, "gradient_checkpointing_enable", None)
        if not callable(hook):
            raise SpecError("vision encoder has no gradient checkpointing hook")
        hook(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})


def _factory(config: TransformersVisionConfig, context):
    del context
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise SpecError("Transformers vision encoder requires transformers") from exc
    model = AutoModel.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    return TransformersVisionEncoder(model, config)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("encoder:trainomni/transformers_vision@1"),
        config_type=TransformersVisionConfig,
        factory=_factory,
        provides=CapabilitySet.of(
            {"component.encoder", "modal_features.input", "encoder.vision"}
        ),
    )
