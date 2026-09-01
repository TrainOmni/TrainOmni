"""Monolithic Transformers-compatible VLM adapter."""

from __future__ import annotations

from torch import nn

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import MonolithicModelConfig


class MonolithicModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        inputs.setdefault("use_cache", False)
        return self.model(**inputs)


def _factory(config: MonolithicModelConfig, context):
    del context
    try:
        import transformers
    except ImportError as exc:
        raise SpecError("monolithic model module requires transformers") from exc
    auto_class = getattr(transformers, config.auto_class, None)
    if auto_class is None:
        raise SpecError(
            f"installed transformers has no {config.auto_class}; choose a supported auto class"
        )
    model = auto_class.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    return MonolithicModel(model)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("model:trainomni/monolithic_transformers@1"),
        config_type=MonolithicModelConfig,
        factory=_factory,
        provides=CapabilitySet.of(
            {"model.monolithic", "model.output.logits", "model.parameters"}
        ),
    )
