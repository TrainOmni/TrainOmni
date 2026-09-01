"""Two-layer modal-feature projector."""

from torch import nn

from trainomni.contracts.features import ModalFeatures
from trainomni.core.capability import CapabilitySet
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import MLPConnectorConfig


def _activation(name: str):
    return {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}[name]()


class MLPConnector(nn.Module):
    def __init__(self, config: MLPConnectorConfig) -> None:
        super().__init__()
        layers = [
            nn.Linear(config.input_dim, config.hidden_dim, bias=config.bias),
            _activation(config.activation),
            nn.Linear(config.hidden_dim, config.output_dim, bias=config.bias),
        ]
        if config.layer_norm:
            layers.append(nn.LayerNorm(config.output_dim))
        self.projection = nn.Sequential(*layers)

    def forward(self, features: ModalFeatures) -> ModalFeatures:
        return ModalFeatures(
            embeddings=self.projection(features.embeddings),
            mask=features.mask,
            positions=features.positions,
            grid=features.grid,
            metadata=features.metadata,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("connector:trainomni/mlp@1"),
        config_type=MLPConnectorConfig,
        factory=lambda config, context: MLPConnector(config),
        provides=CapabilitySet.of({"component.connector", "modal_features.output"}),
        requires=CapabilitySet.of({"modal_features.input"}),
    )
