"""Linear modal-feature projection."""

from torch import nn

from trainomni.contracts.features import ModalFeatures
from trainomni.core.capability import CapabilitySet
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import LinearConnectorConfig


class LinearConnector(nn.Module):
    def __init__(self, config: LinearConnectorConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(config.input_dim, config.output_dim, bias=config.bias)

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
        module_id=ModuleId.parse("connector:trainomni/linear@1"),
        config_type=LinearConnectorConfig,
        factory=lambda config, context: LinearConnector(config),
        provides=CapabilitySet.of({"component.connector", "modal_features.output"}),
        requires=CapabilitySet.of({"modal_features.input"}),
    )
