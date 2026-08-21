"""Select complete named model components for training."""

from trainomni.core.capability import CapabilitySet
from trainomni.core.module import ModuleDescriptor, ModuleId

from .._selection import select_components
from .config import ComponentParameterConfig


class ComponentParameterPolicy:
    def __init__(self, config: ComponentParameterConfig) -> None:
        self.config = config

    def apply(self, model):
        return select_components(
            model,
            train_components=self.config.train,
            group_per_component=self.config.group_per_component,
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("parameter_policy:trainomni/component@1"),
        config_type=ComponentParameterConfig,
        factory=lambda config, context: ComponentParameterPolicy(config),
        provides=CapabilitySet.of({"parameters.component"}),
        requires=CapabilitySet.of({"model.parameters"}),
    )
