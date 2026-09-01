"""Train everything except explicitly frozen components."""

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from ..protocol import ParameterGroup, ParameterSelection
from .config import FreezeParameterConfig


class FreezeParameterPolicy:
    def __init__(self, config: FreezeParameterConfig) -> None:
        self.config = config

    def apply(self, model):
        matched = {component: 0 for component in self.config.freeze}
        trainable_parameters = []
        trainable_names = []
        frozen_names = []
        for name, parameter in model.named_parameters():
            frozen = False
            for component in self.config.freeze:
                if name == component or name.startswith(component + "."):
                    matched[component] += 1
                    frozen = True
                    break
            parameter.requires_grad_(not frozen)
            if frozen:
                frozen_names.append(name)
            else:
                trainable_parameters.append(parameter)
                trainable_names.append(name)
        missing = sorted(component for component, count in matched.items() if count == 0)
        if missing:
            raise SpecError("freeze components matched no parameters: " + ", ".join(missing))
        if not trainable_parameters:
            raise SpecError("freeze policy left no trainable parameters")
        return ParameterSelection(
            groups=(
                ParameterGroup(
                    name=self.config.group_name,
                    parameters=tuple(trainable_parameters),
                    options={},
                ),
            ),
            trainable_names=tuple(trainable_names),
            frozen_names=tuple(frozen_names),
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("parameter_policy:trainomni/freeze@1"),
        config_type=FreezeParameterConfig,
        factory=lambda config, context: FreezeParameterPolicy(config),
        provides=CapabilitySet.of({"parameters.freeze"}),
        requires=CapabilitySet.of({"model.parameters"}),
    )
