"""Full-parameter policy."""

from __future__ import annotations

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from ..protocol import ParameterGroup, ParameterSelection
from .config import FullParameterConfig


class FullParameterPolicy:
    def __init__(self, config: FullParameterConfig) -> None:
        self.config = config

    def apply(self, model):
        grouped = {}
        names = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(True)
            group = (
                name.split(".", 1)[0]
                if self.config.group_by_top_level_component
                else self.config.group_name
            )
            grouped.setdefault(group, []).append(parameter)
            names.append(name)
        if not grouped:
            raise SpecError("full parameter policy found no model parameters")
        return ParameterSelection(
            groups=tuple(
                ParameterGroup(
                    name=group,
                    parameters=tuple(parameters),
                    options={},
                )
                for group, parameters in grouped.items()
            ),
            trainable_names=tuple(names),
            frozen_names=(),
        )


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("parameter_policy:trainomni/full@1"),
        config_type=FullParameterConfig,
        factory=lambda config, context: FullParameterPolicy(config),
        provides=CapabilitySet.of(
            {"parameters.full", "parameters.component_evidence"}
        ),
        requires=CapabilitySet.of({"model.parameters"}),
    )
