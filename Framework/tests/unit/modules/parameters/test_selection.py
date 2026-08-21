import pytest
from torch import nn

from trainomni.core.errors import SpecError
from trainomni.modules.parameters.component.config import ComponentParameterConfig
from trainomni.modules.parameters.component.module import ComponentParameterPolicy
from trainomni.modules.parameters.freeze.config import FreezeParameterConfig
from trainomni.modules.parameters.freeze.module import FreezeParameterPolicy
from trainomni.modules.parameters.full.config import FullParameterConfig
from trainomni.modules.parameters.full.module import FullParameterPolicy
from trainomni.runtime.optimization.optimizer import build_optimizer
from trainomni.specs.run import OptimizerSpec


class Components(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.connector = nn.Linear(2, 2)
        self.language = nn.Linear(2, 2)


def test_component_policy_selects_exact_named_submodule() -> None:
    model = Components()
    selection = ComponentParameterPolicy(
        ComponentParameterConfig(train=("connector",))
    ).apply(model)
    assert selection.trainable_names == ("connector.weight", "connector.bias")
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.connector.parameters())


def test_freeze_policy_and_missing_component_are_fail_closed() -> None:
    model = Components()
    selection = FreezeParameterPolicy(
        FreezeParameterConfig(freeze=("encoder", "language"))
    ).apply(model)
    assert selection.trainable_names == ("connector.weight", "connector.bias")
    with pytest.raises(SpecError, match="matched no parameters"):
        ComponentParameterPolicy(
            ComponentParameterConfig(train=("not_a_component",))
        ).apply(model)


def test_optimizer_overrides_named_component_groups() -> None:
    model = Components()
    selection = ComponentParameterPolicy(
        ComponentParameterConfig(train=("encoder", "connector"))
    ).apply(model)
    spec = OptimizerSpec.from_mapping(
        {
            "learning_rate": 0.001,
            "groups": {
                "encoder": {"learning_rate": 0.0001},
                "connector": {"learning_rate": 0.01, "weight_decay": 0.2},
            },
        }
    )
    optimizer = build_optimizer(spec, selection)
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["encoder"]["lr"] == 0.0001
    assert groups["connector"]["lr"] == 0.01
    assert groups["connector"]["weight_decay"] == 0.2

    unknown = OptimizerSpec.from_mapping(
        {"groups": {"language": {"learning_rate": 0.1}}}
    )
    with pytest.raises(SpecError, match="unknown parameter groups"):
        build_optimizer(unknown, selection)


def test_full_policy_can_group_every_parameter_by_top_level_component() -> None:
    model = Components()
    selection = FullParameterPolicy(
        FullParameterConfig(group_by_top_level_component=True)
    ).apply(model)
    assert tuple(group.name for group in selection.groups) == (
        "encoder",
        "connector",
        "language",
    )
    assert selection.trainable_numel == sum(
        parameter.numel() for parameter in model.parameters()
    )
