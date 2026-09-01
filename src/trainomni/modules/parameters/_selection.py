"""Shared deterministic named-parameter selection helpers."""

from __future__ import annotations

from collections.abc import Iterable

from trainomni.core.errors import SpecError

from .protocol import ParameterGroup, ParameterSelection


def select_components(
    model,
    *,
    train_components: Iterable[str],
    group_per_component: bool,
) -> ParameterSelection:
    requested = tuple(dict.fromkeys(train_components))
    if not requested:
        raise SpecError("parameter policy requires at least one trainable component")
    grouped: dict[str, list] = {name: [] for name in requested}
    grouped_names: dict[str, list[str]] = {name: [] for name in requested}
    frozen = []
    for parameter_name, parameter in model.named_parameters():
        matched = next(
            (
                component
                for component in requested
                if parameter_name == component or parameter_name.startswith(component + ".")
            ),
            None,
        )
        parameter.requires_grad_(matched is not None)
        if matched is None:
            frozen.append(parameter_name)
        else:
            grouped[matched].append(parameter)
            grouped_names[matched].append(parameter_name)
    missing = [name for name, parameters in grouped.items() if not parameters]
    if missing:
        raise SpecError(
            "parameter components matched no parameters: " + ", ".join(sorted(missing))
        )
    if group_per_component:
        groups = tuple(
            ParameterGroup(name=name, parameters=tuple(grouped[name]), options={})
            for name in requested
        )
    else:
        groups = (
            ParameterGroup(
                name="selected",
                parameters=tuple(
                    parameter for name in requested for parameter in grouped[name]
                ),
                options={},
            ),
        )
    return ParameterSelection(
        groups=groups,
        trainable_names=tuple(
            parameter_name
            for name in requested
            for parameter_name in grouped_names[name]
        ),
        frozen_names=tuple(frozen),
    )
