"""Preserve semantic optimizer groups across in-place parallel transforms."""

from __future__ import annotations

from trainomni.core.errors import SpecError
from trainomni.modules.parameters.protocol import ParameterGroup, ParameterSelection


def selection_names(model, selection: ParameterSelection) -> tuple[tuple[str, tuple[str, ...]], ...]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in selection.groups:
        resolved = []
        for parameter in group.parameters:
            try:
                resolved.append(names[id(parameter)])
            except KeyError as exc:
                raise SpecError(
                    f"optimizer group {group.name!r} contains a parameter outside the model"
                ) from exc
        groups.append((group.name, tuple(resolved)))
    return tuple(groups)


def remap_selection(
    model,
    selection: ParameterSelection,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
) -> ParameterSelection:
    parameters = dict(model.named_parameters())
    remapped = []
    for original, (group_name, names) in zip(selection.groups, groups, strict=True):
        if original.name != group_name:
            raise SpecError("parameter group identity changed during backend setup")
        missing = sorted(set(names) - set(parameters))
        if missing:
            raise SpecError(
                f"execution backend lost parameters from group {group_name!r}: "
                + ", ".join(missing)
            )
        remapped.append(
            ParameterGroup(
                name=group_name,
                parameters=tuple(parameters[name] for name in names),
                options=original.options,
            )
        )
    return ParameterSelection(
        groups=tuple(remapped),
        trainable_names=selection.trainable_names,
        frozen_names=selection.frozen_names,
    )
