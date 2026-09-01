"""Model-declared structure used by execution backends.

These hints describe physical module boundaries only. They never choose a
backend or topology; that remains a RunSpec concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trainomni.core.errors import SpecError


@dataclass(frozen=True, slots=True)
class DistributionHints:
    fsdp_units: tuple[str, ...] = ()
    expert_modules: tuple[str, ...] = ()
    router_modules: tuple[str, ...] = ()
    replicated_modules: tuple[str, ...] = ()
    tied_parameter_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        fields = {
            "fsdp_units": self.fsdp_units,
            "expert_modules": self.expert_modules,
            "router_modules": self.router_modules,
            "replicated_modules": self.replicated_modules,
        }
        for field, paths in fields.items():
            if any(not isinstance(path, str) or not path for path in paths):
                raise ValueError(f"{field} must contain non-empty module paths")
            if len(paths) != len(set(paths)):
                raise ValueError(f"{field} contains duplicate module paths")
        for group in self.tied_parameter_groups:
            if len(group) < 2 or any(not isinstance(name, str) or not name for name in group):
                raise ValueError(
                    "tied_parameter_groups must contain at least two parameter names"
                )

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_modules or self.router_modules)

    def validate(self, model: Any) -> None:
        modules = dict(model.named_modules())
        parameters = dict(model.named_parameters())
        missing_modules = sorted(
            path
            for path in (
                *self.fsdp_units,
                *self.expert_modules,
                *self.router_modules,
                *self.replicated_modules,
            )
            if path not in modules
        )
        missing_parameters = sorted(
            name
            for group in self.tied_parameter_groups
            for name in group
            if name not in parameters
        )
        if missing_modules or missing_parameters:
            details = []
            if missing_modules:
                details.append("modules=" + ",".join(missing_modules))
            if missing_parameters:
                details.append("parameters=" + ",".join(missing_parameters))
            raise SpecError("invalid model distribution hints: " + "; ".join(details))


def distribution_hints(model: Any) -> DistributionHints:
    hook = getattr(model, "distribution_hints", None)
    if hook is None:
        return DistributionHints()
    if not callable(hook):
        raise SpecError("model distribution_hints must be callable")
    hints = hook()
    if not isinstance(hints, DistributionHints):
        raise SpecError("model distribution_hints() returned an invalid value")
    hints.validate(model)
    return hints
