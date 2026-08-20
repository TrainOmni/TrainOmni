"""Training objective contracts and implementations."""

from .binding import (
    DEFAULT_OBJECTIVE_IMPLEMENTATIONS,
    ObjectiveBinding,
    resolve_objective,
)
from .protocol import (
    OBJECTIVE_API_VERSION,
    LossOutput,
    LossTerm,
    Objective,
    ObjectiveManifest,
    ObjectiveRequirements,
)
from .registry import (
    MaskedCausalLMObjective,
    ObjectiveRegistry,
    ObjectiveRegistryError,
)

__all__ = [
    "DEFAULT_OBJECTIVE_IMPLEMENTATIONS",
    "OBJECTIVE_API_VERSION",
    "LossOutput",
    "LossTerm",
    "MaskedCausalLMObjective",
    "Objective",
    "ObjectiveBinding",
    "ObjectiveManifest",
    "ObjectiveRegistry",
    "ObjectiveRegistryError",
    "ObjectiveRequirements",
    "resolve_objective",
]
