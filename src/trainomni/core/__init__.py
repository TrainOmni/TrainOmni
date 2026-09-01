"""Stable module-kernel public surface."""

from .capability import CapabilitySet
from .context import BuildContext, ObjectiveContext
from .module import ModuleDescriptor, ModuleId, ModuleKind, ModuleRef
from .registry import ModuleRegistry
from .resolver import ModuleResolver, ResolvedModule

__all__ = [
    "BuildContext",
    "CapabilitySet",
    "ModuleDescriptor",
    "ModuleId",
    "ModuleKind",
    "ModuleRef",
    "ModuleRegistry",
    "ModuleResolver",
    "ObjectiveContext",
    "ResolvedModule",
]
