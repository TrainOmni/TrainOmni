"""Builtin composite model."""

from .config import CompositeBranchConfig, CompositeModelConfig
from .module import CompositeModel, descriptor

__all__ = [
    "CompositeBranchConfig",
    "CompositeModel",
    "CompositeModelConfig",
    "descriptor",
]
