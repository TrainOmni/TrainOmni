"""Builtin catalog public surface."""

from .builtin import builtin_descriptors, builtin_registry
from .local import load_local_descriptor, registry_for_task, source_tree_digest

__all__ = [
    "builtin_descriptors",
    "builtin_registry",
    "load_local_descriptor",
    "registry_for_task",
    "source_tree_digest",
]
