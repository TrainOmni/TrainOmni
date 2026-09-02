"""Canonical identity validation shared by process-bound data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def normalize_identity(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def freeze_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return MappingProxyType(dict(value))
