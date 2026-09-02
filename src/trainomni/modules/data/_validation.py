"""Strict, non-coercing validators shared by builtin data modules."""

from __future__ import annotations

import math
from typing import Any


def require_bool(value: Any, *, field: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")


def require_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")


def require_string(value: Any, *, field: str, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")


def normalize_string_sequence(
    value: Any,
    *,
    field: str,
    allow_empty_sequence: bool = True,
    unique: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field} must be a sequence")
    normalized = tuple(value)
    if not allow_empty_sequence and not normalized:
        raise ValueError(f"{field} must not be empty")
    if any(not isinstance(item, str) or not item for item in normalized):
        raise TypeError(f"{field} must contain non-empty strings")
    if unique and len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must be unique")
    return normalized


def require_number(value: Any, *, field: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{field} must not be NaN")
