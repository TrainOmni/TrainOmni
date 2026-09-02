"""Fail-closed tensor semantics shared by builtin data modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from trainomni.core.errors import SpecError


def binary_mask(
    value: Any,
    *,
    field: str,
    shape: Sequence[int] | torch.Size,
) -> torch.Tensor:
    """Return a boolean view only after proving exact binary mask semantics."""

    if not isinstance(value, torch.Tensor):
        raise SpecError(f"{field} must be a tensor")
    if value.shape != torch.Size(shape):
        raise SpecError(f"{field} must align exactly with its token tensor")
    if value.dtype is torch.bool:
        return value
    if value.is_complex() or value.dtype.is_floating_point and not bool(
        torch.isfinite(value).all().item()
    ):
        raise SpecError(f"{field} must contain only binary 0/1 values")
    if not bool(((value == 0) | (value == 1)).all().item()):
        raise SpecError(f"{field} must contain only binary 0/1 values")
    return value.to(dtype=torch.bool)
