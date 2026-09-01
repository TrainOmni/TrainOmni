"""Gradient utilities kept independent of task semantics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def _local_or_full(tensor: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return tensor
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def clip_gradients(parameters: Iterable[Any], max_norm: float | None) -> float:
    materialized = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not materialized:
        return 0.0
    if max_norm is not None:
        norm = torch.nn.utils.clip_grad_norm_(materialized, max_norm)
        return float(norm.detach().float().item())
    squared = torch.zeros((), device=materialized[0].device, dtype=torch.float32)
    for parameter in materialized:
        if parameter.grad is not None:
            gradient = _local_or_full(parameter.grad.detach())
            squared = squared + gradient.float().square().sum()
    return float(squared.sqrt().item())
