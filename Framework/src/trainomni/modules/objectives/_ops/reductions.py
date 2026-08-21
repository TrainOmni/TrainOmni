"""Mask-aware reductions with explicit numerators and denominators."""

from __future__ import annotations

from typing import Any, Literal

from trainomni.core.errors import ObjectiveError


def reduce_token_losses(
    token_losses: Any,
    mask: Any,
    *,
    reduction: Literal["token_mean", "sample_mean"],
) -> tuple[Any, Any, Any]:
    import torch

    if token_losses.shape != mask.shape:
        raise ObjectiveError(
            f"token loss/mask shape mismatch: {token_losses.shape} vs {mask.shape}"
        )
    token_count = mask.sum()
    if int(token_count.detach().item()) == 0:
        raise ObjectiveError("batch contains no supervised token")
    masked = token_losses * mask.to(dtype=token_losses.dtype)
    if reduction == "token_mean":
        numerator = masked.sum(dtype=torch.float32)
        denominator = token_count
        return numerator / denominator, numerator, denominator
    if reduction == "sample_mean":
        per_sample_count = mask.sum(dim=-1)
        if bool((per_sample_count == 0).any().detach().item()):
            raise ObjectiveError("sample_mean requires supervision in every sample")
        per_sample = masked.sum(dim=-1, dtype=torch.float32) / per_sample_count
        numerator = per_sample.sum(dtype=torch.float32)
        denominator = torch.tensor(
            per_sample.shape[0], device=per_sample.device, dtype=torch.long
        )
        return numerator / denominator, numerator, denominator
    raise ObjectiveError(f"unsupported loss reduction: {reduction}")
