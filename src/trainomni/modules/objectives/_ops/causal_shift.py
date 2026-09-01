"""Causal next-token alignment."""

from __future__ import annotations

from typing import Any

from trainomni.core.errors import ObjectiveError


def causal_shift(logits: Any, labels: Any) -> tuple[Any, Any]:
    if logits.ndim != 3:
        raise ObjectiveError(f"causal logits must be [batch, sequence, vocab], got {logits.shape}")
    if labels.ndim != 2:
        raise ObjectiveError(f"causal labels must be [batch, sequence], got {labels.shape}")
    if logits.shape[:2] != labels.shape:
        raise ObjectiveError(
            f"logit/label leading shape mismatch: {tuple(logits.shape[:2])} vs "
            f"{tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ObjectiveError("causal sequence must contain at least two positions")
    return logits[:, :-1, :], labels[:, 1:]
