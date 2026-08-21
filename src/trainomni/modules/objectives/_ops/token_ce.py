"""FP32 token cross entropy without reduction."""

from __future__ import annotations

from typing import Any


def token_cross_entropy(
    logits: Any,
    labels: Any,
    *,
    ignore_index: int,
    label_smoothing: float,
) -> Any:
    from torch.nn import functional

    vocab_size = logits.shape[-1]
    return functional.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape_as(labels)
