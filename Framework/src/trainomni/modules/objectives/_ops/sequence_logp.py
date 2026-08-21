"""Causal sequence log-probability sums over explicit supervised positions."""

from __future__ import annotations

import torch
from torch.nn import functional

from trainomni.core.errors import ObjectiveError

from .causal_shift import causal_shift


def causal_sequence_logp(logits, labels, *, ignore_index: int):
    shifted_logits, shifted_labels = causal_shift(logits, labels)
    mask = shifted_labels.ne(ignore_index)
    if bool((mask.sum(dim=-1) == 0).any().detach().item()):
        raise ObjectiveError("every preference branch requires supervised tokens")
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = functional.log_softmax(shifted_logits.float(), dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * mask).sum(dim=-1, dtype=torch.float32), mask
