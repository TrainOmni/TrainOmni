"""Stable numerical helpers available to builtin and custom objectives."""

from trainomni.contracts.cache import model_inputs_digest

from ._ops.causal_shift import causal_shift
from ._ops.reductions import reduce_token_losses
from ._ops.sequence_logp import causal_sequence_logp
from ._ops.token_ce import token_cross_entropy
from ._ops.token_kl import dense_token_kl

__all__ = [
    "causal_sequence_logp",
    "causal_shift",
    "dense_token_kl",
    "model_inputs_digest",
    "reduce_token_losses",
    "token_cross_entropy",
]
