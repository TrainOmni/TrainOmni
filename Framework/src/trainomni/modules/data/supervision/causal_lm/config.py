"""Causal supervision configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class CausalSupervisionConfig:
    input_ids_field: str = "input_ids"
    loss_mask_field: str = "loss_mask"
    ignore_index: int = -100
