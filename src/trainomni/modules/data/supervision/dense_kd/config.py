"""Offline dense-logit supervision configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseKDSupervisionConfig:
    input_ids_field: str = "input_ids"
    teacher_logits_field: str = "teacher_logits"
    loss_mask_field: str = "loss_mask"
    ignore_index: int = -100
