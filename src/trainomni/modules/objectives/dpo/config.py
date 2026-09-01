"""Offline-reference sigmoid DPO configuration."""

import math
from dataclasses import dataclass

_ALLOWED_BRANCH_SEQUENCE_FIELDS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "cache_position",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DPOConfig:
    reference_producer_identity_sha256: str
    beta: float = 0.1
    ignore_index: int = -100
    chosen_inputs_field: str = "chosen_inputs"
    rejected_inputs_field: str = "rejected_inputs"
    chosen_labels_field: str = "chosen_labels"
    rejected_labels_field: str = "rejected_labels"
    chosen_reference_logps_field: str = "chosen_reference_logps"
    rejected_reference_logps_field: str = "rejected_reference_logps"
    branch_sequence_fields: tuple[str, ...] = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "cache_position",
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("DPO beta must be positive")
        fields = (
            self.chosen_inputs_field,
            self.rejected_inputs_field,
            self.chosen_labels_field,
            self.rejected_labels_field,
            self.chosen_reference_logps_field,
            self.rejected_reference_logps_field,
        )
        if any(not field for field in fields) or len(set(fields)) != len(fields):
            raise ValueError("DPO supervision field names must be non-empty and unique")
        if len(self.reference_producer_identity_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.reference_producer_identity_sha256
        ):
            raise ValueError(
                "reference_producer_identity_sha256 must be a lowercase SHA-256 digest"
            )
        if not self.branch_sequence_fields or any(
            not field for field in self.branch_sequence_fields
        ):
            raise ValueError("branch_sequence_fields must contain non-empty names")
        unknown_branch_fields = sorted(
            set(self.branch_sequence_fields) - _ALLOWED_BRANCH_SEQUENCE_FIELDS
        )
        if unknown_branch_fields:
            raise ValueError(
                "DPO branch_sequence_fields may only contain token-sequence fields: "
                + ", ".join(unknown_branch_fields)
            )
