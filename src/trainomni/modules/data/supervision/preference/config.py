"""Paired offline-reference preference supervision configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferenceSupervisionConfig:
    chosen_inputs_field: str = "chosen_inputs"
    rejected_inputs_field: str = "rejected_inputs"
    chosen_labels_field: str = "chosen_labels"
    rejected_labels_field: str = "rejected_labels"
    chosen_reference_logps_field: str = "chosen_reference_logps"
    rejected_reference_logps_field: str = "rejected_reference_logps"

    def __post_init__(self) -> None:
        fields = (
            self.chosen_inputs_field,
            self.rejected_inputs_field,
            self.chosen_labels_field,
            self.rejected_labels_field,
            self.chosen_reference_logps_field,
            self.rejected_reference_logps_field,
        )
        if any(not field for field in fields) or len(set(fields)) != len(fields):
            raise ValueError("preference field names must be non-empty and unique")
