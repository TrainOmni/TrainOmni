"""Causal supervision configuration."""

from dataclasses import dataclass

from trainomni.modules.data._validation import require_int, require_string


@dataclass(frozen=True, slots=True, kw_only=True)
class CausalSupervisionConfig:
    input_ids_field: str = "input_ids"
    loss_mask_field: str = "loss_mask"
    ignore_index: int = -100

    def __post_init__(self) -> None:
        require_string(self.input_ids_field, field="input_ids_field")
        require_string(self.loss_mask_field, field="loss_mask_field")
        require_int(self.ignore_index, field="ignore_index")
