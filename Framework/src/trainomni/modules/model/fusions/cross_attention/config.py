"""Cross-attention fusion keyword configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossAttentionFusionConfig:
    hidden_states_argument: str = "encoder_hidden_states"
    mask_argument: str = "encoder_attention_mask"

    def __post_init__(self) -> None:
        if not self.hidden_states_argument or not self.mask_argument:
            raise ValueError("cross-attention argument names must not be empty")
        if self.hidden_states_argument == self.mask_argument:
            raise ValueError("cross-attention argument names must differ")
