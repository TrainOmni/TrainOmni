"""Packed-attention input and output field configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class PackedAttentionConfig:
    block_attention_field: str = "packed_attention_mask"
    segment_ids_field: str = "packed_segment_ids"
    output_format: str = "additive_4d"

    def __post_init__(self) -> None:
        if not self.block_attention_field or not self.segment_ids_field:
            raise ValueError("packed-attention field names must not be empty")
        if self.block_attention_field == self.segment_ids_field:
            raise ValueError("packed-attention field names must differ")
        if self.output_format not in {"bool_4d", "additive_4d"}:
            raise ValueError("output_format must be bool_4d or additive_4d")
