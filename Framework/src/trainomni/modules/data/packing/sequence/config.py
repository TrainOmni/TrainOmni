"""Fail-closed fixed-length sequence-packing configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True, kw_only=True)
class SequencePackerConfig:
    max_length: int
    pad_token_id: int
    ignore_index: int = -100
    max_samples_per_pack: int | None = None
    input_ids_field: str = "input_ids"
    attention_mask_field: str = "attention_mask"
    position_ids_field: str = "position_ids"
    segment_ids_field: str = "packed_segment_ids"
    block_attention_field: str = "packed_attention_mask"
    sequence_fields: tuple[str, ...] = ()
    concat_fields: tuple[str, ...] = ()
    offset_fields: tuple[str, ...] = ()
    list_fields: tuple[str, ...] = ()
    field_pad_values: Mapping[str, int | float] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if self.max_length <= 1:
            raise ValueError("max_length must be greater than one")
        if self.max_samples_per_pack is not None and self.max_samples_per_pack <= 0:
            raise ValueError("max_samples_per_pack must be positive")
        names = (
            self.input_ids_field,
            self.attention_mask_field,
            self.position_ids_field,
            self.segment_ids_field,
            self.block_attention_field,
        )
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("sequence packer core field names must be non-empty and unique")
        groups = {
            "sequence_fields": self.sequence_fields,
            "concat_fields": self.concat_fields,
            "offset_fields": self.offset_fields,
            "list_fields": self.list_fields,
        }
        reserved = set(names)
        configured = []
        for owner, values in groups.items():
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{owner} must contain non-empty field names")
            if len(values) != len(set(values)):
                raise ValueError(f"{owner} contains duplicate fields")
            overlap = sorted(set(values) & reserved)
            if overlap:
                raise ValueError(f"{owner} contains reserved fields: {', '.join(overlap)}")
            configured.extend(values)
        duplicates = sorted(
            name for name in set(configured) if configured.count(name) > 1
        )
        if duplicates:
            raise ValueError(
                "sequence packer fields have multiple policies: "
                + ", ".join(duplicates)
            )
        if not isinstance(self.field_pad_values, Mapping):
            raise TypeError("field_pad_values must be a mapping")
        pad_values = {}
        for name, value in self.field_pad_values.items():
            if name not in self.sequence_fields:
                raise ValueError(
                    f"field_pad_values.{name} has no matching sequence_fields entry"
                )
            if not isinstance(value, int | float):
                raise TypeError(f"field_pad_values.{name} must be numeric")
            pad_values[name] = value
        object.__setattr__(
            self,
            "field_pad_values",
            MappingProxyType(dict(sorted(pad_values.items()))),
        )
