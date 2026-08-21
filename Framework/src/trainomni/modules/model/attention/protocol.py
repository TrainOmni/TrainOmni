"""Semantic attention-policy contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class AttentionInputs:
    attention_mask: Any | None
    model_kwargs: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    consumed_model_inputs: tuple[str, ...] = ()


class AttentionPolicy(Protocol):
    def apply(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None,
        modal_positions: Any | None,
        model_inputs: Mapping[str, Any],
    ) -> AttentionInputs: ...
