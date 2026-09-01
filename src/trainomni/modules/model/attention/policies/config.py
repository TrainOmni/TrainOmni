"""Pass-through semantic attention policy."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class AttentionPolicyConfig:
    require_attention_mask: bool = False
