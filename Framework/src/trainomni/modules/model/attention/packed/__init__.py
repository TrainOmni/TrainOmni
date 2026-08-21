"""Packed block-diagonal causal attention policy."""

from .config import PackedAttentionConfig
from .module import PackedAttentionPolicy, descriptor

__all__ = ["PackedAttentionConfig", "PackedAttentionPolicy", "descriptor"]
