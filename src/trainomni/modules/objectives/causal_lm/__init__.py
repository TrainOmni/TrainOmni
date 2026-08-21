"""Builtin causal language-model objective."""

from .config import CausalLMConfig
from .module import CausalLMObjective, descriptor

__all__ = ["CausalLMConfig", "CausalLMObjective", "descriptor"]
