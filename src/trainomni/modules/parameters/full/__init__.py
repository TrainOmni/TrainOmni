"""Full-parameter module."""

from .config import FullParameterConfig
from .module import FullParameterPolicy, descriptor

__all__ = ["FullParameterConfig", "FullParameterPolicy", "descriptor"]
