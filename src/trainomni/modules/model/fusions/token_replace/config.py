"""Token-replacement fusion configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenReplaceConfig:
    strict_count: bool = True
