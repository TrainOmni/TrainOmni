"""Modal-prefix fusion configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class PrefixFusionConfig:
    position: str = "before_text"

    def __post_init__(self) -> None:
        if self.position != "before_text":
            raise ValueError("prefix fusion v1 only supports before_text")
