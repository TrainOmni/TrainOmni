"""Local media resolution configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class MediaTransformConfig:
    require_sha256: bool = False
