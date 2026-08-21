"""Transformers save_pretrained export configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformersExportConfig:
    safe_serialization: bool = True
    max_shard_size: str = "5GB"
    save_processor: bool = True

    def __post_init__(self) -> None:
        if not self.max_shard_size.strip():
            raise ValueError("max_shard_size must not be empty")
