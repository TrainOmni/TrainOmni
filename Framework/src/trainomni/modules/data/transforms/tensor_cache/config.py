"""Hash-pinned tensor sidecar cache configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorCacheConfig:
    index_path: str
    index_sha256: str
    metadata_key: str = "tensor_cache"

    def __post_init__(self) -> None:
        if not self.index_path or not self.metadata_key:
            raise ValueError("index_path and metadata_key must not be empty")
        if (
            len(self.index_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.index_sha256)
        ):
            raise ValueError("index_sha256 must be a lowercase SHA-256 digest")
