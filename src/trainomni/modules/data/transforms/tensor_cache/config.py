"""Hash-pinned tensor sidecar cache configuration."""

from dataclasses import dataclass

from trainomni.modules.data._validation import require_string


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorCacheConfig:
    index_path: str
    index_sha256: str
    metadata_key: str = "tensor_cache"

    def __post_init__(self) -> None:
        require_string(self.index_path, field="index_path")
        require_string(self.index_sha256, field="index_sha256")
        require_string(self.metadata_key, field="metadata_key")
        if (
            len(self.index_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.index_sha256)
        ):
            raise ValueError("index_sha256 must be a lowercase SHA-256 digest")
