from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True, kw_only=True)
class IOConfig:
    vision_model_path: str
    language_model_path: str
    vision_assets_sha256: Mapping[str, str]
    language_assets_sha256: Mapping[str, str]
    min_pixels: int = 4096
    max_pixels: int = 16384
    max_tokens: int = 512
    supervision_metadata_key: str | None = None
