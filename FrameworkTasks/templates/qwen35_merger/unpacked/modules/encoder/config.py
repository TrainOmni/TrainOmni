from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class VisionConfig:
    model_path: str
    weights_sha256: Mapping[str, str]
    config_sha256: str
