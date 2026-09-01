"""Generic safetensors export configuration."""

from dataclasses import dataclass
from pathlib import PurePath


@dataclass(frozen=True, slots=True, kw_only=True)
class SafetensorsExportConfig:
    filename: str = "model.safetensors"

    def __post_init__(self) -> None:
        if (
            not self.filename.endswith(".safetensors")
            or "/" in self.filename
            or "\\" in self.filename
            or PurePath(self.filename).name != self.filename
        ):
            raise ValueError("filename must be a local .safetensors filename")
