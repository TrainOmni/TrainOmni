"""Native Linear-LoRA adapter export configuration."""

from dataclasses import dataclass
from pathlib import PurePath


@dataclass(frozen=True, slots=True, kw_only=True)
class LoRAAdapterExportConfig:
    filename: str = "adapter.safetensors"
    include_trainable_bias: bool = True

    def __post_init__(self) -> None:
        if (
            not self.filename.endswith(".safetensors")
            or PurePath(self.filename).name != self.filename
            or "/" in self.filename
            or "\\" in self.filename
        ):
            raise ValueError("filename must be a local .safetensors filename")
