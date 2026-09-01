"""Pillow image decoding configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageTransformConfig:
    mode: str = "RGB"
    max_pixels: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"RGB", "RGBA", "L"}:
            raise ValueError("image mode must be RGB, RGBA, or L")
        if self.max_pixels is not None and self.max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
