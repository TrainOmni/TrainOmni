"""Pillow image decoding configuration."""

from dataclasses import dataclass

from trainomni.modules.data._validation import require_int


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageTransformConfig:
    mode: str = "RGB"
    max_pixels: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"RGB", "RGBA", "L"}:
            raise ValueError("image mode must be RGB, RGBA, or L")
        if self.max_pixels is not None:
            require_int(self.max_pixels, field="max_pixels", minimum=1)
