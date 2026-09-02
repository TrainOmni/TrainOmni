"""Deterministic video-frame decoding configuration."""

from dataclasses import dataclass

from trainomni.modules.data._validation import require_int


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoTransformConfig:
    frames: int = 8
    sampling: str = "uniform"
    mode: str = "RGB"
    max_decoded_frames: int = 4096

    def __post_init__(self) -> None:
        require_int(self.frames, field="frames", minimum=1)
        require_int(
            self.max_decoded_frames,
            field="max_decoded_frames",
            minimum=1,
        )
        if self.sampling not in {"uniform", "head", "tail"}:
            raise ValueError("sampling must be uniform, head, or tail")
        if self.mode not in {"RGB", "RGBA", "L"}:
            raise ValueError("video frame mode must be RGB, RGBA, or L")
        if self.max_decoded_frames < self.frames:
            raise ValueError("max_decoded_frames must be at least frames")
