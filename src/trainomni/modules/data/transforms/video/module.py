"""Decode local video blocks into deterministic Pillow frame sequences."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from trainomni.contracts.sample import ContentBlock, OmniSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import VideoTransformConfig


def _indices(length: int, count: int, strategy: str) -> tuple[int, ...]:
    if length <= 0:
        raise SpecError("decoded video has no frames")
    count = min(count, length)
    if strategy == "head":
        return tuple(range(count))
    if strategy == "tail":
        return tuple(range(length - count, length))
    if count == 1:
        return (length // 2,)
    return tuple(round(index * (length - 1) / (count - 1)) for index in range(count))


class VideoTransform:
    def __init__(self, config: VideoTransformConfig) -> None:
        self.config = config

    def _decode_path(self, path: Path):
        try:
            import av
        except ImportError as exc:
            raise SpecError(
                "video path decoding requires the optional PyAV package; predecoded "
                "Pillow frames remain supported without PyAV"
            ) from exc
        frames = []
        try:
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    frames.append(frame.to_image())
                    if len(frames) > self.config.max_decoded_frames:
                        raise SpecError(
                            f"video exceeds max_decoded_frames={self.config.max_decoded_frames}"
                        )
        except SpecError:
            raise
        except Exception as exc:
            raise SpecError(f"cannot decode video {path}: {exc}") from exc
        return frames

    def apply(self, sample: OmniSample) -> OmniSample:
        try:
            from PIL import Image
        except ImportError as exc:
            raise SpecError("video frame conversion requires Pillow") from exc

        def decode(block: ContentBlock) -> ContentBlock:
            if block.kind != "video":
                return block
            if isinstance(block.value, (str, Path)):
                frames = self._decode_path(Path(block.value))
            elif isinstance(block.value, Sequence) and not isinstance(
                block.value, (str, bytes, bytearray)
            ):
                frames = list(block.value)
            else:
                raise SpecError(
                    "video transform requires a local path or Pillow frame sequence"
                )
            if len(frames) > self.config.max_decoded_frames:
                raise SpecError(
                    f"video exceeds max_decoded_frames={self.config.max_decoded_frames}"
                )
            selected_indices = _indices(
                len(frames), self.config.frames, self.config.sampling
            )
            selected = []
            for index in selected_indices:
                frame = frames[index]
                if not isinstance(frame, Image.Image):
                    raise SpecError("predecoded video values must be Pillow Images")
                selected.append(frame.convert(self.config.mode))
            metadata = dict(block.metadata)
            metadata["selected_frame_indices"] = selected_indices
            metadata["decoded_frame_count"] = len(frames)
            return ContentBlock("video", tuple(selected), metadata)

        return sample.map_blocks(decode)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("sample_transform:trainomni/video@1"),
        config_type=VideoTransformConfig,
        factory=lambda config, context: VideoTransform(config),
        provides=CapabilitySet.of({"sample.video.decoded"}),
        requires=CapabilitySet.of({"sample.media.resolved"}),
    )
