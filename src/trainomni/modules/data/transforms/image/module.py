"""Decode resolved image blocks with Pillow."""

from __future__ import annotations

from pathlib import Path

from trainomni.contracts.sample import ContentBlock, OmniSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import ImageTransformConfig


class ImageTransform:
    def __init__(self, config: ImageTransformConfig) -> None:
        self.config = config

    def apply(self, sample: OmniSample) -> OmniSample:
        try:
            from PIL import Image
        except ImportError as exc:
            raise SpecError("image decoding requires Pillow") from exc
        def decode(block: ContentBlock) -> ContentBlock:
            if block.kind != "image":
                return block
            value = block.value
            if isinstance(value, (str, Path)):
                try:
                    with Image.open(value) as opened:
                        image = opened.convert(self.config.mode)
                        image.load()
                except Exception as exc:
                    raise SpecError(f"cannot decode image {value}: {exc}") from exc
            elif isinstance(value, Image.Image):
                image = value.convert(self.config.mode)
            else:
                raise SpecError(
                    "image transform requires a path or Pillow Image value"
                )
            if (
                self.config.max_pixels is not None
                and image.width * image.height > self.config.max_pixels
            ):
                raise SpecError(
                    f"image {image.width}x{image.height} exceeds max_pixels"
                )
            return ContentBlock(
                kind="image",
                value=image,
                metadata=block.metadata,
            )

        return sample.map_blocks(decode)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("sample_transform:trainomni/image@1"),
        config_type=ImageTransformConfig,
        factory=lambda config, context: ImageTransform(config),
        provides=CapabilitySet.of({"sample.image.decoded"}),
        requires=CapabilitySet.of({"sample.media.resolved"}),
    )
