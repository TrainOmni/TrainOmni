"""Resolve local media paths and optionally enforce per-block SHA-256."""

from __future__ import annotations

import hashlib
from pathlib import Path

from trainomni.contracts.sample import ContentBlock, OmniSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import MediaTransformConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MediaTransform:
    def __init__(self, config: MediaTransformConfig, *, task_root: Path | None) -> None:
        self.config = config
        self.task_root = task_root

    def apply(self, sample: OmniSample) -> OmniSample:
        def resolve(block: ContentBlock) -> ContentBlock:
            if block.kind == "text" or not isinstance(block.value, (str, Path)):
                return block
            path = Path(block.value)
            if not path.is_absolute():
                if self.task_root is None:
                    raise SpecError("relative media paths require a task root")
                path = self.task_root / path
            path = path.resolve()
            if not path.is_file():
                raise SpecError(f"media file does not exist: {path}")
            expected = block.metadata.get("sha256")
            if self.config.require_sha256 and expected is None:
                raise SpecError(f"media block {path} is missing metadata.sha256")
            if expected is not None:
                actual = _sha256(path)
                if actual != expected:
                    raise SpecError(
                        f"media digest mismatch for {path}: expected {expected}, got {actual}"
                    )
            return ContentBlock(
                kind=block.kind,
                value=path,
                metadata=block.metadata,
            )

        return sample.map_blocks(resolve)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("sample_transform:trainomni/media@1"),
        config_type=MediaTransformConfig,
        factory=lambda config, context: MediaTransform(
            config, task_root=context.task_root
        ),
        provides=CapabilitySet.of({"sample.media.resolved"}),
        requires=CapabilitySet.of({"data.sample.omni"}),
    )
