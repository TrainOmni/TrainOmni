import hashlib
from pathlib import Path

import pytest
from PIL import Image

from trainomni.contracts.sample import ContentBlock, Message, OmniSample
from trainomni.core.errors import SpecError
from trainomni.modules.data.transforms.image.config import ImageTransformConfig
from trainomni.modules.data.transforms.image.module import ImageTransform
from trainomni.modules.data.transforms.media.config import MediaTransformConfig
from trainomni.modules.data.transforms.media.module import MediaTransform
from trainomni.modules.data.transforms.video.config import VideoTransformConfig
from trainomni.modules.data.transforms.video.module import VideoTransform


def test_local_media_identity_and_image_decode(tmp_path: Path) -> None:
    path = tmp_path / "tiny.png"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sample = OmniSample(
        "image",
        (
            ContentBlock(
                "image",
                "tiny.png",
                {"sha256": digest},
            ),
        ),
    )
    resolved = MediaTransform(
        MediaTransformConfig(require_sha256=True), task_root=tmp_path
    ).apply(sample)
    assert resolved.content[0].value == path.resolve()
    decoded = ImageTransform(ImageTransformConfig(mode="RGB")).apply(resolved)
    assert decoded.content[0].value.size == (3, 2)


def test_media_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "tiny.png"
    Image.new("RGB", (1, 1)).save(path)
    sample = OmniSample(
        "image",
        (ContentBlock("image", path, {"sha256": "0" * 64}),),
    )
    with pytest.raises(SpecError, match="digest mismatch"):
        MediaTransform(
            MediaTransformConfig(require_sha256=True), task_root=tmp_path
        ).apply(sample)


def test_media_and_image_transforms_preserve_conversation_structure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chat.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(path)
    sample = OmniSample(
        "chat",
        (),
        messages=(
            Message(
                "user",
                (
                    ContentBlock("image", "chat.png"),
                    ContentBlock("text", "describe"),
                ),
            ),
            Message("assistant", (ContentBlock("text", "square"),)),
        ),
    )
    resolved = MediaTransform(MediaTransformConfig(), task_root=tmp_path).apply(sample)
    decoded = ImageTransform(ImageTransformConfig()).apply(resolved)
    assert tuple(message.role for message in decoded.messages) == ("user", "assistant")
    assert decoded.messages[0].content[0].value.size == (2, 2)
    assert decoded.messages[0].content[1].value == "describe"


def test_video_transform_uniformly_samples_predecoded_frames() -> None:
    frames = tuple(Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(5))
    sample = OmniSample("video", (ContentBlock("video", frames),))
    decoded = VideoTransform(
        VideoTransformConfig(frames=3, sampling="uniform")
    ).apply(sample)

    block = decoded.content[0]
    assert block.metadata["selected_frame_indices"] == (0, 2, 4)
    assert block.metadata["decoded_frame_count"] == 5
    assert [frame.getpixel((0, 0))[0] for frame in block.value] == [0, 2, 4]
