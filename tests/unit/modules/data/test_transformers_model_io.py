from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trainomni.contracts.batch import EncodedSample
from trainomni.contracts.sample import ContentBlock, Message, OmniSample
from trainomni.core.errors import SpecError
from trainomni.modules.data.collation.multimodal.config import MultimodalCollatorConfig
from trainomni.modules.data.collation.multimodal.module import MultimodalCollator
from trainomni.modules.data.model_io.transformers.config import (
    TransformersModelIOConfig,
)
from trainomni.modules.data.model_io.transformers.module import TransformersModelIO
from trainomni.modules.data.supervision.causal_lm.config import CausalSupervisionConfig
from trainomni.modules.data.supervision.causal_lm.module import CausalSupervision
from trainomni.modules.data.supervision.dense_kd.config import DenseKDSupervisionConfig
from trainomni.modules.data.supervision.dense_kd.module import DenseKDSupervision


class ChatProcessor:
    def __init__(self, *, include_mask: bool = True) -> None:
        self.include_mask = include_mask
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        encoded = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 2, 2),
        }
        if self.include_mask:
            encoded["assistant_masks"] = torch.tensor([[0, 0, 1, 1]])
        return encoded


def conversation_sample() -> OmniSample:
    return OmniSample(
        "chat-1",
        (),
        messages=(
            Message(
                "user",
                (
                    ContentBlock("image", Path("image.png")),
                    ContentBlock("text", "What is shown?"),
                ),
            ),
            Message("assistant", (ContentBlock("text", "A diagram."),)),
        ),
    )


def test_chat_template_produces_explicit_assistant_loss_mask() -> None:
    processor = ChatProcessor()
    model_io = TransformersModelIO(
        processor,
        TransformersModelIOConfig(
            processor_name_or_path="unused",
            conversation_mode="required",
        ),
    )
    encoded = model_io.encode(conversation_sample())
    assert processor.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Path("image.png")},
                {"type": "text", "text": "What is shown?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "A diagram."}],
        },
    ]
    assert processor.kwargs["return_assistant_tokens_mask"] is True
    assert "assistant_masks" not in encoded.model_inputs
    assert encoded.supervision["loss_mask"].tolist() == [False, False, True, True]

    supervised = CausalSupervision(CausalSupervisionConfig()).annotate(encoded)
    assert supervised.labels.tolist() == [-100, -100, 3, 4]


def test_chat_template_without_assistant_mask_fails_closed() -> None:
    model_io = TransformersModelIO(
        ChatProcessor(include_mask=False),
        TransformersModelIOConfig(processor_name_or_path="unused"),
    )
    with pytest.raises(SpecError, match="refusing to train on prompt tokens"):
        model_io.encode(conversation_sample())


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[0.0, 2.0, -1.0, float("nan")]]),
        torch.tensor([[0.0, 1.0, float("inf"), 1.0]]),
    ],
)
def test_chat_template_rejects_non_binary_assistant_masks(mask) -> None:
    class InvalidMaskProcessor(ChatProcessor):
        def apply_chat_template(self, messages, **kwargs):
            encoded = super().apply_chat_template(messages, **kwargs)
            encoded["assistant_masks"] = mask
            return encoded

    model_io = TransformersModelIO(
        InvalidMaskProcessor(),
        TransformersModelIOConfig(processor_name_or_path="unused"),
    )
    with pytest.raises(SpecError, match="binary 0/1"):
        model_io.encode(conversation_sample())


@pytest.mark.parametrize(
    "supervision",
    [
        CausalSupervision(CausalSupervisionConfig()),
        DenseKDSupervision(DenseKDSupervisionConfig()),
    ],
)
def test_supervision_rejects_non_binary_external_loss_mask(supervision) -> None:
    cached = {
        "loss_mask": torch.tensor([0.0, 2.0, float("nan")]),
        "teacher_logits": torch.ones(3, 4),
    }
    sample = EncodedSample(
        "mask",
        {"input_ids": torch.tensor([1, 2, 3])},
        cached,
    )
    with pytest.raises(SpecError, match="binary 0/1"):
        supervision.annotate(sample)


def test_conversation_mode_does_not_silently_flatten_samples() -> None:
    disabled = TransformersModelIO(
        ChatProcessor(),
        TransformersModelIOConfig(
            processor_name_or_path="unused", conversation_mode="disabled"
        ),
    )
    with pytest.raises(SpecError, match="disabled conversation_mode"):
        disabled.encode(conversation_sample())

    required = TransformersModelIO(
        ChatProcessor(),
        TransformersModelIOConfig(
            processor_name_or_path="unused", conversation_mode="required"
        ),
    )
    flat = OmniSample("flat", (ContentBlock("text", "plain"),))
    with pytest.raises(SpecError, match="rejects flat content"):
        required.encode(flat)


@pytest.mark.parametrize("image_counts", [(1, 1), (1, 2)])
def test_processor_media_axes_are_preserved_through_collation(image_counts):
    class MediaProcessor:
        def __call__(self, *, text, return_tensors):
            count = int(text)
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "mm_token_type_ids": torch.zeros(1, 3, dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * count),
                "pixel_values": torch.ones(count, 12),
                "video_grid_thw": torch.tensor([[2, 2, 2]]),
                "pixel_values_videos": torch.ones(1, 2, 3, 4, 4),
            }

    model_io = TransformersModelIO(
        MediaProcessor(), TransformersModelIOConfig(processor_name_or_path="unused")
    )
    examples = []
    for index, count in enumerate(image_counts):
        encoded = model_io.encode(
            OmniSample(str(index), (ContentBlock("text", str(count)),))
        )
        assert encoded.model_inputs["input_ids"].shape == (3,)
        assert encoded.model_inputs["mm_token_type_ids"].shape == (3,)
        assert encoded.model_inputs["image_grid_thw"].shape == (count, 3)
        assert encoded.model_inputs["pixel_values"].shape == (count, 12)
        assert encoded.model_inputs["video_grid_thw"].shape == (1, 3)
        assert encoded.model_inputs["pixel_values_videos"].shape == (1, 2, 3, 4, 4)
        examples.append(CausalSupervision(CausalSupervisionConfig()).annotate(encoded))
    collator = MultimodalCollator(
        MultimodalCollatorConfig(
            field_modes={
                "image_grid_thw": "concat", "pixel_values": "concat",
                "video_grid_thw": "concat", "pixel_values_videos": "concat",
            },
        )
    )
    batch = collator.collate(examples)
    assert batch.model_inputs["image_grid_thw"].shape == (sum(image_counts), 3)
    assert batch.model_inputs["pixel_values"].shape == (sum(image_counts), 12)
    assert batch.model_inputs["video_grid_thw"].shape == (2, 3)
    assert batch.model_inputs["pixel_values_videos"].shape == (2, 2, 3, 4, 4)
    assert batch.model_inputs["input_ids"].shape == (2, 3)


def test_processor_batch_axis_fields_are_explicit_and_validated():
    config = TransformersModelIOConfig(
        processor_name_or_path="unused",
        batch_axis_fields=("input_ids", "attention_mask", "pixel_values"),
    )
    encoded = TransformersModelIO(ChatProcessor(), config).encode(conversation_sample())
    assert encoded.model_inputs["pixel_values"].shape == (3, 2, 2)
    with pytest.raises(TypeError, match="sequence"):
        TransformersModelIOConfig(
            processor_name_or_path="unused", batch_axis_fields="input_ids"
        )

    class InvalidBatchProcessor(ChatProcessor):
        def apply_chat_template(self, *args, **kwargs):
            output = super().apply_chat_template(*args, **kwargs)
            output["input_ids"] = torch.ones(2, 4, dtype=torch.long)
            return output

    with pytest.raises(SpecError, match="singleton batch axis"):
        TransformersModelIO(InvalidBatchProcessor(), config).encode(conversation_sample())


@pytest.mark.parametrize("mask", [torch.tensor([1]), torch.tensor([[1]])])
def test_one_token_assistant_mask_remains_one_dimensional(mask):
    model_io = TransformersModelIO(
        None, TransformersModelIOConfig(processor_name_or_path="unused")
    )
    encoded = model_io._normalize_encoded(
        conversation_sample(),
        {"input_ids": torch.tensor([[1]]), "assistant_masks": mask},
        conversation=True,
    )
    assert encoded.supervision["loss_mask"].shape == (1,)
