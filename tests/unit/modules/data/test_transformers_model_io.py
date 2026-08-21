from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trainomni.contracts.sample import ContentBlock, Message, OmniSample
from trainomni.core.errors import SpecError
from trainomni.modules.data.model_io.transformers.config import (
    TransformersModelIOConfig,
)
from trainomni.modules.data.model_io.transformers.module import TransformersModelIO
from trainomni.modules.data.supervision.causal_lm.config import CausalSupervisionConfig
from trainomni.modules.data.supervision.causal_lm.module import CausalSupervision


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
