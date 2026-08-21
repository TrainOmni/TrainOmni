import pytest
import torch

from trainomni.contracts.batch import SupervisedExample
from trainomni.core.errors import SpecError
from trainomni.modules.data.collation.multimodal.config import (
    MultimodalCollatorConfig,
)
from trainomni.modules.data.collation.multimodal.module import MultimodalCollator


def test_collator_pads_text_but_stacks_fixed_shape_media() -> None:
    examples = (
        SupervisedExample(
            "short",
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "pixel_values": torch.ones(3, 2, 2),
            },
            torch.tensor([-100, 2]),
        ),
        SupervisedExample(
            "long",
            {
                "input_ids": torch.tensor([3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "pixel_values": torch.zeros(3, 2, 2),
            },
            torch.tensor([-100, 4, 5]),
        ),
    )
    batch = MultimodalCollator(
        MultimodalCollatorConfig(
            pad_token_id=7,
            label_pad_id=-100,
            padding_side="right",
            pad_to_multiple_of=4,
        )
    ).collate(examples)
    assert batch.model_inputs["input_ids"].tolist() == [
        [1, 2, 7, 7],
        [3, 4, 5, 7],
    ]
    assert batch.model_inputs["attention_mask"].tolist() == [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
    ]
    assert batch.labels.tolist() == [
        [-100, 2, -100, -100],
        [-100, 4, 5, -100],
    ]
    assert batch.model_inputs["pixel_values"].shape == (2, 3, 2, 2)


def test_collator_explicit_field_policies_cover_variable_modal_tensors() -> None:
    examples = (
        SupervisedExample(
            "one-image",
            {
                "input_ids": torch.tensor([1, 2]),
                "pixel_values": torch.tensor([[1.0, 2.0]]),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "dense_features": torch.ones(1, 2),
                "media_objects": {"images": ("image-a",)},
            },
            torch.tensor([1, 2]),
        ),
        SupervisedExample(
            "two-images",
            {
                "input_ids": torch.tensor([3]),
                "pixel_values": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                "image_grid_thw": torch.tensor([[1, 3, 3], [1, 4, 4]]),
                "dense_features": torch.zeros(2, 2),
                "media_objects": {"images": ("image-b", "image-c")},
            },
            torch.tensor([3]),
        ),
    )
    collator = MultimodalCollator(
        MultimodalCollatorConfig(
            field_modes={
                "pixel_values": "concat",
                "image_grid_thw": "concat",
                "dense_features": "pad",
                "model_inputs.media_objects.images": "list",
            },
            field_pad_values={"dense_features": -2.0},
        )
    )
    batch = collator.collate(examples)
    assert batch.model_inputs["pixel_values"].shape == (3, 2)
    assert batch.model_inputs["image_grid_thw"].tolist() == [
        [1, 2, 2],
        [1, 3, 3],
        [1, 4, 4],
    ]
    assert batch.model_inputs["dense_features"].tolist() == [
        [[1.0, 1.0], [-2.0, -2.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ]
    assert batch.model_inputs["media_objects"]["images"] == (
        ("image-a",),
        ("image-b", "image-c"),
    )


def test_collator_configured_modes_fail_closed() -> None:
    with pytest.raises(ValueError, match="field_modes.pixel_values"):
        MultimodalCollatorConfig(field_modes={"pixel_values": "guess"})

    examples = (
        SupervisedExample("a", {"pixel_values": torch.ones(1, 2)}, torch.tensor([1])),
        SupervisedExample("b", {"pixel_values": torch.ones(2, 3)}, torch.tensor([1])),
    )
    collator = MultimodalCollator(
        MultimodalCollatorConfig(field_modes={"pixel_values": "concat"})
    )
    with pytest.raises(SpecError, match="equal trailing shapes"):
        collator.collate(examples)
