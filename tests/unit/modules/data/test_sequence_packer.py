from __future__ import annotations

import pytest
import torch

from trainomni.contracts.batch import SupervisedExample
from trainomni.core.errors import CheckpointError, SpecError
from trainomni.modules.data.packing.sequence.config import SequencePackerConfig
from trainomni.modules.data.packing.sequence.module import SequencePacker


def example(sample_id: str, tokens: list[int], *, modal_position: int) -> SupervisedExample:
    length = len(tokens)
    return SupervisedExample(
        sample_id,
        {
            "input_ids": torch.tensor(tokens),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "token_type_ids": torch.arange(length),
            "modal_positions": torch.tensor([modal_position]),
            "pixel_values": torch.full((1, 2), float(tokens[0])),
        },
        torch.tensor(tokens),
        {"loss_mask": torch.ones(length, dtype=torch.bool)},
    )


def config() -> SequencePackerConfig:
    return SequencePackerConfig(
        max_length=5,
        pad_token_id=0,
        sequence_fields=("token_type_ids",),
        concat_fields=("pixel_values",),
        offset_fields=("modal_positions",),
    )


def test_sequence_packing_masks_boundaries_and_isolates_attention() -> None:
    packer = SequencePacker(config())
    assert packer.add(example("a", [1, 2, 3], modal_position=1)) == ()
    emitted = packer.add(example("b", [4, 5], modal_position=0))
    assert len(emitted) == 1
    packed = emitted[0]
    assert packed.model_inputs["input_ids"].tolist() == [1, 2, 3, 4, 5]
    assert packed.labels.tolist() == [1, 2, 3, -100, 5]
    assert packed.model_inputs["position_ids"].tolist() == [0, 1, 2, 0, 1]
    assert packed.model_inputs["packed_segment_ids"].tolist() == [0, 0, 0, 1, 1]
    assert packed.model_inputs["modal_positions"].tolist() == [1, 3]
    assert packed.model_inputs["pixel_values"].shape == (2, 2)
    mask = packed.model_inputs["packed_attention_mask"][0]
    assert mask[:3, :3].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    assert not bool(mask[:3, 3:].any().item())
    assert not bool(mask[3:, :3].any().item())
    assert mask[3:, 3:].tolist() == [[True, False], [True, True]]
    assert packed.supervision["loss_mask"].tolist() == [True] * 5
    assert packed.supervision["packed_lengths"].tolist() == [3, 2]


def test_sequence_packer_buffer_resumes_exactly_and_validates_state() -> None:
    first = SequencePacker(config())
    assert first.add(example("a", [1, 2, 3, 4], modal_position=0)) == ()
    state = first.state_dict()
    uninterrupted = first.add(example("b", [5, 6], modal_position=1))[0]

    restored = SequencePacker(config())
    restored.load_state_dict(state)
    resumed = restored.add(example("b", [5, 6], modal_position=1))[0]
    assert uninterrupted.sample_id == resumed.sample_id
    assert uninterrupted.model_inputs.keys() == resumed.model_inputs.keys()
    for name in uninterrupted.model_inputs:
        torch.testing.assert_close(
            uninterrupted.model_inputs[name], resumed.model_inputs[name]
        )
    torch.testing.assert_close(uninterrupted.labels, resumed.labels)

    with pytest.raises(CheckpointError, match="cursor is inconsistent"):
        restored.load_state_dict({"buffer": state["buffer"], "tokens": 1})


def test_sequence_packer_rejects_unknown_or_pre_padded_fields() -> None:
    packer = SequencePacker(
        SequencePackerConfig(max_length=5, pad_token_id=0)
    )
    unknown = SupervisedExample(
        "unknown",
        {"input_ids": torch.tensor([1, 2]), "pixel_values": torch.ones(1, 2)},
        torch.tensor([1, 2]),
    )
    assert packer.add(unknown) == ()
    with pytest.raises(SpecError, match="no field policy"):
        packer.add(
            SupervisedExample(
                "flush",
                {"input_ids": torch.tensor([3, 4, 5, 6])},
                torch.tensor([3, 4, 5, 6]),
            )
        )

    padded = SupervisedExample(
        "padded",
        {
            "input_ids": torch.tensor([1, 0]),
            "attention_mask": torch.tensor([1, 0]),
        },
        torch.tensor([1, -100]),
    )
    with pytest.raises(SpecError, match="must not contain padded"):
        SequencePacker(config()).add(padded)
