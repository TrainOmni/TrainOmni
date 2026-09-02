from dataclasses import replace

import pytest
import torch

from trainomni.contracts.batch import SupervisedExample
from trainomni.core.errors import SpecError
from trainomni.modules.data.collation.multimodal.config import MultimodalCollatorConfig
from trainomni.modules.data.collation.padding_free.module import PaddingFreeCollator
from trainomni.modules.data.packing.padding_free.module import PaddingFreePacker
from trainomni.modules.data.packing.sequence.config import SequencePackerConfig
from trainomni.runtime.kernels.attention.varlen import VarlenLayout


def make_pack():
    packer = PaddingFreePacker(
        SequencePackerConfig(
            max_length=128,
            pad_token_id=0,
            max_samples_per_pack=2,
            concat_fields=("pixel_values",),
            offset_fields=("modal_positions",),
        )
    )
    for name, tokens in (("a", [1, 2, 3]), ("b", [4, 5])):
        ids = torch.tensor(tokens)
        emitted = packer.add(
            SupervisedExample(
                name,
                {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "pixel_values": torch.ones(1, 4),
                    "modal_positions": torch.tensor([1]),
                },
                ids.clone(),
            )
        )
    return emitted[0]


def collate(pack):
    return PaddingFreeCollator(
        MultimodalCollatorConfig(field_modes={"pixel_values": "concat", "modal_positions": "stack"})
    ).collate([pack])


def validate(inputs):
    return VarlenLayout.from_packed(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        position_ids=inputs["position_ids"],
        segment_ids=inputs["packed_segment_ids"],
        cu_seqlens=inputs["packed_cu_seqlens"],
    )


def test_unpadded_pack_keeps_boundaries_media_and_linear_size(monkeypatch):
    # Any allocation of a quadratic mask must fail this test, not just be
    # removed from the final mapping after the allocation.
    original = torch.zeros

    def guard(*args, **kwargs):
        shape = args[0] if isinstance(args[0], tuple) else args
        assert len(shape) < 3, f"quadratic mask allocated: {shape}"
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guard)
    pack = make_pack()
    assert pack.labels.tolist() == [1, 2, 3, -100, 5]
    assert pack.model_inputs["modal_positions"].tolist() == [1, 4]
    assert pack.model_inputs["packed_cu_seqlens"].tolist() == [0, 3, 5]
    assert pack.model_inputs["packed_cu_seqlens"].dtype == torch.int32
    assert "packed_attention_mask" not in pack.model_inputs
    batch = collate(pack)
    assert batch.labels.shape == (1, 5)
    assert batch.model_inputs["pixel_values"].shape == (2, 4)
    assert validate(batch.model_inputs).lengths == (3, 2)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("packed_cu_seqlens", torch.tensor([[0, 2, 5]], dtype=torch.int32), "position_ids"),
        ("packed_cu_seqlens", torch.tensor([[0, 3, 6]], dtype=torch.int32), "endpoints"),
        ("packed_cu_seqlens", torch.tensor([[0, 0, 5]], dtype=torch.int32), "positive"),
        ("packed_cu_seqlens", torch.tensor([[0, 3, 5]]), "int32"),
        ("attention_mask", torch.tensor([[1, 1, 1, 1, 0]]), "padding"),
        ("packed_segment_ids", torch.tensor([[0, 0, 0, 0, 1]]), "segment_ids"),
        ("position_ids", torch.tensor([[0, 1, 2, 3, 4]]), "position_ids"),
    ],
)
def test_padding_free_invalid_alignment_fails(field, value, match):
    inputs = dict(collate(make_pack()).model_inputs)
    inputs[field] = value
    with pytest.raises(SpecError, match=match):
        validate(inputs)


def test_padding_free_collator_rejects_repadding_and_multiple_packs():
    config = MultimodalCollatorConfig()
    for invalid in (
        replace(config, pad_to_multiple_of=8),
        replace(config, padding_side="left"),
        replace(config, field_modes={"input_ids": "pad"}),
        replace(config, field_modes={"packed_cu_seqlens": "concat"}),
        replace(config, field_modes={"input_ids": "concat"}),
        replace(config, field_modes={"labels": "list"}),
    ):
        with pytest.raises(SpecError, match="padding-free"):
            PaddingFreeCollator(invalid)
    with pytest.raises(SpecError, match="batch_size=1"):
        PaddingFreeCollator(config).collate([make_pack(), make_pack()])


def test_padding_free_eof_and_buffer_restore_do_not_pad():
    config = SequencePackerConfig(max_length=32, pad_token_id=0)
    packer = PaddingFreePacker(config)
    ids = torch.tensor([1, 2, 3])
    packer.add(SupervisedExample("tail", {"input_ids": ids}, ids.clone()))
    restored = PaddingFreePacker(config)
    restored.load_state_dict(packer.state_dict())
    result = restored.flush()[0]
    assert result.labels.shape == (3,)
    assert result.model_inputs["packed_cu_seqlens"].tolist() == [0, 3]
    assert restored.flush() == ()


def test_padding_free_requires_explicit_model_capability():
    from trainomni.core.capability import CapabilitySet
    from trainomni.core.errors import CapabilityError
    from trainomni.modules.data.packing.padding_free.module import descriptor

    with pytest.raises(CapabilityError, match="model.attention.padding_free"):
        CapabilitySet.of({"data.supervised", "model.attention.packed"}).require(
            descriptor().requires, owner="test"
        )
