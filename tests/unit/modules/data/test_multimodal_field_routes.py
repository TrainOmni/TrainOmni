from dataclasses import replace

import pytest
import torch

from trainomni.catalog.builtin import builtin_registry
from trainomni.contracts.batch import SupervisedExample
from trainomni.contracts.sample import ContentBlock, OmniSample
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleKind, ModuleRef
from trainomni.core.resolver import ModuleResolver
from trainomni.modules.data.collation.multimodal.config import MultimodalCollatorConfig
from trainomni.modules.data.collation.multimodal.module import MultimodalCollator
from trainomni.modules.data.model_io.transformers.config import TransformersModelIOConfig
from trainomni.modules.data.model_io.transformers.module import TransformersModelIO
from trainomni.modules.data.packing.sequence.config import SequencePackerConfig
from trainomni.modules.data.packing.sequence.module import SequencePacker
from trainomni.modules.data.supervision.causal_lm.config import CausalSupervisionConfig
from trainomni.modules.data.supervision.causal_lm.module import CausalSupervision


class Processor:
    def __call__(self, **kwargs):
        return {
            "input_ids": torch.tensor([[3, 7, 7, 5]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "pixel_values": torch.arange(24).reshape(8, 3).float(),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
        }


def config():
    return TransformersModelIOConfig(
        processor_name_or_path="unused", modal_token_id=7, unmapped_fields="error",
        field_routes={
            "input_ids": "input_ids", "attention_mask": "attention_mask",
            "pixel_values": "vision.hidden_states", "image_grid_thw": "vision.grid_thw",
        },
        discard_fields=("mm_token_type_ids",),
    )


def example():
    sample = OmniSample("test", (ContentBlock("text", "example"),))
    encoded = TransformersModelIO(Processor(), config()).encode(sample)
    return CausalSupervision(CausalSupervisionConfig()).annotate(encoded)


def test_builtin_routes_fields_and_extracts_positions_without_model_specific_constants():
    value = example()
    assert set(value.model_inputs) == {"input_ids", "attention_mask", "vision", "modal_positions"}
    assert value.model_inputs["vision"]["grid_thw"].shape == (1, 3)
    assert value.model_inputs["modal_positions"].tolist() == [1, 2]
    assert value.labels.tolist() == [3, -100, -100, 5]
    resolver = ModuleResolver(builtin_registry())
    for token, expected in [(None, False), (7, True)]:
        ref = ModuleRef.from_mapping({
            "module": "model_io:trainomni/transformers@1",
            "config": {"processor_name_or_path": "unused", "modal_token_id": token},
        }, field_name="model_io")
        resolved = resolver.resolve(ref, kind=ModuleKind.MODEL_IO)
        assert ("batch.modal_positions" in resolved.descriptor.provides.values) is expected


def test_strict_routes_reject_unknown_fields_and_collisions():
    model_io = TransformersModelIO(Processor(), replace(config(), discard_fields=()))
    with pytest.raises(SpecError, match="mm_token_type_ids.*no field_routes"):
        model_io.encode(OmniSample("test", (ContentBlock("text", "example"),)))
    with pytest.raises(ValueError, match="overlapping field paths"):
        replace(config(), field_routes={"pixel_values": "vision", "image_grid_thw": "vision.grid"})
    with pytest.raises(ValueError, match="input_ids must remain"):
        replace(config(), field_routes={}, discard_fields=("input_ids",))


def test_nested_packing_uses_explicit_leaf_axes_and_keeps_structure():
    value = example()
    packer = SequencePacker(SequencePackerConfig(
        max_length=8, pad_token_id=0,
        concat_fields=("vision.hidden_states", "vision.grid_thw"),
        offset_fields=("modal_positions",),
    ))
    assert packer.add(value) == ()
    state = packer.state_dict()
    second = replace(value, sample_id="second")
    packed = packer.add(second)[0]
    assert packed.model_inputs["vision"]["hidden_states"].shape == (16, 3)
    assert packed.model_inputs["vision"]["grid_thw"].shape == (2, 3)
    assert packed.model_inputs["modal_positions"].tolist() == [1, 2, 5, 6]
    restored = SequencePacker(packer.config)
    restored.load_state_dict(state)
    continuation = restored.add(second)[0]
    assert torch.equal(continuation.model_inputs["vision"]["hidden_states"],
                       packed.model_inputs["vision"]["hidden_states"])
    batch = MultimodalCollator(MultimodalCollatorConfig(field_modes={
        "model_inputs.vision.hidden_states": "concat",
        "model_inputs.vision.grid_thw": "concat",
    })).collate([packed, continuation])
    assert batch.model_inputs["vision"]["hidden_states"].shape == (32, 3)
    assert batch.model_inputs["packed_attention_mask"].shape == (2, 1, 8, 8)


def test_nested_packing_never_guesses_unknown_axes_or_overlapping_policies():
    value = example()
    with pytest.raises(SpecError, match="no field policy.*vision"):
        SequencePacker(SequencePackerConfig(max_length=4, pad_token_id=0)).add(value)
    with pytest.raises(ValueError, match="overlapping field paths"):
        SequencePackerConfig(max_length=8, pad_token_id=0,
                             list_fields=("vision",), concat_fields=("vision.grid_thw",))
    listed = SequencePacker(SequencePackerConfig(
        max_length=4, pad_token_id=0, list_fields=("vision",), offset_fields=("modal_positions",),
    )).add(value)[0]
    assert listed.model_inputs["vision"][0]["grid_thw"].shape == (1, 3)
    bad = SupervisedExample("bad", {"input_ids": torch.tensor([1, 2])}, torch.tensor([1, 2]))
    with pytest.raises(SpecError, match="missing configured fields"):
        SequencePacker(SequencePackerConfig(max_length=2, pad_token_id=0,
                                           concat_fields=("vision.grid",))).add(bad)
