import pytest
import torch
from torch import nn

from trainomni.core.errors import SpecError
from trainomni.modules.model.attention.packed.config import PackedAttentionConfig
from trainomni.modules.model.attention.packed.module import PackedAttentionPolicy
from trainomni.modules.model.attention.policies.config import AttentionPolicyConfig
from trainomni.modules.model.attention.policies.module import ModelDefaultAttentionPolicy
from trainomni.runtime.kernels.activation_checkpointing import (
    apply_activation_checkpointing,
)
from trainomni.runtime.kernels.attention import apply_attention_kernel
from trainomni.specs.run import ActivationCheckpointSpec


class Checkpointable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_reentrant = None

    def enable_activation_checkpointing(self, *, use_reentrant: bool) -> None:
        self.use_reentrant = use_reentrant


class KernelSelectable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = None

    def set_attn_implementation(self, implementation: str) -> None:
        self.kernel = implementation


class Composite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = Checkpointable()
        self.language = Checkpointable()
        self.attention = KernelSelectable()


def test_activation_checkpointing_is_component_scoped_and_non_reentrant() -> None:
    model = Composite()
    applied = apply_activation_checkpointing(
        model,
        ActivationCheckpointSpec(
            enabled=True,
            components=("vision", "language"),
            use_reentrant=False,
        ),
    )
    assert applied == ("vision", "language")
    assert model.vision.use_reentrant is False
    assert model.language.use_reentrant is False


def test_missing_activation_checkpoint_hook_fails_closed() -> None:
    model = Composite()
    with pytest.raises(SpecError, match="does not exist"):
        apply_activation_checkpointing(
            model,
            ActivationCheckpointSpec(enabled=True, components=("missing",)),
        )


def test_attention_kernel_uses_explicit_model_boundary() -> None:
    model = Composite()
    assert apply_attention_kernel(model, "sdpa") == ("attention",)
    assert model.attention.kernel == "sdpa"
    with pytest.raises(SpecError, match="no set_attn_implementation"):
        apply_attention_kernel(nn.Linear(2, 2), "eager")


def test_semantic_attention_policies_validate_default_and_packed_masks() -> None:
    input_ids = torch.tensor([[1, 2, 3, 0]])
    validity = torch.tensor([[1, 1, 1, 0]])
    default = ModelDefaultAttentionPolicy(AttentionPolicyConfig())
    result = default.apply(
        input_ids=input_ids,
        attention_mask=validity,
        modal_positions=None,
        model_inputs={},
    )
    assert result.attention_mask is validity

    segments = torch.tensor([[0, 0, 1, -1]])
    block = torch.tensor(
        [[[[True, False, False, False],
           [True, True, False, False],
           [False, False, True, False],
           [False, False, False, False]]]]
    )
    packed = PackedAttentionPolicy(PackedAttentionConfig(output_format="bool_4d"))
    result = packed.apply(
        input_ids=input_ids,
        attention_mask=validity,
        modal_positions=None,
        model_inputs={
            "packed_attention_mask": block,
            "packed_segment_ids": segments,
        },
    )
    assert result.attention_mask is block
    assert result.consumed_model_inputs == (
        "packed_attention_mask",
        "packed_segment_ids",
    )

    corrupted = block.clone()
    corrupted[0, 0, 2, 0] = True
    with pytest.raises(SpecError, match="disagrees"):
        packed.apply(
            input_ids=input_ids,
            attention_mask=validity,
            modal_positions=None,
            model_inputs={
                "packed_attention_mask": corrupted,
                "packed_segment_ids": segments,
            },
        )
