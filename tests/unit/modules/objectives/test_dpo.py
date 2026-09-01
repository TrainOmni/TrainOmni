import pytest
import torch
from torch import nn
from torch.nn import functional

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.cache import (
    current_model_inputs_field,
    digest_tensor,
    model_inputs_digest,
)
from trainomni.contracts.forward import ForwardResult
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import ObjectiveError
from trainomni.modules.objectives._ops.cache_identity import value_digest
from trainomni.modules.objectives.dpo.config import DPOConfig
from trainomni.modules.objectives.dpo.module import DPOObjective
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.loop.step import execute_forward_plan

PRODUCER = "b" * 64


def binding(
    field,
    input_ids,
    labels,
    branch,
    attention_mask=None,
    *,
    producer_model_inputs=None,
    current_model_inputs=None,
):
    explicit_attention = attention_mask is not None
    if attention_mask is None:
        attention_mask = torch.ones_like(labels)
    if producer_model_inputs is None:
        producer_model_inputs = {"input_ids": input_ids[0]}
        if explicit_attention:
            producer_model_inputs["attention_mask"] = attention_mask[0]
    if current_model_inputs is None:
        current_model_inputs = producer_model_inputs
    positions = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
    prefix = f"__cache_identity__{field}__"
    return {
        prefix + "input_ids_sha256": digest_tensor(value_digest(input_ids[0])).unsqueeze(0),
        prefix + "attention_mask_sha256": digest_tensor(
            value_digest(attention_mask[0])
        ).unsqueeze(0),
        prefix + "supervised_positions_sha256": digest_tensor(
            value_digest(positions)
        ).unsqueeze(0),
        prefix + "target_token_ids_sha256": digest_tensor(
            value_digest(labels[0].index_select(0, positions))
        ).unsqueeze(0),
        prefix + "model_inputs_sha256": digest_tensor(
            model_inputs_digest(producer_model_inputs)
        ).unsqueeze(0),
        prefix + "producer_identity_sha256": digest_tensor(PRODUCER).unsqueeze(0),
        prefix + "branch": torch.tensor([branch]),
        current_model_inputs_field(field): digest_tensor(
            model_inputs_digest(current_model_inputs)
        ).unsqueeze(0),
    }


def unbatched(inputs):
    return {
        name: value[0] if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def make_batch(reference_dtype=torch.float32) -> OmniBatch:
    chosen_ids = torch.tensor([[1, 2, 3, 4]])
    rejected_ids = torch.tensor([[1, 2, 4, 3]])
    chosen_labels = torch.tensor([[-100, 2, 3, 4]])
    rejected_labels = torch.tensor([[-100, 2, 4, 3]])
    return OmniBatch(
        sample_ids=("pair",),
        model_inputs={"input_ids": chosen_ids},
        labels=chosen_ids,
        supervision={
            "chosen_inputs": {"input_ids": chosen_ids},
            "rejected_inputs": {"input_ids": rejected_ids},
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "chosen_reference_logps": torch.tensor(
                [[-0.4, -0.3, -0.2]], dtype=reference_dtype
            ),
            "rejected_reference_logps": torch.tensor(
                [[-0.5, -0.6, -0.4]], dtype=reference_dtype
            ),
            **binding("chosen_reference_logps", chosen_ids, chosen_labels, 1),
            **binding("rejected_reference_logps", rejected_ids, rejected_labels, 2),
        },
    )


def sequence_logp(logits, labels):
    shifted = logits[:, :-1].float()
    targets = labels[:, 1:]
    return functional.log_softmax(shifted, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1).sum(-1)


def test_offline_reference_dpo_matches_oracle_and_both_branches_have_gradient() -> None:
    torch.manual_seed(9)
    chosen_logits = torch.randn(1, 4, 6, requires_grad=True)
    rejected_logits = torch.randn(1, 4, 6, requires_grad=True)
    batch = make_batch()
    objective = DPOObjective(
        DPOConfig(reference_producer_identity_sha256=PRODUCER, beta=0.1)
    )
    plan = objective.plan(batch, ObjectiveContext(0, 0))
    assert [request.name for request in plan.requests] == [
        "chosen_policy",
        "rejected_policy",
    ]
    bundle = objective.compute(
        batch,
        {
            "chosen_policy": ForwardResult(
                "chosen_policy", {"logits": chosen_logits}
            ),
            "rejected_policy": ForwardResult(
                "rejected_policy", {"logits": rejected_logits}
            ),
        },
        ObjectiveContext(0, 0),
    )
    chosen_policy = sequence_logp(
        chosen_logits, batch.supervision["chosen_labels"]
    )
    rejected_policy = sequence_logp(
        rejected_logits, batch.supervision["rejected_labels"]
    )
    chosen_reference = batch.supervision["chosen_reference_logps"].sum(-1)
    rejected_reference = batch.supervision["rejected_reference_logps"].sum(-1)
    expected = -functional.logsigmoid(
        0.1
        * (
            (chosen_policy - rejected_policy)
            - (chosen_reference - rejected_reference)
        )
    ).mean()
    torch.testing.assert_close(bundle.total, expected)
    bundle.total.backward()
    assert torch.count_nonzero(chosen_logits.grad) > 0
    assert torch.count_nonzero(rejected_logits.grad) > 0
    assert int(bundle.metrics["preference_pairs"].numerator) == 1


def test_dpo_rejects_non_fp32_reference_cache() -> None:
    batch = make_batch(torch.bfloat16)
    outputs = {
        "chosen_policy": ForwardResult(
            "chosen_policy", {"logits": torch.zeros(1, 4, 6)}
        ),
        "rejected_policy": ForwardResult(
            "rejected_policy", {"logits": torch.zeros(1, 4, 6)}
        ),
    }
    with pytest.raises(ObjectiveError, match="must be FP32"):
        DPOObjective(
            DPOConfig(reference_producer_identity_sha256=PRODUCER)
        ).compute(batch, outputs, ObjectiveContext(0, 0))


def test_dpo_two_forwards_share_one_policy_and_backward_path() -> None:
    class Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(6, 5)
            self.head = nn.Linear(5, 6)
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            return {"logits": self.head(self.embedding(input_ids))}

    policy = Policy()
    bundle = execute_forward_plan(
        model=policy,
        objective=DPOObjective(
            DPOConfig(reference_producer_identity_sha256=PRODUCER)
        ),
        batch=make_batch(),
        context=ObjectiveContext(0, 0),
        device=DeviceContext("cpu", "fp32"),
    )
    bundle.total.backward()
    assert policy.calls == 2
    assert policy.embedding.weight.grad is not None
    assert torch.count_nonzero(policy.embedding.weight.grad) > 0


@pytest.mark.parametrize("corruption", ["prompt", "media", "branch"])
def test_dpo_pair_and_cache_identity_fail_before_policy_forward(corruption) -> None:
    batch = make_batch()
    supervision = dict(batch.supervision)
    chosen_inputs = dict(supervision["chosen_inputs"])
    rejected_inputs = dict(supervision["rejected_inputs"])
    chosen_inputs["pixel_values"] = torch.ones(1, 1, 2)
    rejected_inputs["pixel_values"] = torch.ones(1, 1, 2)
    if corruption == "prompt":
        rejected_inputs["input_ids"] = rejected_inputs["input_ids"].clone()
        rejected_inputs["input_ids"][0, 0] = 5
    elif corruption == "media":
        rejected_inputs["pixel_values"] = torch.zeros(1, 1, 2)
    else:
        field = "__cache_identity__chosen_reference_logps__branch"
        supervision[field] = torch.tensor([2])
    supervision["chosen_inputs"] = chosen_inputs
    supervision["rejected_inputs"] = rejected_inputs
    corrupted = OmniBatch(
        batch.sample_ids,
        batch.model_inputs,
        batch.labels,
        supervision,
    )

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return {"logits": torch.zeros(*kwargs["input_ids"].shape, 6)}

    policy = Policy()
    with pytest.raises(ObjectiveError):
        execute_forward_plan(
            model=policy,
            objective=DPOObjective(
                DPOConfig(reference_producer_identity_sha256=PRODUCER)
            ),
            batch=corrupted,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0


def test_dpo_cannot_declare_media_as_a_branch_varying_field() -> None:
    with pytest.raises(ValueError, match="token-sequence fields"):
        DPOConfig(
            reference_producer_identity_sha256=PRODUCER,
            branch_sequence_fields=("input_ids", "pixel_values"),
        )


def test_dpo_padding_layout_collision_fails_before_policy_forward() -> None:
    producer_attention = torch.tensor([[1, 1, 1, 0]])
    producer_chosen = torch.tensor([[1, 2, 3, 0]])
    producer_rejected = torch.tensor([[1, 2, 4, 0]])
    producer_chosen_labels = torch.tensor([[-100, 2, 3, -100]])
    producer_rejected_labels = torch.tensor([[-100, 2, 4, -100]])
    consumer_attention = torch.tensor([[0, 1, 1, 1]])
    consumer_chosen = torch.tensor([[0, 1, 2, 3]])
    consumer_rejected = torch.tensor([[0, 1, 2, 4]])
    consumer_chosen_labels = torch.tensor([[-100, -100, 2, 3]])
    consumer_rejected_labels = torch.tensor([[-100, -100, 2, 4]])
    batch = OmniBatch(
        sample_ids=("padding-pair",),
        model_inputs={"input_ids": consumer_chosen},
        labels=consumer_chosen,
        supervision={
            "chosen_inputs": {
                "input_ids": consumer_chosen,
                "attention_mask": consumer_attention,
            },
            "rejected_inputs": {
                "input_ids": consumer_rejected,
                "attention_mask": consumer_attention,
            },
            "chosen_labels": consumer_chosen_labels,
            "rejected_labels": consumer_rejected_labels,
            "chosen_reference_logps": torch.zeros(1, 3),
            "rejected_reference_logps": torch.zeros(1, 3),
            **binding(
                "chosen_reference_logps",
                producer_chosen,
                producer_chosen_labels,
                1,
                producer_attention,
            ),
            **binding(
                "rejected_reference_logps",
                producer_rejected,
                producer_rejected_labels,
                2,
                producer_attention,
            ),
        },
    )

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return {"logits": torch.zeros(*kwargs["input_ids"].shape, 6)}

    policy = Policy()
    with pytest.raises(ObjectiveError, match="identity mismatch"):
        execute_forward_plan(
            model=policy,
            objective=DPOObjective(
                DPOConfig(reference_producer_identity_sha256=PRODUCER)
            ),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0


@pytest.mark.parametrize("field", ["token_type_ids", "position_ids", "cache_position"])
def test_dpo_prompt_sequence_context_mismatch_fails_before_forward(field: str) -> None:
    batch = make_batch()
    supervision = dict(batch.supervision)
    chosen_inputs = dict(supervision["chosen_inputs"])
    rejected_inputs = dict(supervision["rejected_inputs"])
    chosen_inputs[field] = torch.arange(4).unsqueeze(0)
    rejected_inputs[field] = chosen_inputs[field].clone()
    rejected_inputs[field][0, 0] += 1
    supervision["chosen_inputs"] = chosen_inputs
    supervision["rejected_inputs"] = rejected_inputs
    corrupted = OmniBatch(
        batch.sample_ids,
        batch.model_inputs,
        batch.labels,
        supervision,
    )

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return {"logits": torch.zeros(*kwargs["input_ids"].shape, 6)}

    policy = Policy()
    with pytest.raises(ObjectiveError, match="common prompt sequence field"):
        execute_forward_plan(
            model=policy,
            objective=DPOObjective(
                DPOConfig(reference_producer_identity_sha256=PRODUCER)
            ),
            batch=corrupted,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0


def test_dpo_response_sequence_context_may_differ() -> None:
    batch = make_batch()
    supervision = dict(batch.supervision)
    chosen_inputs = dict(supervision["chosen_inputs"])
    rejected_inputs = dict(supervision["rejected_inputs"])
    chosen_inputs["position_ids"] = torch.arange(4).unsqueeze(0)
    rejected_inputs["position_ids"] = chosen_inputs["position_ids"].clone()
    rejected_inputs["position_ids"][0, 2:] += 10
    supervision.update(
        binding(
            "chosen_reference_logps",
            chosen_inputs["input_ids"],
            supervision["chosen_labels"],
            1,
            producer_model_inputs=unbatched(chosen_inputs),
        )
    )
    supervision.update(
        binding(
            "rejected_reference_logps",
            rejected_inputs["input_ids"],
            supervision["rejected_labels"],
            2,
            producer_model_inputs=unbatched(rejected_inputs),
        )
    )
    supervision["chosen_inputs"] = chosen_inputs
    supervision["rejected_inputs"] = rejected_inputs
    varied = OmniBatch(
        batch.sample_ids,
        batch.model_inputs,
        batch.labels,
        supervision,
    )
    plan = DPOObjective(
        DPOConfig(reference_producer_identity_sha256=PRODUCER)
    ).plan(varied, ObjectiveContext(0, 0))
    assert len(plan.requests) == 2


def test_dpo_same_logical_prompt_allows_unequal_left_padded_responses() -> None:
    chosen_ids = torch.tensor([[0, 0, 1, 2, 3, 4]])
    rejected_ids = torch.tensor([[0, 1, 2, 5, 6, 7]])
    chosen_attention = torch.tensor([[0, 0, 1, 1, 1, 1]])
    rejected_attention = torch.tensor([[0, 1, 1, 1, 1, 1]])
    chosen_labels = torch.tensor([[-100, -100, -100, -100, 3, 4]])
    rejected_labels = torch.tensor([[-100, -100, -100, 5, 6, 7]])
    chosen_positions = torch.tensor([[0, 0, 0, 1, 2, 3]])
    rejected_positions = torch.tensor([[0, 0, 1, 2, 3, 4]])
    chosen_types = torch.tensor([[0, 0, 7, 7, 8, 8]])
    rejected_types = torch.tensor([[0, 7, 7, 8, 8, 8]])
    chosen_inputs = {
        "input_ids": chosen_ids,
        "attention_mask": chosen_attention,
        "position_ids": chosen_positions,
        "token_type_ids": chosen_types,
        "cache_position": chosen_positions,
    }
    rejected_inputs = {
        "input_ids": rejected_ids,
        "attention_mask": rejected_attention,
        "position_ids": rejected_positions,
        "token_type_ids": rejected_types,
        "cache_position": rejected_positions,
    }
    batch = OmniBatch(
        sample_ids=("unequal-left-pad",),
        model_inputs={"input_ids": chosen_ids},
        labels=chosen_ids,
        supervision={
            "chosen_inputs": chosen_inputs,
            "rejected_inputs": rejected_inputs,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "chosen_reference_logps": torch.zeros(1, 5),
            "rejected_reference_logps": torch.zeros(1, 5),
            **binding(
                "chosen_reference_logps",
                chosen_ids,
                chosen_labels,
                1,
                chosen_attention,
                producer_model_inputs=unbatched(chosen_inputs),
            ),
            **binding(
                "rejected_reference_logps",
                rejected_ids,
                rejected_labels,
                2,
                rejected_attention,
                producer_model_inputs=unbatched(rejected_inputs),
            ),
        },
    )
    plan = DPOObjective(
        DPOConfig(reference_producer_identity_sha256=PRODUCER)
    ).plan(batch, ObjectiveContext(0, 0))
    assert [request.name for request in plan.requests] == [
        "chosen_policy",
        "rejected_policy",
    ]


@pytest.mark.parametrize("field", ["pixel_values", "position_ids"])
def test_dpo_stale_model_input_cache_fails_before_forward(field: str) -> None:
    chosen_ids = torch.tensor([[1, 2, 3, 4]])
    rejected_ids = torch.tensor([[1, 2, 4, 3]])
    chosen_labels = torch.tensor([[-100, 2, 3, 4]])
    rejected_labels = torch.tensor([[-100, 2, 4, 3]])
    producer_chosen = {
        "input_ids": chosen_ids[0],
        "pixel_values": torch.ones(1, 2),
        "position_ids": torch.arange(4),
    }
    producer_rejected = {
        "input_ids": rejected_ids[0],
        "pixel_values": torch.ones(1, 2),
        "position_ids": torch.arange(4),
    }
    current_chosen = {
        name: value.clone() for name, value in producer_chosen.items()
    }
    current_rejected = {
        name: value.clone() for name, value in producer_rejected.items()
    }
    current_chosen[field].reshape(-1)[0] += 1
    current_rejected[field].reshape(-1)[0] += 1
    chosen_inputs = {
        name: value.unsqueeze(0) for name, value in current_chosen.items()
    }
    rejected_inputs = {
        name: value.unsqueeze(0) for name, value in current_rejected.items()
    }
    batch = OmniBatch(
        sample_ids=("stale-pair-input",),
        model_inputs=chosen_inputs,
        labels=chosen_labels,
        supervision={
            "chosen_inputs": chosen_inputs,
            "rejected_inputs": rejected_inputs,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "chosen_reference_logps": torch.zeros(1, 3, dtype=torch.float32),
            "rejected_reference_logps": torch.zeros(1, 3, dtype=torch.float32),
            **binding(
                "chosen_reference_logps",
                chosen_ids,
                chosen_labels,
                1,
                producer_model_inputs=producer_chosen,
                current_model_inputs=current_chosen,
            ),
            **binding(
                "rejected_reference_logps",
                rejected_ids,
                rejected_labels,
                2,
                producer_model_inputs=producer_rejected,
                current_model_inputs=current_rejected,
            ),
        },
    )

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return {"logits": torch.zeros(*kwargs["input_ids"].shape, 6)}

    policy = Policy()
    with pytest.raises(ObjectiveError, match="model_inputs identity mismatch"):
        execute_forward_plan(
            model=policy,
            objective=DPOObjective(
                DPOConfig(reference_producer_identity_sha256=PRODUCER)
            ),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0


def test_dpo_prompt_attention_layout_mismatch_fails_with_self_consistent_cache() -> None:
    chosen_ids = torch.tensor([[9, 1, 2, 3]])
    rejected_ids = torch.tensor([[1, 0, 2, 4]])
    chosen_attention = torch.tensor([[0, 1, 1, 1]])
    rejected_attention = torch.tensor([[1, 0, 1, 1]])
    chosen_labels = torch.tensor([[-100, -100, 2, 3]])
    rejected_labels = torch.tensor([[-100, -100, 2, 4]])
    batch = OmniBatch(
        sample_ids=("attention-context",),
        model_inputs={"input_ids": chosen_ids},
        labels=chosen_ids,
        supervision={
            "chosen_inputs": {
                "input_ids": chosen_ids,
                "attention_mask": chosen_attention,
            },
            "rejected_inputs": {
                "input_ids": rejected_ids,
                "attention_mask": rejected_attention,
            },
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "chosen_reference_logps": torch.zeros(1, 3),
            "rejected_reference_logps": torch.zeros(1, 3),
            **binding(
                "chosen_reference_logps",
                chosen_ids,
                chosen_labels,
                1,
                chosen_attention,
            ),
            **binding(
                "rejected_reference_logps",
                rejected_ids,
                rejected_labels,
                2,
                rejected_attention,
            ),
        },
    )

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return {"logits": torch.zeros(*kwargs["input_ids"].shape, 6)}

    policy = Policy()
    with pytest.raises(ObjectiveError, match="contiguous span"):
        execute_forward_plan(
            model=policy,
            objective=DPOObjective(
                DPOConfig(reference_producer_identity_sha256=PRODUCER)
            ),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0
