import pytest
import torch
from torch import nn
from torch.nn import functional

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.forward import ForwardResult
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import ObjectiveError
from trainomni.modules.objectives._ops.cache_identity import digest_tensor, value_digest
from trainomni.modules.objectives.dense_kd.config import DenseKDConfig
from trainomni.modules.objectives.dense_kd.module import DenseKDObjective
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.loop.step import execute_forward_plan

PRODUCER = "a" * 64


def cache_identity(input_ids, labels, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(labels)
    positions = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
    prefix = "__cache_identity__teacher_logits__"
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
        prefix + "producer_identity_sha256": digest_tensor(PRODUCER).unsqueeze(0),
        prefix + "branch": torch.tensor([0]),
    }


def test_dense_kd_matches_fp32_oracle_and_preserves_student_gradient() -> None:
    student = torch.tensor(
        [
            [
                [1.0, 0.0, -1.0],
                [0.2, 0.4, -0.2],
                [-0.1, 0.3, 0.8],
                [0.0, 0.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[0.5, 0.2, -0.1], [0.7, -0.5, 0.1], [0.2, 0.3, 0.4]]],
        dtype=torch.bfloat16,
    )
    labels = torch.tensor([[-100, 1, -100, 2]])
    input_ids = torch.tensor([[0, 1, 2, 0]])
    batch = OmniBatch(
        sample_ids=("kd",),
        model_inputs={"input_ids": input_ids},
        labels=labels,
        supervision={"teacher_logits": teacher, **cache_identity(input_ids, labels)},
    )
    config = DenseKDConfig(
        producer_identity_sha256=PRODUCER,
        ce_weight=0.5,
        kd_weight=0.5,
        temperature=2.0,
    )
    objective = DenseKDObjective(config)
    objective.plan(batch, ObjectiveContext(global_step=0, micro_step=0))
    bundle = objective.compute(
        batch,
        {"policy": ForwardResult("policy", {"logits": student})},
        ObjectiveContext(global_step=0, micro_step=0),
    )
    shifted_student = student[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    ce_tokens = functional.cross_entropy(
        shifted_student.reshape(-1, 3),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    ce = ce_tokens[mask].mean()
    teacher_float = teacher.float()
    teacher_probs = functional.softmax(teacher_float / 2.0, dim=-1)
    kl_tokens = (
        teacher_probs
        * (
            functional.log_softmax(teacher_float / 2.0, dim=-1)
            - functional.log_softmax(shifted_student / 2.0, dim=-1)
        )
    ).sum(dim=-1)
    kl = kl_tokens[mask].mean()
    expected = 0.5 * ce + 0.5 * 4.0 * kl
    torch.testing.assert_close(bundle.total, expected)
    bundle.total.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) > 0


def test_dense_kd_alignment_mismatch_fails_closed() -> None:
    student = torch.zeros(1, 4, 3)
    batch = OmniBatch(
        sample_ids=("kd",),
        model_inputs={"input_ids": torch.zeros(1, 4, dtype=torch.long)},
        labels=torch.zeros(1, 4, dtype=torch.long),
        supervision={"teacher_logits": torch.zeros(1, 2, 3)},
    )
    with torch.no_grad(), pytest.raises(ObjectiveError, match="align"):
        DenseKDObjective(DenseKDConfig(producer_identity_sha256=PRODUCER)).compute(
            batch,
            {"policy": ForwardResult("policy", {"logits": student})},
            ObjectiveContext(global_step=0, micro_step=0),
        )


def test_dense_kd_wrong_self_consistent_cache_identity_fails_before_forward() -> None:
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            return {"logits": torch.zeros(*input_ids.shape, 3)}

    input_ids = torch.tensor([[0, 1, 2, 0]])
    labels = torch.tensor([[-100, 1, -100, 2]])
    supervision = {
        "teacher_logits": torch.zeros(1, 3, 3),
        **cache_identity(input_ids, labels),
    }
    field = "__cache_identity__teacher_logits__target_token_ids_sha256"
    supervision[field] = supervision[field].clone()
    supervision[field][0, 0] ^= 1
    batch = OmniBatch(
        sample_ids=("wrong-cache",),
        model_inputs={"input_ids": input_ids},
        labels=labels,
        supervision=supervision,
    )
    policy = Policy()
    with pytest.raises(ObjectiveError, match="target.*identity mismatch"):
        execute_forward_plan(
            model=policy,
            objective=DenseKDObjective(
                DenseKDConfig(producer_identity_sha256=PRODUCER)
            ),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0


@pytest.mark.parametrize(
    "corruption",
    ["input_ids", "attention_mask", "positions", "producer", "branch"],
)
def test_dense_kd_all_cache_bindings_fail_before_forward(corruption: str) -> None:
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            return {"logits": torch.zeros(*input_ids.shape, 3)}

    input_ids = torch.tensor([[0, 1, 2, 0]])
    labels = torch.tensor([[-100, 1, -100, 2]])
    attention_mask = torch.ones_like(labels)
    supervision = {
        "teacher_logits": torch.zeros(1, 3, 3),
        **cache_identity(input_ids, labels, attention_mask),
    }
    if corruption == "input_ids":
        input_ids = input_ids.clone()
        input_ids[0, 0] = 2
    elif corruption == "attention_mask":
        attention_mask[0, 2] = 0
    elif corruption == "positions":
        field = "__cache_identity__teacher_logits__supervised_positions_sha256"
        supervision[field] = supervision[field].clone()
        supervision[field][0, 0] ^= 1
    elif corruption == "producer":
        field = "__cache_identity__teacher_logits__producer_identity_sha256"
        supervision[field] = digest_tensor("c" * 64).unsqueeze(0)
    else:
        supervision["__cache_identity__teacher_logits__branch"] = torch.tensor([1])
    batch = OmniBatch(
        sample_ids=("wrong-cache",),
        model_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        labels=labels,
        supervision=supervision,
    )
    policy = Policy()
    with pytest.raises(ObjectiveError, match="identity mismatch"):
        execute_forward_plan(
            model=policy,
            objective=DenseKDObjective(
                DenseKDConfig(producer_identity_sha256=PRODUCER)
            ),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0


def test_dense_kd_padding_layout_collision_fails_before_forward() -> None:
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return {"logits": torch.zeros(*kwargs["input_ids"].shape, 5)}

    producer_ids = torch.tensor([[1, 2, 3, 0]])
    producer_attention = torch.tensor([[1, 1, 1, 0]])
    producer_labels = torch.tensor([[-100, 2, 3, -100]])
    consumer_ids = torch.tensor([[0, 1, 2, 3]])
    consumer_attention = torch.tensor([[0, 1, 1, 1]])
    consumer_labels = torch.tensor([[-100, -100, 2, 3]])
    batch = OmniBatch(
        sample_ids=("padding-collision",),
        model_inputs={
            "input_ids": consumer_ids,
            "attention_mask": consumer_attention,
        },
        labels=consumer_labels,
        supervision={
            "teacher_logits": torch.zeros(1, 3, 5),
            **cache_identity(
                producer_ids,
                producer_labels,
                producer_attention,
            ),
        },
    )
    policy = Policy()
    with pytest.raises(ObjectiveError, match="identity mismatch"):
        execute_forward_plan(
            model=policy,
            objective=DenseKDObjective(
                DenseKDConfig(producer_identity_sha256=PRODUCER)
            ),
            batch=batch,
            context=ObjectiveContext(0, 0),
            device=DeviceContext("cpu", "fp32"),
        )
    assert policy.calls == 0
