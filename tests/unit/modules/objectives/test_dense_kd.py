import pytest
import torch
from torch.nn import functional

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.forward import ForwardResult
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import ObjectiveError
from trainomni.modules.objectives.dense_kd.config import DenseKDConfig
from trainomni.modules.objectives.dense_kd.module import DenseKDObjective


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
    batch = OmniBatch(
        sample_ids=("kd",),
        model_inputs={"input_ids": torch.tensor([[0, 1, 2, 0]])},
        labels=labels,
        supervision={"teacher_logits": teacher},
    )
    config = DenseKDConfig(ce_weight=0.5, kd_weight=0.5, temperature=2.0)
    bundle = DenseKDObjective(config).compute(
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
        DenseKDObjective(DenseKDConfig()).compute(
            batch,
            {"policy": ForwardResult("policy", {"logits": student})},
            ObjectiveContext(global_step=0, micro_step=0),
        )
