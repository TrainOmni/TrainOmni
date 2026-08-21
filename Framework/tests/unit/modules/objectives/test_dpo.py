import pytest
import torch
from torch import nn
from torch.nn import functional

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.forward import ForwardResult
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import ObjectiveError
from trainomni.modules.objectives.dpo.config import DPOConfig
from trainomni.modules.objectives.dpo.module import DPOObjective
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.loop.step import execute_forward_plan


def make_batch(reference_dtype=torch.float32) -> OmniBatch:
    chosen_ids = torch.tensor([[1, 2, 3, 4]])
    rejected_ids = torch.tensor([[1, 2, 4, 3]])
    return OmniBatch(
        sample_ids=("pair",),
        model_inputs={"input_ids": chosen_ids},
        labels=chosen_ids,
        supervision={
            "chosen_inputs": {"input_ids": chosen_ids},
            "rejected_inputs": {"input_ids": rejected_ids},
            "chosen_labels": torch.tensor([[-100, 2, 3, 4]]),
            "rejected_labels": torch.tensor([[-100, 2, 4, 3]]),
            "chosen_reference_logps": torch.tensor(
                [[-0.4, -0.3, -0.2]], dtype=reference_dtype
            ),
            "rejected_reference_logps": torch.tensor(
                [[-0.5, -0.6, -0.4]], dtype=reference_dtype
            ),
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
    objective = DPOObjective(DPOConfig(beta=0.1))
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
    assert int(bundle.metrics["preference_pairs"]) == 1


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
        DPOObjective(DPOConfig()).compute(batch, outputs, ObjectiveContext(0, 0))


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
        objective=DPOObjective(DPOConfig()),
        batch=make_batch(),
        context=ObjectiveContext(0, 0),
        device=DeviceContext("cpu", "fp32"),
    )
    bundle.total.backward()
    assert policy.calls == 2
    assert policy.embedding.weight.grad is not None
    assert torch.count_nonzero(policy.embedding.weight.grad) > 0
