import torch
from torch.nn import functional

from trainomni.contracts.batch import OmniBatch
from trainomni.contracts.forward import ForwardResult
from trainomni.core.context import ObjectiveContext
from trainomni.modules.objectives.causal_lm.config import CausalLMConfig
from trainomni.modules.objectives.causal_lm.module import CausalLMObjective


def test_causal_lm_matches_fp32_masked_token_oracle() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.1, 0.2, 0.3],
                [-0.5, 1.0, 0.5],
                [1.5, -0.5, 0.0],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 2, -100, 0]])
    batch = OmniBatch(
        sample_ids=("sample-0",),
        model_inputs={"input_ids": torch.tensor([[0, 1, 2, 0]])},
        labels=labels,
    )
    objective = CausalLMObjective(CausalLMConfig())
    bundle = objective.compute(
        batch,
        {"policy": ForwardResult("policy", {"logits": logits})},
        ObjectiveContext(global_step=0, micro_step=0),
    )
    shifted_logits = logits[:, :-1, :].reshape(-1, 3)
    shifted_labels = labels[:, 1:].reshape(-1)
    expected = functional.cross_entropy(
        shifted_logits, shifted_labels, ignore_index=-100, reduction="sum"
    ) / shifted_labels.ne(-100).sum()
    torch.testing.assert_close(bundle.total, expected)
    assert int(bundle.metrics["supervised_tokens"].numerator) == 2
    bundle.total.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0
