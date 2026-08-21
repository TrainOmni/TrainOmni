import pytest
import torch
from torch import nn

from trainomni.core.errors import SpecError
from trainomni.modules.parameters.lora.config import LoRAParameterConfig
from trainomni.modules.parameters.lora.module import LoRALinear, LoRAParameterPolicy


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.language = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))

    def forward(self, inputs):
        return self.language(self.encoder(inputs))


def test_lora_injects_only_explicit_linear_targets_and_routes_gradients() -> None:
    torch.manual_seed(4)
    model = Network()
    selection = LoRAParameterPolicy(
        LoRAParameterConfig(
            target_patterns=(r"language\.0", r"language\.2"),
            rank=2,
            alpha=4,
        )
    ).apply(model)
    assert isinstance(model.language[0], LoRALinear)
    assert isinstance(model.language[2], LoRALinear)
    assert isinstance(model.encoder, nn.Linear)
    assert selection.trainable_names == (
        "language.0.lora_a",
        "language.0.lora_b",
        "language.2.lora_a",
        "language.2.lora_b",
    )
    output = model(torch.randn(3, 4)).sum()
    output.backward()
    assert model.language[0].lora_b.grad is not None
    assert model.encoder.weight.grad is None
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name in selection.trainable_names
    )


def test_lora_target_mismatch_fails_closed() -> None:
    with pytest.raises(SpecError, match="matched no Linear"):
        LoRAParameterPolicy(
            LoRAParameterConfig(target_patterns=(r"missing\..*",))
        ).apply(Network())
