from pathlib import Path

import pytest
import torch
from torch import nn

from trainomni.core.errors import SpecError
from trainomni.modules.export.lora_adapter.config import LoRAAdapterExportConfig
from trainomni.modules.export.lora_adapter.module import (
    LoRAAdapterExporter,
    load_lora_adapter,
)
from trainomni.modules.parameters.lora.config import LoRAParameterConfig
from trainomni.modules.parameters.lora.module import LoRAParameterPolicy


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def forward(self, inputs):
        return self.projection(inputs)


def inject(model):
    LoRAParameterPolicy(
        LoRAParameterConfig(target_patterns=(r"projection",), rank=2, alpha=4)
    ).apply(model)
    return model


def test_lora_adapter_export_load_roundtrip_preserves_output(tmp_path: Path) -> None:
    torch.manual_seed(3)
    source = inject(Network())
    source.projection.lora_b.data.normal_()
    inputs = torch.randn(2, 4)
    expected = source(inputs).detach()

    artifact = LoRAAdapterExporter(LoRAAdapterExportConfig()).export(
        model=source,
        destination=tmp_path / "adapter",
        identity={"base_model": "fixture"},
    )
    torch.manual_seed(3)
    target = inject(Network())
    load_lora_adapter(target, artifact.uri)

    assert torch.equal(target(inputs), expected)
    assert artifact.kind == "trainomni_linear_lora"


def test_lora_adapter_target_mismatch_fails_closed(tmp_path: Path) -> None:
    source = inject(Network())
    artifact = LoRAAdapterExporter(LoRAAdapterExportConfig()).export(
        model=source,
        destination=tmp_path / "adapter",
        identity={"base_model": "fixture"},
    )
    with pytest.raises(SpecError, match="targets differ"):
        load_lora_adapter(Network(), artifact.uri)


def test_lora_adapter_base_weight_mismatch_fails_closed(tmp_path: Path) -> None:
    torch.manual_seed(8)
    source = inject(Network())
    artifact = LoRAAdapterExporter(LoRAAdapterExportConfig()).export(
        model=source,
        destination=tmp_path / "adapter",
        identity={"base_model": "fixture"},
    )
    torch.manual_seed(8)
    target = inject(Network())
    target.projection.base.weight.data[0, 0] += 1
    with pytest.raises(SpecError, match="metadata differs"):
        load_lora_adapter(target, artifact.uri)
