from pathlib import Path

import torch
from torch import nn

from trainomni.modules.export.safetensors.config import SafetensorsExportConfig
from trainomni.modules.export.safetensors.module import (
    SafetensorsExporter,
    load_safetensors_artifact,
)


class TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.head = nn.Linear(4, 7, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, input_ids):
        return self.head(self.embedding(input_ids))


def test_generic_export_roundtrips_tied_parameters(tmp_path: Path) -> None:
    torch.manual_seed(4)
    source = TiedModel().eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected = source(input_ids).detach()
    artifact = SafetensorsExporter(SafetensorsExportConfig()).export(
        model=source,
        destination=tmp_path / "artifact",
        identity={"task_digest": "a" * 64},
    )

    target = TiedModel().eval()
    load_safetensors_artifact(target, Path(artifact.uri).parent)
    assert torch.equal(target(input_ids), expected)
    assert target.head.weight is target.embedding.weight
