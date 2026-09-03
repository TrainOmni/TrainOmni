"""Fresh merger weights plus explicit reconstruction of per-pack feature rows."""

import torch
from torch import nn
from torch.nn import functional as F

from trainomni.contracts.features import ModalFeatures
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import MergerConfig


class MergerConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchMerger

        self.config = config
        vision_config = Qwen3_5VisionConfig(
            hidden_size=config.input_dim, out_hidden_size=config.output_dim,
            spatial_merge_size=config.spatial_merge_size,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.merger = Qwen3_5VisionPatchMerger(vision_config)
            # Match upstream model initialization; no pretrained merger is read.
            for module in self.merger.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=config.initializer_range)
                    nn.init.zeros_(module.bias)

    def forward(self, features):
        raw, grid = features.embeddings, features.grid
        counts = features.metadata.get("image_counts")
        if raw.ndim != 2 or raw.shape[1] != self.config.input_dim:
            raise SpecError(f"merger requires raw ViT [patches,{self.config.input_dim}]; got {raw.shape}")
        merge = self.config.spatial_merge_size
        if counts is None or counts.ndim != 2 or int(counts.sum()) != len(grid):
            raise SpecError("merger requires per-pack image counts matching the grid rows")
        if (grid[:, 1:] % merge).any():
            raise SpecError("vision height/width grids must be divisible by spatial_merge_size")
        per_image = (grid.prod(-1) // (merge * merge)).tolist()
        per_pack = []
        cursor = 0
        for number in counts.sum(1).tolist():
            per_pack.append(sum(per_image[cursor:cursor + number]))
            cursor += number
        merged = self.merger(raw.to(dtype=self.merger.linear_fc1.weight.dtype))
        if sum(per_pack) != len(merged) or min(per_pack) <= 0:
            raise SpecError("merger output count disagrees with the packed image boundaries")
        width = max(per_pack)
        embeddings = torch.stack([
            F.pad(part, (0, 0, 0, width - len(part)))
            for part in merged.split(per_pack)
        ])
        mask = torch.arange(width, device=merged.device)[None] < torch.tensor(
            per_pack, device=merged.device
        )[:, None]
        return ModalFeatures(
            embeddings=embeddings, mask=mask, grid=grid,
            metadata={**features.metadata, "merged_tokens_per_pack": tuple(per_pack)},
        )


def descriptor():
    return ModuleDescriptor(
        module_id=ModuleId.parse("connector:example/qwen35_merger@1"),
        config_type=MergerConfig, factory=lambda config, context: MergerConnector(config),
        provides=CapabilitySet.of({"component.connector", "modal_features.projected"}),
    )
