"""Use upstream ViT execution, excluding every pretrained merger tensor."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from trainomni.contracts.features import ModalFeatures
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import VisionConfig


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RawVision(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        if not isinstance(inputs, Mapping) or set(inputs) != {
            "hidden_states", "grid_thw", "image_counts"
        }:
            raise SpecError("vision requires hidden_states, grid_thw and per-pack image_counts")
        pixels, grid, counts = (inputs[k] for k in ("hidden_states", "grid_thw", "image_counts"))
        if grid.ndim != 2 or grid.shape[1] != 3 or counts.ndim != 2:
            raise SpecError(f"invalid vision layout: grid={grid.shape}, image_counts={counts.shape}")
        if grid.is_floating_point() or counts.is_floating_point() or grid.dtype == torch.bool:
            raise SpecError("vision grids/counts must use integer dtypes")
        if (grid <= 0).any() or (counts < 0).any() or not (counts.sum(1) > 0).all():
            raise SpecError("vision grids/counts must describe at least one image in each pack")
        if int(counts.sum()) != len(grid) or int(grid.prod(-1).sum()) != len(pixels):
            raise SpecError("vision patch/grid/image_counts identities disagree")
        output = self.model(
            hidden_states=pixels.to(dtype=next(self.model.parameters()).dtype),
            grid_thw=grid, return_dict=True,
        )
        return ModalFeatures(
            embeddings=output.last_hidden_state, grid=grid,
            metadata={"image_counts": counts},
        )

    def enable_activation_checkpointing(self, *, use_reentrant):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )


def build(config, context):
    del context
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    root = Path(config.model_path)
    if not config.weights_sha256 or any(
        Path(name).name != name or digest(root / name) != expected
        for name, expected in config.weights_sha256.items()
    ) or digest(root / "config.json") != config.config_sha256:
        raise SpecError("Qwen pretrained asset digest mismatch")
    model = Qwen3_5VisionModel(AutoConfig.from_pretrained(root, local_files_only=True).vision_config)
    # Upstream forward exposes last_hidden_state before merger. Identity removes
    # the old merger's execution, parameters, checkpoint keys and optimizer state.
    model.merger = nn.Identity()
    state = {}
    prefix = "model.visual."
    for name in sorted(config.weights_sha256):
        with safe_open(root / name, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.startswith(prefix) and not key.startswith(prefix + "merger."):
                    state[key[len(prefix):]] = reader.get_tensor(key)
    model.load_state_dict(state, strict=True, assign=True)
    assert not any("merger." in name for name, _ in model.named_parameters())
    return RawVision(model)


def descriptor():
    return ModuleDescriptor(
        module_id=ModuleId.parse("encoder:example/qwen35_raw_vit@1"),
        config_type=VisionConfig, factory=build,
        provides=CapabilitySet.of({"component.encoder", "encoder.vision", "modal_features.input"}),
    )
