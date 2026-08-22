"""Real task-local composite model used to validate TrainOmni's public ABI."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from trainomni.core.capability import CapabilitySet
from trainomni.contracts.distribution import DistributionHints
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.export.safetensors.module import load_safetensors_artifact


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    vision_checkpoint: str
    language_checkpoint: str
    load_dtype: str = "bf16"
    initial_artifact: str | None = None
    initial_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.load_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("load_dtype must be bf16, fp16, or fp32")
        if bool(self.initial_artifact) != bool(self.initial_artifact_sha256):
            raise ValueError(
                "initial_artifact and initial_artifact_sha256 must be set together"
            )


class QwenVisionMiniCPM(nn.Module):
    """Visual-prefix causal LM whose output logits stay text-position aligned."""

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module) -> None:
        super().__init__()
        vision_width = int(vision_encoder.config.out_hidden_size)
        language_width = int(language_model.config.hidden_size)
        self.vision_encoder = vision_encoder
        self.connector = nn.Sequential(
            nn.LayerNorm(vision_width),
            nn.Linear(vision_width, language_width),
        )
        self.language_model = language_model
        self.language_model.config.use_cache = False

    def distribution_hints(self) -> DistributionHints:
        return DistributionHints(
            fsdp_units=(
                *(f"vision_encoder.blocks.{index}" for index in range(len(self.vision_encoder.blocks))),
                *(
                    f"language_model.model.layers.{index}"
                    for index in range(len(self.language_model.model.layers))
                ),
            )
        )

    def _vision_features(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        frozen = not any(
            parameter.requires_grad for parameter in self.vision_encoder.parameters()
        )
        context = torch.no_grad() if frozen else torch.enable_grad()
        with context:
            output = self.vision_encoder(
                pixel_values.to(dtype=self.vision_encoder.dtype),
                grid_thw=image_grid_thw,
                return_dict=True,
            )
        merged = output.pooler_output
        merge_size = int(self.vision_encoder.spatial_merge_size)
        split_sizes = (
            image_grid_thw.prod(dim=-1) // (merge_size * merge_size)
        ).tolist()
        return list(torch.split(merged, split_sizes, dim=0))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_counts: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must be aligned [batch, text]")
        image_features = self._vision_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        counts = [int(value) for value in image_counts.detach().cpu().tolist()]
        if len(counts) != input_ids.shape[0] or sum(counts) != len(image_features):
            raise ValueError("image_counts does not align with batch/images")

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        connector_dtype = next(self.connector.parameters()).dtype
        sequences: list[torch.Tensor] = []
        prefix_lengths: list[int] = []
        text_lengths: list[int] = []
        cursor = 0
        for row, count in enumerate(counts):
            if count < 1:
                raise ValueError("every validation sample requires at least one image")
            visual = torch.cat(image_features[cursor : cursor + count], dim=0)
            cursor += count
            projected = self.connector(visual.to(dtype=connector_dtype)).to(
                dtype=text_embeddings.dtype
            )
            text_length = int(attention_mask[row].sum().item())
            sequences.append(
                torch.cat((projected, text_embeddings[row, :text_length]), dim=0)
            )
            prefix_lengths.append(int(projected.shape[0]))
            text_lengths.append(text_length)

        embeddings = nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0.0
        )
        combined_mask = torch.zeros(
            embeddings.shape[:2], dtype=attention_mask.dtype, device=embeddings.device
        )
        for row, sequence in enumerate(sequences):
            combined_mask[row, : sequence.shape[0]] = 1
        output = self.language_model(
            inputs_embeds=embeddings,
            attention_mask=combined_mask,
            use_cache=False,
            return_dict=True,
        )

        # Objectives consume labels aligned to the text tokens. Visual prefix logits
        # are an internal fusion detail and must not leak into the objective ABI.
        aligned_rows = []
        text_width = input_ids.shape[1]
        for row, (prefix_length, text_length) in enumerate(
            zip(prefix_lengths, text_lengths, strict=True)
        ):
            aligned = output.logits.new_zeros((text_width, output.logits.shape[-1]))
            aligned[:text_length] = output.logits[
                row, prefix_length : prefix_length + text_length
            ]
            aligned_rows.append(aligned)
        return SimpleNamespace(logits=torch.stack(aligned_rows, dim=0))


def _artifact_digest(path: Path) -> str:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise SpecError(f"initial artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read initial artifact manifest: {exc}") from exc
    digest = manifest.get("sha256")
    if not isinstance(digest, str):
        raise SpecError("initial artifact manifest has no sha256")
    return digest


def _factory(config: Config, context) -> QwenVisionMiniCPM:
    from transformers import (
        LlamaForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )

    vision_path = Path(config.vision_checkpoint).resolve()
    language_path = Path(config.language_checkpoint).resolve()
    if not vision_path.is_dir() or not language_path.is_dir():
        raise SpecError("real validation checkpoints must exist locally")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[config.load_dtype]

    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
        vision_path,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    vision_encoder = qwen.model.visual
    del qwen
    gc.collect()
    language_model = LlamaForCausalLM.from_pretrained(
        language_path,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = QwenVisionMiniCPM(vision_encoder, language_model)
    if config.initial_artifact is not None:
        artifact = Path(config.initial_artifact)
        if not artifact.is_absolute():
            if context.task_root is None:
                raise SpecError("relative initial artifact requires a task root")
            artifact = context.task_root / artifact
        artifact = artifact.resolve()
        observed = _artifact_digest(artifact)
        if observed != config.initial_artifact_sha256:
            raise SpecError(
                "initial artifact identity mismatch: "
                f"expected {config.initial_artifact_sha256}, got {observed}"
            )
        load_safetensors_artifact(model, artifact)
    return model


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("model:validation/qwen35_vision_minicpm5@1"),
        config_type=Config,
        factory=_factory,
        provides=CapabilitySet.of(
            {"model.monolithic", "model.output.logits", "model.parameters"}
        ),
    )
