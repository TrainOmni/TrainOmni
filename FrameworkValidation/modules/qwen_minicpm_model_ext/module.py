"""Real VLM adapter with semantic-attention and packed-sequence support."""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


class QwenVisionMiniCPMExtended(nn.Module):
    """Visual-prefix LM exposing explicit semantic and kernel attention boundaries."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        language_model: nn.Module,
        *,
        attention_policy: Any | None,
    ) -> None:
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
        self.attention_policy = attention_policy
        self.attention_kernel = "eager"

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

    def set_attn_implementation(self, implementation: str) -> None:
        """Forward the runtime kernel choice to both upstream Transformer towers."""

        configured = []
        for name, module in (
            ("vision_encoder", self.vision_encoder),
            ("language_model", self.language_model),
        ):
            setter = getattr(module, "set_attn_implementation", None)
            if callable(setter):
                setter(implementation)
                configured.append(name)
        if "language_model" not in configured:
            raise ValueError("language model has no dynamic attention-kernel boundary")
        self.attention_kernel = implementation

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

    def _semantic_attention(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_attention_mask: torch.Tensor | None,
        packed_segment_ids: torch.Tensor | None,
        model_inputs: Mapping[str, Any],
    ):
        if self.attention_policy is None:
            if packed_attention_mask is not None or packed_segment_ids is not None:
                raise SpecError("packed inputs require an explicit semantic attention policy")
            return attention_mask
        policy_inputs = dict(model_inputs)
        if packed_attention_mask is not None:
            policy_inputs["packed_attention_mask"] = packed_attention_mask
        if packed_segment_ids is not None:
            policy_inputs["packed_segment_ids"] = packed_segment_ids
        result = self.attention_policy.apply(
            input_ids=input_ids,
            attention_mask=attention_mask,
            modal_positions=None,
            model_inputs=policy_inputs,
        )
        if result.model_kwargs:
            raise SpecError("validation model does not accept attention model_kwargs")
        return result.attention_mask

    def _project_images(
        self,
        image_features: list[torch.Tensor],
        *,
        cursor: int,
        count: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        if count < 1:
            raise ValueError("every validation sample requires at least one visual input")
        visual = torch.cat(image_features[cursor : cursor + count], dim=0)
        connector_dtype = next(self.connector.parameters()).dtype
        projected = self.connector(visual.to(dtype=connector_dtype)).to(dtype=dtype)
        return projected, cursor + count

    def _forward_regular(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_features: list[torch.Tensor],
        counts: list[int],
    ) -> SimpleNamespace:
        if len(counts) != input_ids.shape[0] or sum(counts) != len(image_features):
            raise ValueError("image_counts does not align with batch/images")
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        sequences = []
        prefix_lengths = []
        text_lengths = []
        cursor = 0
        for row, count in enumerate(counts):
            projected, cursor = self._project_images(
                image_features,
                cursor=cursor,
                count=count,
                dtype=text_embeddings.dtype,
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
        rows = []
        text_width = input_ids.shape[1]
        for row, (prefix_length, text_length) in enumerate(
            zip(prefix_lengths, text_lengths, strict=True)
        ):
            aligned = output.logits.new_zeros((text_width, output.logits.shape[-1]))
            aligned[:text_length] = output.logits[
                row, prefix_length : prefix_length + text_length
            ]
            rows.append(aligned)
        return SimpleNamespace(logits=torch.stack(rows, dim=0))

    def _forward_packed(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        semantic_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        image_features: list[torch.Tensor],
        counts: list[int],
    ) -> SimpleNamespace:
        if input_ids.shape[0] != 1:
            raise SpecError("real packed validation currently requires batch size one")
        valid = attention_mask[0].bool()
        observed = segment_ids[0, valid]
        if observed.numel() == 0:
            raise SpecError("packed batch has no valid tokens")
        segment_count = int(observed.max().item()) + 1
        expected = torch.arange(segment_count, device=observed.device)
        if not torch.equal(torch.unique_consecutive(observed), expected):
            raise SpecError("packed segment ids must be contiguous and ordered from zero")
        if len(counts) != segment_count or sum(counts) != len(image_features):
            raise SpecError("packed image_counts does not align with segments/images")

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        sequences = []
        text_ranges = []
        cursor = 0
        for segment, count in enumerate(counts):
            token_positions = torch.nonzero(
                (segment_ids[0] == segment) & valid, as_tuple=False
            ).flatten()
            if token_positions.numel() == 0:
                raise SpecError("packed segment has no text tokens")
            projected, cursor = self._project_images(
                image_features,
                cursor=cursor,
                count=count,
                dtype=text_embeddings.dtype,
            )
            sequences.append(
                torch.cat((projected, text_embeddings[0, token_positions]), dim=0)
            )
            text_ranges.append((token_positions, int(projected.shape[0])))

        embeddings = torch.cat(sequences, dim=0).unsqueeze(0)
        expanded_bool = torch.zeros(
            (1, 1, embeddings.shape[1], embeddings.shape[1]),
            dtype=torch.bool,
            device=embeddings.device,
        )
        position_parts = []
        offset = 0
        for sequence in sequences:
            stop = offset + sequence.shape[0]
            expanded_bool[:, :, offset:stop, offset:stop] = torch.ones(
                (sequence.shape[0], sequence.shape[0]),
                dtype=torch.bool,
                device=embeddings.device,
            ).tril()
            position_parts.append(
                torch.arange(sequence.shape[0], device=embeddings.device)
            )
            offset = stop
        if semantic_mask.dtype is torch.bool:
            expanded_mask = expanded_bool
        else:
            expanded_mask = torch.zeros(
                expanded_bool.shape, dtype=torch.float32, device=embeddings.device
            ).masked_fill(~expanded_bool, torch.finfo(torch.float32).min)
        position_ids = torch.cat(position_parts).unsqueeze(0)
        output = self.language_model(
            inputs_embeds=embeddings,
            attention_mask=expanded_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

        aligned = output.logits.new_zeros(
            (input_ids.shape[1], output.logits.shape[-1])
        )
        offset = 0
        for sequence, (token_positions, prefix_length) in zip(
            sequences, text_ranges, strict=True
        ):
            text_length = int(token_positions.numel())
            aligned[token_positions] = output.logits[
                0, offset + prefix_length : offset + prefix_length + text_length
            ]
            offset += sequence.shape[0]
        return SimpleNamespace(logits=aligned.unsqueeze(0))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_counts: torch.Tensor,
        packed_attention_mask: torch.Tensor | None = None,
        packed_segment_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must be aligned [batch, text]")
        semantic_mask = self._semantic_attention(
            input_ids=input_ids,
            attention_mask=attention_mask,
            packed_attention_mask=packed_attention_mask,
            packed_segment_ids=packed_segment_ids,
            model_inputs=kwargs,
        )
        image_features = self._vision_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        counts = [int(value) for value in image_counts.detach().cpu().flatten().tolist()]
        if packed_segment_ids is None:
            if semantic_mask.ndim != 2:
                raise SpecError("non-packed attention policy must return a 2D mask")
            return self._forward_regular(
                input_ids=input_ids,
                attention_mask=semantic_mask,
                image_features=image_features,
                counts=counts,
            )
        if packed_attention_mask is None or semantic_mask.ndim != 4:
            raise SpecError("packed inputs require a validated four-dimensional mask")
        return self._forward_packed(
            input_ids=input_ids,
            attention_mask=attention_mask,
            semantic_mask=semantic_mask,
            segment_ids=packed_segment_ids,
            image_features=image_features,
            counts=counts,
        )


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


def _factory(config: Config, context) -> QwenVisionMiniCPMExtended:
    from transformers import LlamaForCausalLM, Qwen3_5ForConditionalGeneration

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
    model = QwenVisionMiniCPMExtended(
        vision_encoder,
        language_model,
        attention_policy=context.components.get("__attention_policy__"),
    )
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
        module_id=ModuleId.parse("model:validation/qwen35_vision_minicpm5_ext@2"),
        config_type=Config,
        factory=_factory,
        provides=CapabilitySet.of(
            {
                "model.monolithic",
                "model.output.logits",
                "model.parameters",
                "model.attention.semantic",
                "fusion.sequence_length_preserving",
            }
        ),
    )
