"""Dependency-free external plugin fixture.

This module deliberately lives outside ``src/trainomni``. Loading it by an
explicit file path proves that model registration requires no core edit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trainomni.contracts import BatchPlan, CostVector
from trainomni.models import (
    ComponentCatalog,
    ComponentRule,
    EncodedSample,
    ModelBatch,
    ModelCapabilities,
    ModelPluginManifest,
    SourceSpan,
)


@dataclass(frozen=True, slots=True)
class ToyBundle:
    parameter_names: tuple[str, ...] = (
        "vision.block.weight",
        "connector.proj.weight",
        "language.layer.weight",
    )


class ToyVLMPlugin:
    manifest = ModelPluginManifest(
        plugin_id="toy-vlm",
        plugin_version="1.0.0",
        capabilities=ModelCapabilities(
            modalities=frozenset({"text", "image"}),
            content_blocks=frozenset({"text", "media", "bbox"}),
            objectives=frozenset({"cpt", "sft"}),
            max_media_per_sample=4,
            supports_packing=True,
            supports_padding_free=False,
            supports_generation=True,
            attention_backends=frozenset({"sdpa"}),
            parallelism=frozenset({"single", "ddp", "fsdp2"}),
            engine_backends=frozenset({"torch"}),
            export_formats=frozenset({"hf"}),
        ),
        component_ids=("vision_encoder", "connector", "language_model"),
        model_patterns=("toy/*",),
    )

    def capabilities(self) -> ModelCapabilities:
        return self.manifest.capabilities

    def build(self, config: Mapping[str, Any]) -> ToyBundle:
        return ToyBundle()

    def component_catalog(self, bundle: ToyBundle) -> ComponentCatalog:
        return ComponentCatalog(
            rules=(
                ComponentRule("vision_encoder", ("vision.",)),
                ComponentRule("connector", ("connector.",)),
                ComponentRule("language_model", ("language.",)),
            )
        )

    def validate_sample(self, sample: Any, objective: str) -> tuple[Any, ...]:
        return ()

    def encode(self, sample: Any, context: Mapping[str, Any]) -> EncodedSample:
        tokens: list[int] = []
        spans = []
        pixels = 0
        for message_index, message in enumerate(sample.messages):
            for block_index, block in enumerate(message.content):
                if block.type == "text":
                    start = len(tokens)
                    block_tokens = [len(word) for word in block.text.split()]
                    tokens.extend(block_tokens)
                    spans.append(
                        SourceSpan(
                            field="input_ids",
                            start=start,
                            end=len(tokens),
                            source_path=(
                                f"messages[{message_index}].content[{block_index}]"
                            ),
                            loss_weight=block.loss_weight,
                        )
                    )
        for asset in sample.assets:
            if asset.width is not None and asset.height is not None:
                pixels += asset.width * asset.height
        return EncodedSample(
            sample_id=sample.id,
            model_inputs={
                "input_ids": tokens or [0],
                "labels": tokens or [0],
                "media_count": len(sample.assets),
            },
            cost=CostVector(
                text_tokens=max(1, len(tokens)),
                vision_tokens=len(sample.assets) * 16,
                pixels=pixels,
            ),
            source_spans=tuple(spans),
            trace={"toy_encoder": "word-length"},
        )

    def collate(
        self, samples: list[EncodedSample], plan: BatchPlan
    ) -> ModelBatch:
        return ModelBatch(
            sample_ids=tuple(sample.sample_id for sample in samples),
            model_inputs={
                "input_ids": [list(sample.model_inputs["input_ids"]) for sample in samples],
                "labels": [list(sample.model_inputs["labels"]) for sample in samples],
                "media_count": [sample.model_inputs["media_count"] for sample in samples],
            },
            plan=plan,
            trace={"collator": "toy-padding-free-list"},
        )

    def export(
        self, bundle: ToyBundle, checkpoint: Any, target: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {"format": target.get("format", "hf")}


PLUGIN = ToyVLMPlugin()
