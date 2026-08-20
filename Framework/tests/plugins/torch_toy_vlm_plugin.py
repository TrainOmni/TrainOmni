"""Small real PyTorch VLM-shaped plugin for execution tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trainomni.contracts import BatchPlan, CostVector
from trainomni.models import (
    ActivationCheckpointingReceipt,
    ComponentCatalog,
    ComponentRule,
    EncodedSample,
    ModelBatch,
    ModelBundle,
    ModelCapabilities,
    ModelPluginManifest,
    SourceSpan,
)


class TorchToyPlugin:
    manifest = ModelPluginManifest(
        plugin_id="torch-toy-vlm",
        plugin_version="1.0.0",
        capabilities=ModelCapabilities(
            modalities=frozenset({"text"}),
            content_blocks=frozenset({"text"}),
            objectives=frozenset({"sft", "cpt"}),
            max_media_per_sample=0,
            supports_packing=False,
            supports_generation=True,
            attention_backends=frozenset({"eager"}),
            parallelism=frozenset({"single", "ddp", "fsdp2"}),
            engine_backends=frozenset({"torch"}),
            export_formats=frozenset({"torch"}),
        ),
        component_ids=("vision_encoder", "connector", "language_model"),
        model_patterns=("torch-toy/*",),
        dependency_constraints=("torch>=2.4",),
    )

    def capabilities(self):
        return self.manifest.capabilities

    def build(self, config: Mapping[str, Any]) -> ModelBundle:
        import torch

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision = torch.nn.Linear(1, 8)
                self.connector = torch.nn.Linear(8, 8)
                self.language = torch.nn.Embedding(64, 8)
                self.lm_head = torch.nn.Linear(8, 64)
                self.activation_checkpointing = {}

            def forward(self, input_ids, labels, **kwargs):
                from torch.utils.checkpoint import checkpoint

                def apply(component_id, module, *args):
                    if component_id not in self.activation_checkpointing:
                        return module(*args)
                    return checkpoint(
                        module,
                        *args,
                        use_reentrant=self.activation_checkpointing[component_id],
                    )

                hidden = apply("language_model", self.language, input_ids)
                # Preserve a VLM-shaped component in the graph even for text-only tests.
                vision = apply(
                    "vision_encoder",
                    self.vision,
                    torch.ones((*hidden.shape[:-1], 1), device=hidden.device),
                )
                connected = apply("connector", self.connector, hidden + vision)
                logits = self.lm_head(connected)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                return {"loss": loss, "logits": logits}

        return ModelBundle(ToyModel(), metadata={"kind": "torch-toy"})

    def configure_activation_checkpointing(self, bundle, requests):
        known = {"vision_encoder", "connector", "language_model"}
        unknown = set(requests) - known
        if unknown:
            raise ValueError(f"unknown checkpoint components: {sorted(unknown)}")
        receipts = {}
        for component_id, request in requests.items():
            bundle.model.activation_checkpointing[component_id] = (
                request.use_reentrant
            )
            receipts[component_id] = ActivationCheckpointingReceipt(
                component_id=component_id,
                implementation="torch.utils.checkpoint.checkpoint",
                use_reentrant=request.use_reentrant,
                metadata={"toy_component": component_id},
            )
        return receipts

    def component_catalog(self, bundle):
        return ComponentCatalog(
            (
                ComponentRule("vision_encoder", ("vision.",)),
                ComponentRule("connector", ("connector.",)),
                ComponentRule("language_model", ("language.", "lm_head.")),
            )
        )

    def validate_sample(self, sample, objective):
        return ()

    def encode(self, sample, context):
        tokens = []
        assistant_start = 0
        for message in sample.messages:
            message_tokens = [min(63, max(1, len(word))) for block in message.content if block.type == "text" for word in block.text.split()]
            if message.role == "assistant" and assistant_start == 0:
                assistant_start = len(tokens)
            tokens.extend(message_tokens)
        tokens = tokens or [1]
        labels = [-100] * assistant_start + tokens[assistant_start:]
        if sample.objective == "cpt":
            labels = list(tokens)
            assistant_start = 0
        return EncodedSample(
            sample_id=sample.id,
            model_inputs={"input_ids": tokens, "labels": labels},
            cost=CostVector(text_tokens=len(tokens)),
            source_spans=(
                SourceSpan("labels", assistant_start, len(tokens), "messages", 1.0),
            ),
        )

    def collate(self, samples, plan: BatchPlan):
        import torch

        ids = [torch.tensor(sample.model_inputs["input_ids"], dtype=torch.long) for sample in samples]
        labels = [torch.tensor(sample.model_inputs["labels"], dtype=torch.long) for sample in samples]
        return ModelBatch(
            sample_ids=tuple(sample.sample_id for sample in samples),
            model_inputs={
                "input_ids": torch.nn.utils.rnn.pad_sequence(ids, batch_first=True),
                "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
            },
            plan=plan,
        )

    def export(self, bundle, checkpoint, target):
        import torch

        output = Path(target["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        torch.save(bundle.model.state_dict(), output / "model.pt")
        return {"path": str(output / "model.pt"), "format": "torch"}


PLUGIN = TorchToyPlugin()
