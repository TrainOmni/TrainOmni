"""Public tiny LLaVA integration used as the framework's real smoke plugin.

This file is intentionally external to ``trainomni``: adding the model requires
registration through ``--plugin`` and no core edit.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trainomni.contracts import BatchPlan, CostVector
from trainomni.models import (
    ComponentCatalog,
    ComponentRule,
    EncodedSample,
    ModelBatch,
    ModelBundle,
    ModelCapabilities,
    ModelPluginManifest,
    SourceSpan,
)

MODEL_ID = "Xenova/tiny-random-LlavaForConditionalGeneration"


class TinyLlavaPlugin:
    manifest = ModelPluginManifest(
        plugin_id="tiny-llava",
        plugin_version="1.0.0",
        capabilities=ModelCapabilities(
            modalities=frozenset({"text", "image"}),
            content_blocks=frozenset({"text", "media"}),
            objectives=frozenset({"sft", "cpt"}),
            max_media_per_sample=1,
            supports_packing=False,
            supports_padding_free=False,
            supports_generation=True,
            attention_backends=frozenset({"eager", "sdpa"}),
            parallelism=frozenset({"single", "ddp", "fsdp2"}),
            engine_backends=frozenset({"torch"}),
            export_formats=frozenset({"hf"}),
        ),
        component_ids=("vision_encoder", "connector", "language_model"),
        model_patterns=(MODEL_ID,),
        dependency_constraints=("torch>=2.4", "transformers>=4.49"),
    )

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._checkpoint = MODEL_ID

    def capabilities(self) -> ModelCapabilities:
        return self.manifest.capabilities

    def build(self, config: Mapping[str, Any]) -> ModelBundle:
        torch, transformers = _dependencies()
        checkpoint = str(config.get("checkpoint", MODEL_ID))
        self._checkpoint = checkpoint
        self._processor = transformers.AutoProcessor.from_pretrained(checkpoint)
        model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_class is None:
            model_class = transformers.LlavaForConditionalGeneration
        kwargs: dict[str, Any] = {}
        dtype = config.get("load_dtype")
        if dtype:
            dtype_name = {
                "fp32": "float32",
                "fp16": "float16",
                "bf16": "bfloat16",
            }.get(str(dtype), str(dtype))
            kwargs["dtype"] = getattr(torch, dtype_name)
        model = model_class.from_pretrained(checkpoint, **kwargs)
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and (
            tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0
        ):
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if getattr(model.config, "pad_token_id", None) is None or model.config.pad_token_id < 0:
            model.config.pad_token_id = tokenizer.pad_token_id if tokenizer else 1
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and (
            generation_config.pad_token_id is None
            or generation_config.pad_token_id < 0
        ):
            generation_config.pad_token_id = model.config.pad_token_id
        return ModelBundle(
            model=model,
            processor=self._processor,
            tokenizer=tokenizer,
            metadata={"checkpoint": checkpoint},
        )

    def component_catalog(self, bundle: ModelBundle) -> ComponentCatalog:
        return ComponentCatalog(
            rules=(
                ComponentRule("vision_encoder", ("model.vision_tower.",)),
                ComponentRule("connector", ("model.multi_modal_projector.",)),
                ComponentRule(
                    "language_model", ("model.language_model.", "lm_head.")
                ),
            )
        )

    def validate_sample(self, sample: Any, objective: str) -> tuple[Any, ...]:
        if objective not in {"sft", "cpt"}:
            from trainomni.models import CapabilityIssue

            return (
                CapabilityIssue(
                    "tiny_llava.objective", f"unsupported objective {objective!r}"
                ),
            )
        return ()

    def encode(self, sample: Any, context: Mapping[str, Any]) -> EncodedSample:
        torch, _ = _dependencies()
        processor = self._processor or self._load_processor()
        assets = {asset.id: asset for asset in sample.assets}
        source_uri = context.get("source_trace", {}).get("source_uri")
        base_dir = Path(source_uri).parent if source_uri else Path.cwd()
        messages = []
        images = []
        assistant_texts = []
        pixels = 0
        for message in sample.messages:
            content = []
            for block in message.content:
                if block.type == "text":
                    content.append({"type": "text", "text": block.text})
                    if message.role == "assistant" and block.loss_weight != 0:
                        assistant_texts.append(block.text)
                elif block.type == "media":
                    asset = assets[block.asset_id]
                    path = Path(asset.uri)
                    if not path.is_absolute():
                        path = (base_dir / path).resolve()
                    content.append({"type": "image", "path": str(path)})
                    images.append(_load_image(path))
                    if asset.width and asset.height:
                        pixels += asset.width * asset.height
                else:
                    raise ValueError(
                        f"tiny LLaVA does not encode block type {block.type!r}"
                    )
            messages.append({"role": message.role, "content": content})
        prompt = _format_chat(processor, messages)
        encoded = processor(
            text=prompt,
            images=images or None,
            return_tensors="pt",
            padding=False,
        )
        inputs = {
            key: value.squeeze(0) if hasattr(value, "shape") and value.shape[0] == 1 else value
            for key, value in encoded.items()
        }
        input_ids = inputs["input_ids"]
        labels = torch.full_like(input_ids, -100)
        spans = []
        search_from = 0
        tokenizer = getattr(processor, "tokenizer", processor)
        for index, text in enumerate(assistant_texts):
            answer_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            start = _find_subsequence(input_ids.tolist(), answer_ids, search_from)
            if start < 0:
                raise ValueError(
                    "assistant answer tokens were not found in the formatted prompt; "
                    "the checkpoint chat template is incompatible"
                )
            end = start + len(answer_ids)
            labels[start:end] = input_ids[start:end]
            spans.append(
                SourceSpan(
                    field="labels",
                    start=start,
                    end=end,
                    source_path=f"assistant[{index}]",
                    loss_weight=1.0,
                )
            )
            search_from = end
        if sample.objective == "cpt":
            labels = input_ids.clone()
            spans = [
                SourceSpan("labels", 0, int(input_ids.numel()), "messages", 1.0)
            ]
        inputs["labels"] = labels
        vision_tokens = 0
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None and hasattr(pixel_values, "shape"):
            vision_tokens = int(pixel_values.shape[-1] * pixel_values.shape[-2] // 4)
        return EncodedSample(
            sample_id=sample.id,
            model_inputs=inputs,
            cost=CostVector(
                text_tokens=int(input_ids.numel()),
                vision_tokens=vision_tokens,
                pixels=pixels,
            ),
            source_spans=tuple(spans),
            trace={"checkpoint": self._checkpoint, "chat_template": True},
        )

    def collate(self, samples: list[EncodedSample], plan: BatchPlan) -> ModelBatch:
        torch, _ = _dependencies()
        processor = self._processor or self._load_processor()
        tokenizer = getattr(processor, "tokenizer", processor)
        pad_id = tokenizer.pad_token_id
        if pad_id is None or pad_id < 0:
            pad_id = tokenizer.eos_token_id
        inputs: dict[str, Any] = {}
        keys = set.intersection(*(set(sample.model_inputs) for sample in samples))
        for key in sorted(keys):
            values = [sample.model_inputs[key] for sample in samples]
            if key in {"input_ids", "attention_mask", "labels"}:
                padding = -100 if key == "labels" else (pad_id if key == "input_ids" else 0)
                inputs[key] = torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=padding
                )
            elif all(hasattr(value, "shape") for value in values):
                inputs[key] = torch.stack(values)
            else:
                inputs[key] = values
        return ModelBatch(
            sample_ids=tuple(sample.sample_id for sample in samples),
            model_inputs=inputs,
            plan=plan,
            trace={"collator": "tiny-llava"},
        )

    def export(
        self, bundle: ModelBundle, checkpoint: Any, target: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        output_dir = Path(target["output_dir"])
        state_dict = target.get("state_dict")
        kwargs = {"safe_serialization": True}
        if state_dict is not None:
            kwargs["state_dict"] = state_dict
        bundle.model.save_pretrained(output_dir, **kwargs)
        if bundle.processor is not None:
            bundle.processor.save_pretrained(output_dir)
        return {"format": "hf", "path": str(output_dir)}

    def _load_processor(self) -> Any:
        _, transformers = _dependencies()
        self._processor = transformers.AutoProcessor.from_pretrained(self._checkpoint)
        return self._processor


def _dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "tiny LLaVA plugin requires trainomni-framework[torch] and Pillow"
        ) from exc
    return torch, transformers


def _load_image(path: Path) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("tiny LLaVA image loading requires Pillow") from exc
    with Image.open(path) as image:
        return image.convert("RGB")


def _format_chat(processor: Any, messages: list[dict[str, Any]]) -> str:
    apply_template = getattr(processor, "apply_chat_template", None)
    if callable(apply_template) and getattr(processor, "chat_template", None):
        return apply_template(messages, tokenize=False, add_generation_prompt=False)
    pieces = []
    for message in messages:
        text = "".join(
            "<image>\n" if item["type"] == "image" else item["text"]
            for item in message["content"]
        )
        role = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}.get(
            message["role"], message["role"].upper()
        )
        pieces.append(f"{role}: {text}")
    return " ".join(pieces)


def _find_subsequence(values: list[int], needle: list[int], start: int) -> int:
    if not needle:
        return -1
    for index in range(start, len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


PLUGIN = TinyLlavaPlugin()
