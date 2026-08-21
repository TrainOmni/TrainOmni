"""Canonical samples to real Qwen-vision/MiniCPM tensors for all five stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from trainomni.contracts.batch import EncodedSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    vision_checkpoint: str
    language_checkpoint: str
    mode: str
    max_text_tokens: int = 96
    image_min_pixels: int = 65536
    image_max_pixels: int = 262144

    def __post_init__(self) -> None:
        if self.mode not in {"caption", "sft", "dense_kd", "preference"}:
            raise ValueError("unsupported model-io mode")
        if self.max_text_tokens < 16:
            raise ValueError("max_text_tokens must be at least 16")


class ModelIO:
    def __init__(self, config: Config, *, task_root: Path | None) -> None:
        from transformers import AutoProcessor, AutoTokenizer

        self.config = config
        self.task_root = task_root
        processor = AutoProcessor.from_pretrained(
            config.vision_checkpoint, local_files_only=True
        )
        self.processor = processor.image_processor
        self.processor.size = {
            "shortest_edge": config.image_min_pixels,
            "longest_edge": config.image_max_pixels,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.language_checkpoint, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _resolve_image(self, raw: Any):
        from PIL import Image

        path = Path(str(raw))
        if not path.is_absolute():
            if self.task_root is None:
                raise SpecError("relative image path requires a task root")
            path = self.task_root / path
        path = path.resolve()
        if not path.is_file():
            raise SpecError(f"image does not exist: {path}")
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def _images(self, sample) -> list[Any]:
        blocks = list(sample.content)
        for message in sample.messages:
            blocks.extend(message.content)
        images = [self._resolve_image(block.value) for block in blocks if block.kind == "image"]
        if not images:
            raise SpecError("real VLM sample requires at least one image")
        return images

    def _vision_inputs(self, sample) -> dict[str, torch.Tensor]:
        images = self._images(sample)
        values = self.processor(images=images, return_tensors="pt")
        return {
            "pixel_values": values["pixel_values"],
            "image_grid_thw": values["image_grid_thw"],
            "image_counts": torch.tensor([len(images)], dtype=torch.long),
        }

    def _encode_prompt_answer(
        self, prompt: str, answer: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_ids = self.tokenizer(
            prompt.rstrip() + "\n",
            add_special_tokens=True,
            truncation=True,
            max_length=max(8, self.config.max_text_tokens // 2),
        )["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            answer_ids = [*answer_ids, int(eos)]
        available = self.config.max_text_tokens - len(prefix_ids)
        if available < 1:
            raise SpecError("prompt consumes the full text budget")
        answer_ids = answer_ids[:available]
        if not answer_ids:
            raise SpecError("answer produces no supervised tokens")
        ids = torch.tensor([*prefix_ids, *answer_ids], dtype=torch.long)
        mask = torch.zeros(ids.shape, dtype=torch.bool)
        mask[len(prefix_ids) :] = True
        return ids, torch.ones_like(ids), mask

    def _caption_values(self, sample) -> tuple[str, str]:
        if not sample.content:
            raise SpecError("caption/KD sample must use flat content")
        answers = [str(block.value) for block in sample.content if block.kind == "text"]
        if len(answers) != 1:
            raise SpecError("caption/KD sample requires exactly one text answer")
        return str(sample.metadata.get("prompt", "Describe the image.")), answers[0]

    def _sft_values(self, sample) -> tuple[str, str]:
        if not sample.messages or sample.messages[-1].role != "assistant":
            raise SpecError("SFT sample must end in an assistant message")
        answer = "\n".join(
            str(block.value)
            for block in sample.messages[-1].content
            if block.kind == "text"
        )
        if not answer:
            raise SpecError("SFT assistant message has no text")
        prompt_messages = []
        for message in sample.messages[:-1]:
            text = "\n".join(
                str(block.value) for block in message.content if block.kind == "text"
            )
            prompt_messages.append({"role": message.role, "content": text})
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, answer

    def _branch(self, sample, answer: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        prompt_messages = []
        for message in sample.messages:
            text = "\n".join(
                str(block.value) for block in message.content if block.kind == "text"
            )
            prompt_messages.append({"role": message.role, "content": text})
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids, attention_mask, loss_mask = self._encode_prompt_answer(prompt, answer)
        labels = input_ids.clone().masked_fill(~loss_mask, -100)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **self._vision_inputs(sample),
        }
        return inputs, labels

    def encode(self, sample) -> EncodedSample:
        if self.config.mode == "preference":
            chosen = sample.metadata.get("chosen")
            rejected = sample.metadata.get("rejected")
            cache = sample.metadata.get("tensor_cache")
            if not isinstance(chosen, str) or not isinstance(rejected, str):
                raise SpecError("preference sample requires chosen/rejected text")
            if not isinstance(cache, dict):
                raise SpecError("preference sample requires a hash-pinned tensor cache")
            chosen_inputs, chosen_labels = self._branch(sample, chosen)
            rejected_inputs, rejected_labels = self._branch(sample, rejected)
            supervision = {
                "chosen_inputs": chosen_inputs,
                "rejected_inputs": rejected_inputs,
                "chosen_labels": chosen_labels,
                "rejected_labels": rejected_labels,
                "chosen_reference_logps": cache.get("chosen_reference_logps"),
                "rejected_reference_logps": cache.get("rejected_reference_logps"),
            }
            return EncodedSample(sample.sample_id, chosen_inputs, supervision)

        if self.config.mode == "sft":
            prompt, answer = self._sft_values(sample)
        else:
            prompt, answer = self._caption_values(sample)
        input_ids, attention_mask, loss_mask = self._encode_prompt_answer(prompt, answer)
        supervision: dict[str, Any] = {"loss_mask": loss_mask}
        if self.config.mode == "dense_kd":
            cache = sample.metadata.get("tensor_cache")
            if not isinstance(cache, dict) or not isinstance(
                cache.get("teacher_logits"), torch.Tensor
            ):
                raise SpecError("dense KD sample requires cached teacher_logits")
            supervision["teacher_logits"] = cache["teacher_logits"]
        return EncodedSample(
            sample.sample_id,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **self._vision_inputs(sample),
            },
            supervision,
        )


def _factory(config: Config, context) -> ModelIO:
    return ModelIO(config, task_root=context.task_root)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("model_io:validation/qwen35_vision_minicpm5@1"),
        config_type=Config,
        factory=_factory,
        provides=CapabilitySet.of({"data.encoded"}),
        requires=CapabilitySet.of({"data.sample.omni"}),
    )

