"""Model-specific tokenization; builtin routing/positions/packing are reused."""

import hashlib
from pathlib import Path

import torch

from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.data.model_io.transformers.config import TransformersModelIOConfig
from trainomni.modules.data.model_io.transformers.module import TransformersModelIO

from .config import IOConfig


class ModelIO:
    def __init__(self, config):
        from transformers import AutoImageProcessor, AutoTokenizer

        self.config = config
        for directory, hashes in (
            (config.vision_model_path, config.vision_assets_sha256),
            (config.language_model_path, config.language_assets_sha256),
        ):
            if not hashes:
                raise SpecError("processor/tokenizer asset hashes must not be empty")
            for name, expected in hashes.items():
                if Path(name).name != name:
                    raise SpecError("processor/tokenizer asset filename must be relative and flat")
                with (Path(directory) / name).open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise SpecError(f"processor/tokenizer asset changed: {name}")
        self.image_processor = AutoImageProcessor.from_pretrained(config.vision_model_path,
                                                                  local_files_only=True)
        self.image_processor.size = {"shortest_edge": config.min_pixels, "longest_edge": config.max_pixels}
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model_path, local_files_only=True)
        self.processor = self.image_processor
        self.modal_id = self.tokenizer.unk_token_id
        if self.modal_id is None or self.tokenizer.encode(self.tokenizer.unk_token,
                                                         add_special_tokens=False) != [self.modal_id]:
            raise SpecError("this example requires a one-token UNK placeholder")
        self.normalizer = TransformersModelIO(None, TransformersModelIOConfig(
            processor_name_or_path=config.vision_model_path,
            modal_token_id=self.modal_id, unmapped_fields="error",
            supervision_metadata_key=config.supervision_metadata_key,
            field_routes={
                "input_ids": "input_ids", "attention_mask": "attention_mask",
                "pixel_values": "vision.hidden_states", "image_grid_thw": "vision.grid_thw",
                "image_counts": "vision.image_counts",
            },
            discard_fields=("mm_token_type_ids",),
        ))

    def encode(self, sample):
        if not sample.messages or sample.messages[-1].role != "assistant":
            raise SpecError("example requires conversations ending in a supervised assistant message")
        images = [b.value for m in sample.messages for b in m.content if b.kind == "image"]
        if not images:
            raise SpecError("this vision validation task requires at least one image")
        visual = self.image_processor(images=images, return_tensors="pt")
        grid = visual["image_grid_thw"]
        merge = self.image_processor.merge_size
        counts = (grid.prod(-1) // (merge * merge)).tolist()
        messages, cursor = [], 0
        for message in sample.messages:
            content = []
            for block in message.content:
                if block.kind == "image":
                    if message.role == "assistant":
                        raise SpecError("assistant images are not supervised text in this template")
                    content.append(self.tokenizer.unk_token * counts[cursor])
                    cursor += 1
                elif block.kind == "text":
                    if self.modal_id in self.tokenizer.encode(block.value, add_special_tokens=False):
                        raise SpecError("text contains the reserved visual UNK placeholder")
                    content.append(block.value)
                else:
                    raise SpecError("this example supports image/text messages only")
            messages.append({"role": message.role, "content": "\n".join(content)})
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if len(full) > self.config.max_tokens:
            raise SpecError(f"encoded sequence {len(full)} exceeds max_tokens={self.config.max_tokens}")
        mask = torch.zeros(len(full), dtype=torch.bool)
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prefix_text = self.tokenizer.apply_chat_template(
                messages[:index], tokenize=False, add_generation_prompt=True
            )
            end_text = self.tokenizer.apply_chat_template(
                messages[:index + 1], tokenize=False, add_generation_prompt=False
            )
            prefix = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            end = self.tokenizer(end_text, add_special_tokens=False)["input_ids"]
            if full[:len(prefix)] != prefix or full[:len(end)] != end or len(prefix) >= len(end):
                raise SpecError("chat template assistant span is not token-prefix aligned")
            mask[len(prefix):len(end)] = True
        ids = torch.tensor([full], dtype=torch.long)
        if int((ids == self.modal_id).sum()) != sum(counts) or not mask.any():
            raise SpecError("visual placeholder count or assistant loss span is invalid")
        return self.normalizer.normalize_encoded(sample, {
            "input_ids": ids, "attention_mask": torch.ones_like(ids),
            "assistant_masks": mask[None], "pixel_values": visual["pixel_values"],
            "image_grid_thw": grid, "image_counts": torch.tensor([len(images)]),
            "mm_token_type_ids": (ids == self.modal_id).long(),
        }, conversation=True)


def descriptor():
    return ModuleDescriptor(
        module_id=ModuleId.parse("model_io:example/qwen35_minicpm5@1"),
        config_type=IOConfig, factory=lambda config, context: ModelIO(config),
        provides=CapabilitySet.of({"data.encoded", "batch.modal_positions"}),
        requires=CapabilitySet.of({"data.sample.omni"}),
    )
