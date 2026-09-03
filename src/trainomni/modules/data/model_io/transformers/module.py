"""Basic canonical-sample to Transformers processor adapter."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from trainomni.contracts._mapping import FrozenDict
from trainomni.contracts.batch import EncodedSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId
from trainomni.modules.data._fields import flatten_fields, unflatten_fields
from trainomni.modules.data._tensors import binary_mask

from .config import TransformersModelIOConfig


class TransformersModelIO:
    def __init__(self, processor, config: TransformersModelIOConfig) -> None:
        self.processor = processor
        self.config = config

    @staticmethod
    def _chat_messages(sample):
        messages = []
        for message in sample.messages:
            content = []
            for block in message.content:
                if block.kind == "text":
                    if not isinstance(block.value, str):
                        raise SpecError("chat text blocks must contain strings")
                    content.append({"type": "text", "text": block.value})
                else:
                    content.append({"type": block.kind, block.kind: block.value})
            messages.append({"role": message.role, "content": content})
        return messages

    def _encode_conversation(self, sample):
        if self.config.conversation_mode == "disabled":
            raise SpecError("conversation sample rejected by disabled conversation_mode")
        apply_template = getattr(self.processor, "apply_chat_template", None)
        if not callable(apply_template):
            raise SpecError("processor does not implement apply_chat_template")
        try:
            encoded = apply_template(
                self._chat_messages(sample),
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=self.config.add_generation_prompt,
                return_assistant_tokens_mask=True,
            )
        except Exception as exc:
            raise SpecError(f"processor chat-template encoding failed: {exc}") from exc
        return self._normalize_encoded(sample, encoded, conversation=True)

    def _normalize_encoded(self, sample, encoded, *, conversation: bool):
        return self.normalize_encoded(sample, encoded, conversation=conversation)

    def normalize_encoded(self, sample, encoded, *, conversation: bool):
        """Normalize a custom processor result using the same builtin field contract."""
        if not isinstance(encoded, Mapping):
            raise SpecError("processor output must be a mapping")
        encoded = dict(encoded)
        assistant_mask = None
        for field in self.config.assistant_mask_fields:
            if field in encoded:
                if assistant_mask is not None:
                    raise SpecError("processor returned multiple assistant mask fields")
                assistant_mask = encoded.pop(field)
        model_inputs = {}
        for key, value in encoded.items():
            if key in self.config.batch_axis_fields:
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim < 2
                    or value.shape[0] != 1
                ):
                    raise SpecError(
                        f"processor field {key!r} requires an explicit singleton "
                        "batch axis with shape [1, ...]"
                    )
                value = value.squeeze(0)
            model_inputs[str(key)] = value
        supervision = {}
        metadata_key = self.config.supervision_metadata_key
        if metadata_key is not None:
            cached = sample.metadata.get(metadata_key)
            if not isinstance(cached, Mapping) or not cached:
                raise SpecError(
                    f"sample metadata {metadata_key!r} must contain cached tensors"
                )
            for name, value in cached.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(value, torch.Tensor)
                ):
                    raise SpecError(
                        f"sample metadata {metadata_key!r} must map names to tensors"
                    )
                supervision[name] = value
        if assistant_mask is not None:
            if not isinstance(assistant_mask, torch.Tensor):
                raise SpecError("assistant mask must be a tensor")
            if assistant_mask.ndim == 2 and assistant_mask.shape[0] == 1:
                assistant_mask = assistant_mask.squeeze(0)
            input_ids = model_inputs.get("input_ids")
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.ndim != 1
            ):
                raise SpecError("assistant mask must align exactly with one-dimensional input_ids")
            assistant_mask = binary_mask(
                assistant_mask,
                field="assistant mask",
                shape=input_ids.shape,
            )
            if self.config.loss_mask_field in supervision:
                raise SpecError("assistant loss mask collides with cached supervision")
            supervision[self.config.loss_mask_field] = assistant_mask
        elif conversation and self.config.require_assistant_mask:
            raise SpecError(
                "processor chat template returned no assistant token mask; "
                "refusing to train on prompt tokens"
            )
        if self.config.modal_token_id is not None:
            input_ids = model_inputs.get("input_ids")
            if (
                not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1
                or input_ids.is_floating_point() or input_ids.is_complex()
                or input_ids.dtype is torch.bool
            ):
                raise SpecError("modal token extraction requires one-dimensional integer input_ids")
            positions = (input_ids == self.config.modal_token_id).nonzero().flatten()
            position_field = self.config.modal_positions_field
            if position_field in model_inputs:
                raise SpecError(f"generated modal positions would overwrite {position_field!r}")
            valid = input_ids != self.config.modal_token_id
            existing_mask = supervision.get(self.config.loss_mask_field)
            supervision[self.config.loss_mask_field] = (
                valid if existing_mask is None else existing_mask & valid
            )
        else:
            positions = None
        routed = {}
        for field, value in flatten_fields(model_inputs).items():
            if field in self.config.discard_fields:
                continue
            if field not in self.config.field_routes and self.config.unmapped_fields == "error":
                raise SpecError(
                    f"processor field {field!r} has no field_routes or discard_fields rule"
                )
            target = self.config.field_routes.get(field, field)
            if target in routed:
                raise SpecError(f"processor routes collide at {target!r}")
            routed[target] = value
        if positions is not None:
            if self.config.modal_positions_field in routed:
                raise SpecError("processor route collides with generated modal positions")
            routed[self.config.modal_positions_field] = positions
        return EncodedSample(
            sample.sample_id,
            unflatten_fields(routed),
            FrozenDict(supervision),
        )

    def encode(self, sample):
        if sample.messages:
            return self._encode_conversation(sample)
        if self.config.conversation_mode == "required":
            raise SpecError("conversation_mode=required rejects flat content samples")
        texts = [block.value for block in sample.content if block.kind == "text"]
        images = [block.value for block in sample.content if block.kind == "image"]
        videos = [block.value for block in sample.content if block.kind == "video"]
        if any(not isinstance(text, str) for text in texts):
            raise SpecError("Transformers ModelIO requires string text blocks")
        arguments = {"text": self.config.text_separator.join(texts), "return_tensors": "pt"}
        if images:
            arguments["images"] = images
        if videos:
            arguments["videos"] = videos
        encoded = self.processor(**arguments)
        return self._normalize_encoded(sample, encoded, conversation=False)


def _factory(config: TransformersModelIOConfig, context):
    del context
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise SpecError("Transformers ModelIO requires transformers") from exc
    processor = AutoProcessor.from_pretrained(
        config.processor_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    return TransformersModelIO(processor, config)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("model_io:trainomni/transformers@1"),
        config_type=TransformersModelIOConfig,
        factory=_factory,
        provides=CapabilitySet.of({"data.encoded"}),
        requires=CapabilitySet.of({"data.sample.omni"}),
        configured_provides=lambda config: CapabilitySet.of(
            {"batch.modal_positions"} if config.modal_token_id is not None else set()
        ),
    )
