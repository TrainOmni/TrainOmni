"""Deterministic in-memory OmniSample source."""

from __future__ import annotations

from collections.abc import Mapping

from trainomni.contracts.sample import ContentBlock, Message, OmniSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import MemorySourceConfig


def _sample_from_mapping(value: Mapping, index: int) -> OmniSample:
    allowed = {"sample_id", "content", "messages", "metadata"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SpecError(f"memory sample[{index}] contains unknown keys: {', '.join(unknown)}")
    has_content = "content" in value
    has_messages = "messages" in value
    if has_content == has_messages:
        raise SpecError(
            f"memory sample[{index}] requires exactly one of content or messages"
        )
    raw_content = value["content"] if has_content else None
    raw_messages = value["messages"] if has_messages else None

    def parse_block(raw_block, *, location: str):
        if not isinstance(raw_block, Mapping):
            raise SpecError(f"{location} must be a mapping")
        block_unknown = sorted(set(raw_block) - {"kind", "value", "metadata"})
        if block_unknown:
            raise SpecError(
                f"{location} contains unknown keys: {', '.join(block_unknown)}"
            )
        try:
            return ContentBlock(
                kind=raw_block.get("kind"),
                value=raw_block.get("value"),
                metadata=raw_block.get("metadata", {}),
            )
        except (TypeError, ValueError) as exc:
            raise SpecError(f"invalid {location}: {exc}") from exc

    blocks = ()
    messages = ()
    if has_content:
        if not isinstance(raw_content, (list, tuple)) or not raw_content:
            raise SpecError(
                f"memory sample[{index}].content must be a non-empty sequence"
            )
        blocks = tuple(
            parse_block(
                raw_block,
                location=f"memory sample[{index}].content[{block_index}]",
            )
            for block_index, raw_block in enumerate(raw_content)
        )
    else:
        if not isinstance(raw_messages, (list, tuple)):
            raise SpecError(f"memory sample[{index}].messages must be a sequence")
        parsed_messages = []
        for message_index, raw_message in enumerate(raw_messages):
            location = f"memory sample[{index}].messages[{message_index}]"
            if not isinstance(raw_message, Mapping):
                raise SpecError(f"{location} must be a mapping")
            message_unknown = sorted(
                set(raw_message) - {"role", "content", "metadata"}
            )
            if message_unknown:
                raise SpecError(
                    f"{location} contains unknown keys: {', '.join(message_unknown)}"
                )
            raw_blocks = raw_message.get("content")
            if not isinstance(raw_blocks, (list, tuple)) or not raw_blocks:
                raise SpecError(f"{location}.content must be a non-empty sequence")
            try:
                parsed_messages.append(
                    Message(
                        role=raw_message.get("role"),
                        content=tuple(
                            parse_block(
                                raw_block,
                                location=f"{location}.content[{block_index}]",
                            )
                            for block_index, raw_block in enumerate(raw_blocks)
                        ),
                        metadata=raw_message.get("metadata", {}),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise SpecError(f"invalid {location}: {exc}") from exc
        messages = tuple(parsed_messages)
    try:
        return OmniSample(
            sample_id=value.get("sample_id"),
            content=blocks,
            metadata=value.get("metadata", {}),
            messages=messages,
        )
    except (TypeError, ValueError) as exc:
        raise SpecError(f"invalid memory sample[{index}]: {exc}") from exc


class MemorySource:
    def __init__(self, config: MemorySourceConfig) -> None:
        self.samples = tuple(
            _sample_from_mapping(sample, index) for index, sample in enumerate(config.samples)
        )
        self.repeat = config.repeat
        self.is_finite = not config.repeat
        self.cursor = 0

    def next_sample(self) -> OmniSample:
        if not self.repeat and self.cursor >= len(self.samples):
            raise StopIteration
        sample = self.samples[self.cursor % len(self.samples)]
        self.cursor += 1
        return sample

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        if set(state) != {"cursor"}:
            raise SpecError("invalid memory source state")
        cursor = state["cursor"]
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise SpecError("memory source cursor must be an integer")
        if cursor < 0:
            raise SpecError("memory source cursor must be non-negative")
        if not self.repeat and cursor > len(self.samples):
            raise SpecError("finite memory source cursor is out of range")
        self.cursor = cursor


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("data_source:trainomni/memory@1"),
        config_type=MemorySourceConfig,
        factory=lambda config, context: MemorySource(config),
        provides=CapabilitySet.of({"data.sample.omni", "data.source.stateful"}),
    )
