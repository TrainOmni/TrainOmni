"""Resumable line-offset JSONL source with immutable file identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from trainomni.contracts.sample import ContentBlock, Message, OmniSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import CheckpointError, SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import JsonlSourceConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_block(block, *, location: str) -> ContentBlock:
    if not isinstance(block, Mapping):
        raise SpecError(f"{location} must be a mapping")
    block_unknown = sorted(set(block) - {"kind", "value", "metadata"})
    if block_unknown:
        raise SpecError(
            f"{location} contains unknown keys: {', '.join(block_unknown)}"
        )
    try:
        return ContentBlock(
            kind=block.get("kind"),
            value=block.get("value"),
            metadata=block.get("metadata", {}),
        )
    except (TypeError, ValueError) as exc:
        raise SpecError(f"invalid {location}: {exc}") from exc


def _parse_sample(value, *, line_number: int) -> OmniSample:
    if not isinstance(value, Mapping):
        raise SpecError(f"JSONL line {line_number} must be a mapping")
    unknown = sorted(set(value) - {"sample_id", "content", "messages", "metadata"})
    if unknown:
        raise SpecError(
            f"JSONL line {line_number} contains unknown keys: {', '.join(unknown)}"
        )
    raw_content = value.get("content")
    raw_messages = value.get("messages")
    if bool(raw_content) == bool(raw_messages):
        raise SpecError(
            f"JSONL line {line_number} requires exactly one of content or messages"
        )
    blocks = ()
    messages = ()
    if raw_content:
        if not isinstance(raw_content, list):
            raise SpecError(f"JSONL line {line_number}.content must be a list")
        blocks = tuple(
            _parse_block(
                block,
                location=f"JSONL line {line_number}.content[{index}]",
            )
            for index, block in enumerate(raw_content)
        )
    else:
        if not isinstance(raw_messages, list):
            raise SpecError(f"JSONL line {line_number}.messages must be a list")
        parsed_messages = []
        for index, message in enumerate(raw_messages):
            location = f"JSONL line {line_number}.messages[{index}]"
            if not isinstance(message, Mapping):
                raise SpecError(f"{location} must be a mapping")
            message_unknown = sorted(set(message) - {"role", "content", "metadata"})
            if message_unknown:
                raise SpecError(
                    f"{location} contains unknown keys: {', '.join(message_unknown)}"
                )
            raw_blocks = message.get("content")
            if not isinstance(raw_blocks, list) or not raw_blocks:
                raise SpecError(f"{location}.content must be a non-empty list")
            try:
                parsed_messages.append(
                    Message(
                        role=message.get("role"),
                        content=tuple(
                            _parse_block(
                                block,
                                location=f"{location}.content[{block_index}]",
                            )
                            for block_index, block in enumerate(raw_blocks)
                        ),
                        metadata=message.get("metadata", {}),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise SpecError(f"invalid {location}: {exc}") from exc
        messages = tuple(parsed_messages)
    try:
        return OmniSample(
            sample_id=str(value.get("sample_id", "")),
            content=blocks,
            metadata=value.get("metadata", {}),
            messages=messages,
        )
    except (TypeError, ValueError) as exc:
        raise SpecError(f"invalid JSONL line {line_number}: {exc}") from exc


class JsonlSource:
    def __init__(self, path: Path, config: JsonlSourceConfig) -> None:
        if not path.is_file():
            raise SpecError(f"JSONL source does not exist: {path}")
        actual_digest = _sha256(path)
        if actual_digest != config.sha256:
            raise SpecError(
                f"JSONL source digest mismatch: expected {config.sha256}, "
                f"got {actual_digest}"
            )
        self.path = path
        self.file_sha256 = actual_digest
        self.repeat = config.repeat
        self.is_finite = not config.repeat
        self.offset = 0
        self.line_number = 0
        self.epoch = 0

    def next_sample(self) -> OmniSample:
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            raw_line = stream.readline()
            if not raw_line:
                if not self.repeat:
                    raise StopIteration
                if self.offset == 0:
                    raise SpecError("JSONL source is empty")
                self.offset = 0
                self.line_number = 0
                self.epoch += 1
                stream.seek(0)
                raw_line = stream.readline()
            self.offset = stream.tell()
            self.line_number += 1
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpecError(f"invalid JSONL line {self.line_number}: {exc}") from exc
        return _parse_sample(value, line_number=self.line_number)

    def state_dict(self):
        return {
            "file_sha256": self.file_sha256,
            "offset": self.offset,
            "line_number": self.line_number,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        expected = {"file_sha256", "offset", "line_number", "epoch"}
        if set(state) != expected:
            raise CheckpointError("invalid JSONL source state keys")
        if state["file_sha256"] != self.file_sha256:
            raise CheckpointError("JSONL source identity changed")
        offset = int(state["offset"])
        line_number = int(state["line_number"])
        epoch = int(state["epoch"])
        if min(offset, line_number, epoch) < 0 or offset > self.path.stat().st_size:
            raise CheckpointError("invalid JSONL source cursor")
        self.offset = offset
        self.line_number = line_number
        self.epoch = epoch


def _factory(config: JsonlSourceConfig, context):
    raw_path = Path(config.path)
    if not raw_path.is_absolute():
        if context.task_root is None:
            raise SpecError("relative JSONL paths require a task root")
        raw_path = context.task_root / raw_path
    return JsonlSource(raw_path.resolve(), config)


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("data_source:trainomni/jsonl@1"),
        config_type=JsonlSourceConfig,
        factory=_factory,
        provides=CapabilitySet.of({"data.sample.omni", "data.source.stateful"}),
    )
