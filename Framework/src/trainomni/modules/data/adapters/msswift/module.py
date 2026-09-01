"""Normalize common ms-swift/Hugging Face VLM rows into OmniSample."""

from __future__ import annotations

import io
import re
from collections.abc import Mapping, Sequence
from typing import Any

from trainomni.contracts.data import DataRecord
from trainomni.contracts.sample import ContentBlock, Message, OmniSample
from trainomni.core.capability import CapabilitySet
from trainomni.core.errors import CheckpointError, SpecError
from trainomni.core.module import ModuleDescriptor, ModuleId

from .config import MSSwiftAdapterConfig

_PLACEHOLDER = re.compile(r"<(image|video|audio)>")
_ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class MSSwiftAdapter:
    def __init__(self, config: MSSwiftAdapterConfig) -> None:
        self.config = config

    def _image_value(self, value: Any, *, location: str) -> Any:
        if isinstance(value, Mapping):
            raw_bytes = value.get("bytes")
            path = value.get("path")
            if raw_bytes is not None:
                value = raw_bytes
            elif path is not None:
                value = path
            else:
                raise SpecError(
                    f"{location} image mapping requires 'bytes' or 'path'"
                )
        if isinstance(value, (bytes, bytearray, memoryview)):
            payload = bytes(value)
            if not self.config.decode_image_bytes:
                return payload
            try:
                from PIL import Image

                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    return image.convert("RGB").copy()
            except Exception as exc:
                raise SpecError(f"cannot decode {location} image bytes: {exc}") from exc
        return value

    def _media_value(self, kind: str, value: Any, *, location: str) -> Any:
        if kind == "image":
            return self._image_value(value, location=location)
        if isinstance(value, Mapping):
            raw_bytes = value.get("bytes")
            path = value.get("path")
            if raw_bytes is not None:
                return bytes(raw_bytes)
            if path is not None:
                return path
            raise SpecError(
                f"{location} {kind} mapping requires 'bytes' or 'path'"
            )
        return value

    def _media_queues(self, fields: Mapping[str, Any]) -> dict[str, list[Any]]:
        queues = {
            "image": _sequence(fields.get(self.config.images_column)),
            "video": _sequence(fields.get(self.config.videos_column)),
            "audio": _sequence(fields.get(self.config.audios_column)),
        }
        for kind, column in (
            ("image", self.config.images_column),
            ("video", self.config.videos_column),
            ("audio", self.config.audios_column),
        ):
            queues[kind] = [
                self._media_value(
                    kind,
                    value,
                    location=f"{column}[{index}]",
                )
                for index, value in enumerate(queues[kind])
            ]
        return queues

    @staticmethod
    def _take_media(
        kind: str,
        queues: dict[str, list[Any]],
        consumed: dict[str, int],
        *,
        location: str,
    ) -> ContentBlock:
        if not queues[kind]:
            raise SpecError(f"{location} contains <{kind}> without matching media")
        consumed[kind] += 1
        return ContentBlock(kind, queues[kind].pop(0))

    def _string_blocks(
        self,
        value: str,
        queues: dict[str, list[Any]],
        consumed: dict[str, int],
        *,
        location: str,
    ) -> list[ContentBlock]:
        if not isinstance(value, str):
            raise SpecError(f"{location} must be a string")
        blocks: list[ContentBlock] = []
        offset = 0
        for match in _PLACEHOLDER.finditer(value):
            if match.start() > offset:
                blocks.append(ContentBlock("text", value[offset : match.start()]))
            kind = match.group(1)
            blocks.append(
                self._take_media(
                    kind,
                    queues,
                    consumed,
                    location=location,
                )
            )
            offset = match.end()
        if offset < len(value):
            blocks.append(ContentBlock("text", value[offset:]))
        if not blocks and value == "":
            raise SpecError(f"{location} must not be empty")
        return blocks

    def _structured_blocks(
        self,
        content: Sequence[Any],
        queues: dict[str, list[Any]],
        consumed: dict[str, int],
        *,
        location: str,
    ) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []
        for index, item in enumerate(content):
            item_location = f"{location}[{index}]"
            if isinstance(item, str):
                blocks.extend(
                    self._string_blocks(
                        item,
                        queues,
                        consumed,
                        location=item_location,
                    )
                )
                continue
            if not isinstance(item, Mapping):
                raise SpecError(f"{item_location} must be a string or mapping")
            kind = item.get("type", item.get("kind"))
            if kind == "text":
                blocks.extend(
                    self._string_blocks(
                        item.get("text", item.get("value")),
                        queues,
                        consumed,
                        location=item_location,
                    )
                )
            elif kind in {"image", "video", "audio"}:
                embedded = item.get(kind, item.get("value"))
                if embedded is None:
                    blocks.append(
                        self._take_media(
                            kind,
                            queues,
                            consumed,
                            location=item_location,
                        )
                    )
                else:
                    embedded = self._media_value(
                        kind,
                        embedded,
                        location=item_location,
                    )
                    blocks.append(ContentBlock(kind, embedded))
            else:
                raise SpecError(f"{item_location} has unsupported content type {kind!r}")
        if not blocks:
            raise SpecError(f"{location} must not be empty")
        return blocks

    def _message_blocks(
        self,
        content: Any,
        queues: dict[str, list[Any]],
        consumed: dict[str, int],
        *,
        location: str,
    ) -> list[ContentBlock]:
        if isinstance(content, str):
            return self._string_blocks(
                content,
                queues,
                consumed,
                location=location,
            )
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            return self._structured_blocks(
                content,
                queues,
                consumed,
                location=location,
            )
        raise SpecError(f"{location} must be a string or content sequence")

    def _standard_messages(
        self,
        value: Any,
        queues: dict[str, list[Any]],
        consumed: dict[str, int],
    ) -> list[tuple[str, list[ContentBlock], Mapping[str, Any]]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise SpecError(f"{self.config.messages_column} must be a sequence")
        parsed = []
        for index, raw_message in enumerate(value):
            location = f"{self.config.messages_column}[{index}]"
            if not isinstance(raw_message, Mapping):
                raise SpecError(f"{location} must be a mapping")
            role = raw_message.get("role", raw_message.get("from"))
            if not isinstance(role, str) or not role.strip():
                raise SpecError(f"{location}.role must not be empty")
            role = _ROLE_ALIASES.get(role.strip().lower(), role.strip().lower())
            content = raw_message.get("content", raw_message.get("value"))
            blocks = self._message_blocks(
                content,
                queues,
                consumed,
                location=f"{location}.content",
            )
            metadata = raw_message.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise SpecError(f"{location}.metadata must be a mapping")
            parsed.append((role, blocks, metadata))
        if not parsed:
            raise SpecError(f"{self.config.messages_column} must not be empty")
        return parsed

    def _text_pair_messages(
        self,
        value: Any,
        queues: dict[str, list[Any]],
        consumed: dict[str, int],
    ) -> list[tuple[str, list[ContentBlock], Mapping[str, Any]]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise SpecError(f"{self.config.text_pairs_column} must be a sequence")
        parsed = []
        for index, pair in enumerate(value):
            location = f"{self.config.text_pairs_column}[{index}]"
            if not isinstance(pair, Mapping):
                raise SpecError(f"{location} must be a mapping")
            user = pair.get("user")
            assistant = pair.get("assistant")
            source = pair.get("source")
            pair_metadata = {} if source is None else {"source": source}
            parsed.append(
                (
                    "user",
                    self._message_blocks(
                        user,
                        queues,
                        consumed,
                        location=f"{location}.user",
                    ),
                    pair_metadata,
                )
            )
            parsed.append(
                (
                    "assistant",
                    self._message_blocks(
                        assistant,
                        queues,
                        consumed,
                        location=f"{location}.assistant",
                    ),
                    pair_metadata,
                )
            )
        if not parsed:
            raise SpecError(f"{self.config.text_pairs_column} must not be empty")
        return parsed

    def _query_messages(
        self,
        fields: Mapping[str, Any],
        queues: dict[str, list[Any]],
        consumed: dict[str, int],
    ) -> list[tuple[str, list[ContentBlock], Mapping[str, Any]]]:
        parsed = []
        system = fields.get(self.config.system_column)
        if system:
            parsed.append(
                (
                    "system",
                    self._message_blocks(
                        system,
                        queues,
                        consumed,
                        location=self.config.system_column,
                    ),
                    {},
                )
            )
        history = fields.get(self.config.history_column)
        if history is not None:
            for index, pair in enumerate(_sequence(history)):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise SpecError(
                        f"{self.config.history_column}[{index}] must be [query, response]"
                    )
                for role, value in zip(("user", "assistant"), pair, strict=True):
                    parsed.append(
                        (
                            role,
                            self._message_blocks(
                                value,
                                queues,
                                consumed,
                                location=f"{self.config.history_column}[{index}]",
                            ),
                            {},
                        )
                    )
        query = fields.get(self.config.query_column)
        response = fields.get(self.config.response_column)
        if query is None or response is None:
            raise SpecError(
                f"row requires both {self.config.query_column!r} and "
                f"{self.config.response_column!r}"
            )
        parsed.append(
            (
                "user",
                self._message_blocks(
                    query,
                    queues,
                    consumed,
                    location=self.config.query_column,
                ),
                {},
            )
        )
        parsed.append(
            (
                "assistant",
                self._message_blocks(
                    response,
                    queues,
                    consumed,
                    location=self.config.response_column,
                ),
                {},
            )
        )
        return parsed

    def _attach_unreferenced_media(
        self,
        messages: list[tuple[str, list[ContentBlock], Mapping[str, Any]]],
        queues: dict[str, list[Any]],
        consumed: dict[str, int],
    ) -> None:
        leftovers = {kind: values for kind, values in queues.items() if values}
        if not leftovers:
            return
        partially_referenced = [kind for kind in leftovers if consumed[kind]]
        if partially_referenced:
            raise SpecError(
                "media placeholder count mismatch for: "
                + ", ".join(sorted(partially_referenced))
            )
        if self.config.media_without_placeholders == "error":
            raise SpecError(
                "row contains media without placeholders: "
                + ", ".join(sorted(leftovers))
            )
        target = next(
            (index for index, item in enumerate(messages) if item[0] == "user"),
            None,
        )
        if target is None:
            raise SpecError("cannot prepend unreferenced media without a user message")
        prefix = [
            ContentBlock(kind, value)
            for kind in ("image", "video", "audio")
            for value in queues[kind]
        ]
        role, blocks, metadata = messages[target]
        messages[target] = (role, prefix + blocks, metadata)
        for values in queues.values():
            values.clear()

    def adapt(self, record: DataRecord) -> OmniSample:
        fields = record.fields
        queues = self._media_queues(fields)
        consumed = {"image": 0, "video": 0, "audio": 0}
        if fields.get(self.config.messages_column) is not None:
            parsed = self._standard_messages(
                fields[self.config.messages_column],
                queues,
                consumed,
            )
        elif fields.get(self.config.text_pairs_column) is not None:
            parsed = self._text_pair_messages(
                fields[self.config.text_pairs_column],
                queues,
                consumed,
            )
        elif fields.get(self.config.query_column) is not None:
            parsed = self._query_messages(fields, queues, consumed)
        else:
            text = fields.get(self.config.text_column)
            if text is None:
                raise SpecError(
                    "ms-swift row requires messages, texts, query/response, or text"
                )
            blocks = self._message_blocks(
                text,
                queues,
                consumed,
                location=self.config.text_column,
            )
            leftovers = {kind: values for kind, values in queues.items() if values}
            if any(consumed[kind] for kind in leftovers):
                raise SpecError(
                    "media placeholder count mismatch for: "
                    + ", ".join(
                        sorted(kind for kind in leftovers if consumed[kind])
                    )
                )
            if leftovers and self.config.media_without_placeholders == "error":
                raise SpecError(
                    "row contains media without placeholders: "
                    + ", ".join(sorted(leftovers))
                )
            prefix = [
                ContentBlock(kind, value)
                for kind in ("image", "video", "audio")
                for value in queues[kind]
            ]
            return OmniSample(
                sample_id=str(fields.get(self.config.sample_id_column) or record.sample_id),
                content=tuple(prefix + blocks),
                metadata=self._metadata(record),
            )
        self._attach_unreferenced_media(parsed, queues, consumed)
        messages = tuple(
            Message(role=role, content=tuple(blocks), metadata=metadata)
            for role, blocks, metadata in parsed
        )
        return OmniSample(
            sample_id=str(fields.get(self.config.sample_id_column) or record.sample_id),
            content=(),
            messages=messages,
            metadata=self._metadata(record),
        )

    def _metadata(self, record: DataRecord) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "trainomni.dataset": record.source,
            "trainomni.position": dict(record.position),
        }
        for column in self.config.metadata_columns:
            if column not in record.fields:
                raise SpecError(f"metadata column is missing: {column}")
            metadata[column] = record.fields[column]
        common = {
            name: record.fields[name]
            for name in ("tools", "objects", "label", "rejected_response")
            if name in record.fields and name not in metadata
        }
        if common:
            metadata["msswift"] = common
        return metadata

    @staticmethod
    def state_dict():
        return {}

    @staticmethod
    def load_state_dict(state) -> None:
        if state:
            raise CheckpointError("ms-swift adapter is stateless")


def descriptor() -> ModuleDescriptor:
    return ModuleDescriptor(
        module_id=ModuleId.parse("data_adapter:trainomni/msswift@1"),
        config_type=MSSwiftAdapterConfig,
        factory=lambda config, context: MSSwiftAdapter(config),
        provides=CapabilitySet.of({"data.sample.omni"}),
        requires=CapabilitySet.of({"data.record.row"}),
    )
