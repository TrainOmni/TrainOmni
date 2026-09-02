"""Canonical modality-neutral sample contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from ._identity import freeze_mapping, normalize_identity


@dataclass(frozen=True, slots=True)
class ContentBlock:
    kind: Literal["text", "image", "video", "audio"]
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.kind not in {"text", "image", "video", "audio"}:
            raise ValueError(f"unsupported content block kind: {self.kind!r}")
        object.__setattr__(
            self, "metadata", freeze_mapping(self.metadata, field="block metadata")
        )


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: tuple[ContentBlock, ...]
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("message role must not be empty")
        if not self.content:
            raise ValueError("message content must not be empty")
        if not isinstance(self.content, (tuple, list)) or any(
            not isinstance(block, ContentBlock) for block in self.content
        ):
            raise TypeError("message content must contain ContentBlock values")
        object.__setattr__(self, "role", normalize_identity(self.role, field="message role"))
        object.__setattr__(self, "content", tuple(self.content))
        object.__setattr__(
            self, "metadata", freeze_mapping(self.metadata, field="message metadata")
        )


@dataclass(frozen=True, slots=True)
class OmniSample:
    sample_id: str
    content: tuple[ContentBlock, ...]
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    messages: tuple[Message, ...] = ()

    def __post_init__(self) -> None:
        sample_id = normalize_identity(self.sample_id, field="sample_id")
        if not isinstance(self.content, (tuple, list)) or any(
            not isinstance(block, ContentBlock) for block in self.content
        ):
            raise TypeError("sample content must contain ContentBlock values")
        if not isinstance(self.messages, (tuple, list)) or any(
            not isinstance(message, Message) for message in self.messages
        ):
            raise TypeError("sample messages must contain Message values")
        if bool(self.content) == bool(self.messages):
            raise ValueError("sample requires exactly one of content or messages")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "content", tuple(self.content))
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(
            self, "metadata", freeze_mapping(self.metadata, field="sample metadata")
        )

    def map_blocks(self, transform: Callable[[ContentBlock], ContentBlock]) -> OmniSample:
        if self.content:
            return OmniSample(
                self.sample_id,
                tuple(transform(block) for block in self.content),
                self.metadata,
            )
        return OmniSample(
            self.sample_id,
            (),
            self.metadata,
            tuple(
                Message(
                    message.role,
                    tuple(transform(block) for block in message.content),
                    message.metadata,
                )
                for message in self.messages
            ),
        )
