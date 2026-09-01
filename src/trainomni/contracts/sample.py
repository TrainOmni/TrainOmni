"""Canonical modality-neutral sample contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ContentBlock:
    kind: Literal["text", "image", "video", "audio"]
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.kind not in {"text", "image", "video", "audio"}:
            raise ValueError(f"unsupported content block kind: {self.kind!r}")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


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
        object.__setattr__(self, "role", self.role.strip())
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class OmniSample:
    sample_id: str
    content: tuple[ContentBlock, ...]
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    messages: tuple[Message, ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if bool(self.content) == bool(self.messages):
            raise ValueError("sample requires exactly one of content or messages")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

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
