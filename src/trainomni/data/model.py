"""Typed canonical sample models.

These types deliberately have no dependency on Transformers, TRL, NeMo, or a
model-specific processor. They represent task semantics before tokenization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

SCHEMA_VERSION = "trainomni.sample.v0.1"
OBJECTIVES = frozenset({"cpt", "sft", "preference", "prompt_only"})
ROLES = frozenset({"system", "user", "assistant", "tool", "document"})
BLOCK_TYPES = frozenset(
    {"text", "media", "bbox", "point", "json", "tool_call", "tool_result"}
)

Objective: TypeAlias = Literal["cpt", "sft", "preference", "prompt_only"]
Role: TypeAlias = Literal["system", "user", "assistant", "tool", "document"]
CoordinateSpace: TypeAlias = Literal["pixel", "norm_0_1"]
Modality: TypeAlias = Literal["image", "video", "audio"]


def freeze_json(value: Any) -> Any:
    """Deep-freeze JSON-compatible data used inside frozen dataclasses."""

    if isinstance(value, dict):
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    return value


def thaw_json(value: Any) -> Any:
    """Convert frozen JSON-compatible data back to dict/list values."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    loss_weight: float | None = None
    type: Literal["text"] = field(default="text", init=False)


@dataclass(frozen=True, slots=True)
class MediaBlock:
    asset_id: str
    loss_weight: float | None = None
    type: Literal["media"] = field(default="media", init=False)


@dataclass(frozen=True, slots=True)
class BBoxBlock:
    asset_id: str
    xyxy: tuple[float, float, float, float]
    coordinate_space: CoordinateSpace
    label: str | None = None
    loss_weight: float | None = None
    type: Literal["bbox"] = field(default="bbox", init=False)


@dataclass(frozen=True, slots=True)
class PointBlock:
    asset_id: str
    xy: tuple[float, float]
    coordinate_space: CoordinateSpace
    label: str | None = None
    loss_weight: float | None = None
    type: Literal["point"] = field(default="point", init=False)


@dataclass(frozen=True, slots=True)
class JsonBlock:
    value: Any
    loss_weight: float | None = None
    type: Literal["json"] = field(default="json", init=False)


@dataclass(frozen=True, slots=True)
class ToolCallBlock:
    name: str
    arguments: Mapping[str, Any]
    call_id: str | None = None
    loss_weight: float | None = None
    type: Literal["tool_call"] = field(default="tool_call", init=False)


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    value: Any
    name: str | None = None
    call_id: str | None = None
    loss_weight: float | None = None
    type: Literal["tool_result"] = field(default="tool_result", init=False)


ContentBlock: TypeAlias = (
    TextBlock
    | MediaBlock
    | BBoxBlock
    | PointBlock
    | JsonBlock
    | ToolCallBlock
    | ToolResultBlock
)


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: tuple[ContentBlock, ...]
    name: str | None = None
    loss_weight: float | None = None


@dataclass(frozen=True, slots=True)
class Asset:
    id: str
    modality: Modality
    uri: str
    mime_type: str | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    num_frames: int | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    messages: tuple[Message, ...]
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class Preference:
    chosen: Candidate
    rejected: Candidate
    margin: float | None = None
    judge: str | None = None


@dataclass(frozen=True, slots=True)
class Verifier:
    type: str
    weight: float = 1.0
    spec: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class Rollout:
    verifiers: tuple[Verifier, ...]
    environment: str | None = None
    max_completion_tokens: int | None = None
    reference_answer: Any = None


@dataclass(frozen=True, slots=True)
class CanonicalSample:
    id: str
    objective: Objective
    messages: tuple[Message, ...]
    assets: tuple[Asset, ...] = ()
    tools: tuple[Mapping[str, Any], ...] = ()
    preference: Preference | None = None
    rollout: Rollout | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    schema_version: str = SCHEMA_VERSION

