"""Parsing, validation, serialization, and hashing for canonical samples."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import (
    BLOCK_TYPES,
    OBJECTIVES,
    ROLES,
    SCHEMA_VERSION,
    Asset,
    BBoxBlock,
    Candidate,
    CanonicalSample,
    ContentBlock,
    JsonBlock,
    MediaBlock,
    Message,
    PointBlock,
    Preference,
    Rollout,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    Verifier,
    freeze_json,
    thaw_json,
)

_SHA256_RE = re.compile(r"^[A-Fa-f0-9]{64}$")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "id",
    "objective",
    "messages",
    "assets",
    "tools",
    "preference",
    "rollout",
    "metadata",
}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable structural or semantic sample problem."""

    path: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: [{self.code}] {self.message}"


class SampleValidationError(ValueError):
    """Raised when a raw mapping cannot become a CanonicalSample."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Canonical sample validation failed:\n{details}")


def _add(
    issues: list[ValidationIssue], path: str, code: str, message: str
) -> None:
    issues.append(ValidationIssue(path=path, code=code, message=message))


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_keys(
    value: Mapping[str, Any],
    *,
    path: str,
    allowed: set[str],
    required: set[str],
    issues: list[ValidationIssue],
) -> None:
    for key in sorted(required - set(value)):
        _add(issues, f"{path}.{key}", "required", "field is required")
    for key in sorted(set(value) - allowed):
        _add(issues, f"{path}.{key}", "unknown_field", "field is not allowed")


def _check_nonempty_string(
    value: Any, path: str, issues: list[ValidationIssue]
) -> None:
    if not isinstance(value, str) or not value:
        _add(issues, path, "type.string_nonempty", "must be a non-empty string")


def _check_optional_string(
    owner: Mapping[str, Any], key: str, path: str, issues: list[ValidationIssue]
) -> None:
    if key in owner and not isinstance(owner[key], str):
        _add(issues, f"{path}.{key}", "type.string", "must be a string")


def _check_loss_weight(
    owner: Mapping[str, Any], path: str, issues: list[ValidationIssue]
) -> None:
    if "loss_weight" not in owner:
        return
    value = owner["loss_weight"]
    if not _is_number(value) or value < 0:
        _add(
            issues,
            f"{path}.loss_weight",
            "loss_weight.invalid",
            "must be a finite non-negative number",
        )


def _validate_json_value(
    value: Any, path: str, issues: list[ValidationIssue]
) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _add(issues, path, "json.non_finite", "JSON number must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]", issues)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _add(issues, path, "json.key", "JSON object keys must be strings")
                continue
            _validate_json_value(item, f"{path}.{key}", issues)
        return
    _add(
        issues,
        path,
        "json.type",
        f"value of type {type(value).__name__!r} is not JSON-compatible",
    )


def _validate_block(
    value: Any, path: str, issues: list[ValidationIssue]
) -> None:
    if not isinstance(value, dict):
        _add(issues, path, "type.object", "content block must be an object")
        return
    block_type = value.get("type")
    if block_type not in BLOCK_TYPES:
        _add(
            issues,
            f"{path}.type",
            "block.type",
            f"must be one of {sorted(BLOCK_TYPES)}",
        )
        return

    common = {"type", "loss_weight"}
    allowed_by_type = {
        "text": common | {"text"},
        "media": common | {"asset_id"},
        "bbox": common | {"asset_id", "xyxy", "coordinate_space", "label"},
        "point": common | {"asset_id", "xy", "coordinate_space", "label"},
        "json": common | {"value"},
        "tool_call": common | {"name", "arguments", "call_id"},
        "tool_result": common | {"value", "name", "call_id"},
    }
    required_by_type = {
        "text": {"type", "text"},
        "media": {"type", "asset_id"},
        "bbox": {"type", "asset_id", "xyxy", "coordinate_space"},
        "point": {"type", "asset_id", "xy", "coordinate_space"},
        "json": {"type", "value"},
        "tool_call": {"type", "name", "arguments"},
        "tool_result": {"type", "value"},
    }
    _check_keys(
        value,
        path=path,
        allowed=allowed_by_type[block_type],
        required=required_by_type[block_type],
        issues=issues,
    )
    _check_loss_weight(value, path, issues)

    if block_type == "text" and "text" in value and not isinstance(value["text"], str):
        _add(issues, f"{path}.text", "type.string", "must be a string")
    elif block_type == "media" and "asset_id" in value:
        _check_nonempty_string(value["asset_id"], f"{path}.asset_id", issues)
    elif block_type in {"bbox", "point"}:
        if "asset_id" in value:
            _check_nonempty_string(value["asset_id"], f"{path}.asset_id", issues)
        coordinate_key = "xyxy" if block_type == "bbox" else "xy"
        expected_length = 4 if block_type == "bbox" else 2
        if coordinate_key in value:
            coordinates = value[coordinate_key]
            if not isinstance(coordinates, list) or len(coordinates) != expected_length:
                _add(
                    issues,
                    f"{path}.{coordinate_key}",
                    "coordinates.shape",
                    f"must be an array of {expected_length} numbers",
                )
            elif not all(_is_number(item) for item in coordinates):
                _add(
                    issues,
                    f"{path}.{coordinate_key}",
                    "coordinates.number",
                    "all coordinates must be finite numbers",
                )
        if value.get("coordinate_space") not in {"pixel", "norm_0_1"}:
            _add(
                issues,
                f"{path}.coordinate_space",
                "coordinates.space",
                "must be 'pixel' or 'norm_0_1'",
            )
        _check_optional_string(value, "label", path, issues)
    elif block_type == "json" and "value" in value:
        _validate_json_value(value["value"], f"{path}.value", issues)
    elif block_type == "tool_call":
        if "name" in value:
            _check_nonempty_string(value["name"], f"{path}.name", issues)
        if "arguments" in value:
            if not isinstance(value["arguments"], dict):
                _add(
                    issues,
                    f"{path}.arguments",
                    "type.object",
                    "must be an object",
                )
            else:
                _validate_json_value(value["arguments"], f"{path}.arguments", issues)
        _check_optional_string(value, "call_id", path, issues)
    elif block_type == "tool_result":
        if "value" in value:
            _validate_json_value(value["value"], f"{path}.value", issues)
        _check_optional_string(value, "name", path, issues)
        _check_optional_string(value, "call_id", path, issues)


def _validate_message(
    value: Any, path: str, issues: list[ValidationIssue]
) -> None:
    if not isinstance(value, dict):
        _add(issues, path, "type.object", "message must be an object")
        return
    _check_keys(
        value,
        path=path,
        allowed={"role", "content", "name", "loss_weight"},
        required={"role", "content"},
        issues=issues,
    )
    if value.get("role") not in ROLES:
        _add(
            issues,
            f"{path}.role",
            "message.role",
            f"must be one of {sorted(ROLES)}",
        )
    _check_optional_string(value, "name", path, issues)
    _check_loss_weight(value, path, issues)
    content = value.get("content")
    if not isinstance(content, list) or not content:
        _add(
            issues,
            f"{path}.content",
            "content.nonempty",
            "must be a non-empty array",
        )
    else:
        for index, block in enumerate(content):
            _validate_block(block, f"{path}.content[{index}]", issues)


def _validate_messages(
    value: Any, path: str, issues: list[ValidationIssue]
) -> None:
    if not isinstance(value, list) or not value:
        _add(issues, path, "messages.nonempty", "must be a non-empty array")
        return
    for index, message in enumerate(value):
        _validate_message(message, f"{path}[{index}]", issues)


def _validate_asset(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, dict):
        _add(issues, path, "type.object", "asset must be an object")
        return
    _check_keys(
        value,
        path=path,
        allowed={
            "id",
            "modality",
            "uri",
            "mime_type",
            "sha256",
            "width",
            "height",
            "duration_seconds",
            "num_frames",
        },
        required={"id", "modality", "uri"},
        issues=issues,
    )
    if "id" in value:
        _check_nonempty_string(value["id"], f"{path}.id", issues)
    if value.get("modality") not in {"image", "video", "audio"}:
        _add(
            issues,
            f"{path}.modality",
            "asset.modality",
            "must be 'image', 'video', or 'audio'",
        )
    if "uri" in value:
        _check_nonempty_string(value["uri"], f"{path}.uri", issues)
    _check_optional_string(value, "mime_type", path, issues)
    if "sha256" in value and (
        not isinstance(value["sha256"], str) or not _SHA256_RE.fullmatch(value["sha256"])
    ):
        _add(
            issues,
            f"{path}.sha256",
            "asset.sha256",
            "must contain exactly 64 hexadecimal characters",
        )
    for key in ("width", "height", "num_frames"):
        if key in value and not _is_positive_int(value[key]):
            _add(issues, f"{path}.{key}", "number.positive_int", "must be > 0")
    if "duration_seconds" in value and (
        not _is_number(value["duration_seconds"]) or value["duration_seconds"] <= 0
    ):
        _add(
            issues,
            f"{path}.duration_seconds",
            "number.positive",
            "must be a finite number > 0",
        )


def _validate_candidate(
    value: Any, path: str, issues: list[ValidationIssue]
) -> None:
    if not isinstance(value, dict):
        _add(issues, path, "type.object", "candidate must be an object")
        return
    _check_keys(
        value,
        path=path,
        allowed={"messages", "score", "metadata"},
        required={"messages"},
        issues=issues,
    )
    if "messages" in value:
        _validate_messages(value["messages"], f"{path}.messages", issues)
    if "score" in value and not _is_number(value["score"]):
        _add(issues, f"{path}.score", "number.finite", "must be a finite number")
    if "metadata" in value:
        if not isinstance(value["metadata"], dict):
            _add(issues, f"{path}.metadata", "type.object", "must be an object")
        else:
            _validate_json_value(value["metadata"], f"{path}.metadata", issues)


def _validate_preference(
    value: Any, path: str, issues: list[ValidationIssue]
) -> None:
    if not isinstance(value, dict):
        _add(issues, path, "type.object", "preference must be an object")
        return
    _check_keys(
        value,
        path=path,
        allowed={"chosen", "rejected", "margin", "judge"},
        required={"chosen", "rejected"},
        issues=issues,
    )
    if "chosen" in value:
        _validate_candidate(value["chosen"], f"{path}.chosen", issues)
    if "rejected" in value:
        _validate_candidate(value["rejected"], f"{path}.rejected", issues)
    if "margin" in value and not _is_number(value["margin"]):
        _add(issues, f"{path}.margin", "number.finite", "must be a finite number")
    _check_optional_string(value, "judge", path, issues)


def _validate_rollout(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, dict):
        _add(issues, path, "type.object", "rollout must be an object")
        return
    _check_keys(
        value,
        path=path,
        allowed={
            "verifiers",
            "environment",
            "max_completion_tokens",
            "reference_answer",
        },
        required={"verifiers"},
        issues=issues,
    )
    verifiers = value.get("verifiers")
    if not isinstance(verifiers, list) or not verifiers:
        _add(
            issues,
            f"{path}.verifiers",
            "verifiers.nonempty",
            "must be a non-empty array",
        )
    else:
        for index, verifier in enumerate(verifiers):
            verifier_path = f"{path}.verifiers[{index}]"
            if not isinstance(verifier, dict):
                _add(issues, verifier_path, "type.object", "verifier must be an object")
                continue
            _check_keys(
                verifier,
                path=verifier_path,
                allowed={"type", "weight", "spec"},
                required={"type"},
                issues=issues,
            )
            if "type" in verifier:
                _check_nonempty_string(verifier["type"], f"{verifier_path}.type", issues)
            if "weight" in verifier and not _is_number(verifier["weight"]):
                _add(
                    issues,
                    f"{verifier_path}.weight",
                    "number.finite",
                    "must be a finite number",
                )
            if "spec" in verifier:
                if not isinstance(verifier["spec"], dict):
                    _add(
                        issues,
                        f"{verifier_path}.spec",
                        "type.object",
                        "must be an object",
                    )
                else:
                    _validate_json_value(verifier["spec"], f"{verifier_path}.spec", issues)
    _check_optional_string(value, "environment", path, issues)
    if "max_completion_tokens" in value and not _is_positive_int(
        value["max_completion_tokens"]
    ):
        _add(
            issues,
            f"{path}.max_completion_tokens",
            "number.positive_int",
            "must be > 0",
        )
    if "reference_answer" in value:
        _validate_json_value(value["reference_answer"], f"{path}.reference_answer", issues)


def _validate_metadata(
    value: Any, path: str, issues: list[ValidationIssue]
) -> None:
    if not isinstance(value, dict):
        _add(issues, path, "type.object", "metadata must be an object")
        return
    _validate_json_value(value, path, issues)
    for key in ("source", "split", "license", "language"):
        _check_optional_string(value, key, path, issues)
    for key in ("quality", "difficulty"):
        if key in value and not _is_number(value[key]):
            _add(issues, f"{path}.{key}", "number.finite", "must be a finite number")
    if "tags" in value and (
        not isinstance(value["tags"], list)
        or not all(isinstance(tag, str) for tag in value["tags"])
    ):
        _add(issues, f"{path}.tags", "metadata.tags", "must be an array of strings")


def _validate_structure(data: Any) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(data, dict):
        _add(issues, "$", "type.object", "sample must be an object")
        return issues
    _check_keys(
        data,
        path="$",
        allowed=_TOP_LEVEL_KEYS,
        required={"schema_version", "id", "objective", "messages"},
        issues=issues,
    )
    if data.get("schema_version") != SCHEMA_VERSION:
        _add(
            issues,
            "$.schema_version",
            "schema.version",
            f"must equal {SCHEMA_VERSION!r}",
        )
    if "id" in data:
        _check_nonempty_string(data["id"], "$.id", issues)
    if data.get("objective") not in OBJECTIVES:
        _add(
            issues,
            "$.objective",
            "objective.type",
            f"must be one of {sorted(OBJECTIVES)}",
        )
    if "messages" in data:
        _validate_messages(data["messages"], "$.messages", issues)
    if "assets" in data:
        if not isinstance(data["assets"], list):
            _add(issues, "$.assets", "type.array", "must be an array")
        else:
            for index, asset in enumerate(data["assets"]):
                _validate_asset(asset, f"$.assets[{index}]", issues)
    if "tools" in data:
        if not isinstance(data["tools"], list):
            _add(issues, "$.tools", "type.array", "must be an array")
        else:
            for index, tool in enumerate(data["tools"]):
                if not isinstance(tool, dict):
                    _add(issues, f"$.tools[{index}]", "type.object", "tool must be an object")
                else:
                    _validate_json_value(tool, f"$.tools[{index}]", issues)
    if "preference" in data:
        _validate_preference(data["preference"], "$.preference", issues)
    if "rollout" in data:
        _validate_rollout(data["rollout"], "$.rollout", issues)
    if "metadata" in data:
        _validate_metadata(data["metadata"], "$.metadata", issues)
    return issues


def _iter_blocks(
    messages: Sequence[Message], base_path: str
) -> Iterable[tuple[Message, ContentBlock, str]]:
    for message_index, message in enumerate(messages):
        for block_index, block in enumerate(message.content):
            yield message, block, f"{base_path}[{message_index}].content[{block_index}]"


def _default_loss_weight(objective: str, role: str) -> float:
    if objective == "cpt" and role == "document":
        return 1.0
    if objective in {"sft", "preference"} and role == "assistant":
        return 1.0
    return 0.0


def _has_trainable_target(messages: Sequence[Message], objective: str) -> bool:
    for message, block, _ in _iter_blocks(messages, "messages"):
        weight = block.loss_weight
        if weight is None:
            weight = message.loss_weight
        if weight is None:
            weight = _default_loss_weight(objective, message.role)
        if weight > 0:
            return True
    return False


def _validate_tool_ids(
    messages: Sequence[Message], base_path: str, issues: list[ValidationIssue]
) -> None:
    calls: dict[str, str] = {}
    results: dict[str, str] = {}
    for _, block, path in _iter_blocks(messages, base_path):
        if isinstance(block, ToolCallBlock) and block.call_id:
            if block.call_id in calls:
                _add(
                    issues,
                    f"{path}.call_id",
                    "tool_call.duplicate_id",
                    f"call_id {block.call_id!r} was already used at {calls[block.call_id]}",
                )
            calls[block.call_id] = path
        elif isinstance(block, ToolResultBlock) and block.call_id:
            if block.call_id in results:
                _add(
                    issues,
                    f"{path}.call_id",
                    "tool_result.duplicate_id",
                    f"call_id {block.call_id!r} already has a result at {results[block.call_id]}",
                )
            results[block.call_id] = path
    for call_id, result_path in results.items():
        if call_id not in calls:
            _add(
                issues,
                f"{result_path}.call_id",
                "tool_result.orphan",
                f"no tool_call uses call_id {call_id!r}",
            )


def _validate_coordinates(
    messages: Sequence[Message],
    base_path: str,
    assets: Mapping[str, Asset],
    issues: list[ValidationIssue],
) -> None:
    for _, block, path in _iter_blocks(messages, base_path):
        if (
            isinstance(block, (MediaBlock, BBoxBlock, PointBlock))
            and block.asset_id not in assets
        ):
            _add(
                issues,
                f"{path}.asset_id",
                "asset.missing_reference",
                f"asset_id {block.asset_id!r} is not defined",
            )
            continue
        if isinstance(block, BBoxBlock):
            x1, y1, x2, y2 = block.xyxy
            if x1 >= x2 or y1 >= y2:
                _add(
                    issues,
                    f"{path}.xyxy",
                    "bbox.order",
                    "requires x1 < x2 and y1 < y2",
                )
            if block.coordinate_space == "norm_0_1":
                if any(value < 0 or value > 1 for value in block.xyxy):
                    _add(
                        issues,
                        f"{path}.xyxy",
                        "bbox.bounds.normalized",
                        "normalized coordinates must be within [0, 1]",
                    )
            else:
                asset = assets[block.asset_id]
                if asset.width is None or asset.height is None:
                    _add(
                        issues,
                        f"{path}.coordinate_space",
                        "bbox.asset_dimensions",
                        "pixel coordinates require asset width and height",
                    )
                elif not (
                    0 <= x1 <= asset.width
                    and 0 <= x2 <= asset.width
                    and 0 <= y1 <= asset.height
                    and 0 <= y2 <= asset.height
                ):
                    _add(
                        issues,
                        f"{path}.xyxy",
                        "bbox.bounds.pixel",
                        f"coordinates exceed asset size {asset.width}x{asset.height}",
                    )
        elif isinstance(block, PointBlock):
            x, y = block.xy
            if block.coordinate_space == "norm_0_1":
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    _add(
                        issues,
                        f"{path}.xy",
                        "point.bounds.normalized",
                        "normalized coordinates must be within [0, 1]",
                    )
            else:
                asset = assets[block.asset_id]
                if asset.width is None or asset.height is None:
                    _add(
                        issues,
                        f"{path}.coordinate_space",
                        "point.asset_dimensions",
                        "pixel coordinates require asset width and height",
                    )
                elif not (0 <= x <= asset.width and 0 <= y <= asset.height):
                    _add(
                        issues,
                        f"{path}.xy",
                        "point.bounds.pixel",
                        f"coordinates exceed asset size {asset.width}x{asset.height}",
                    )


def _semantic_issues(sample: CanonicalSample) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    assets: dict[str, Asset] = {}
    for index, asset in enumerate(sample.assets):
        if asset.id in assets:
            _add(
                issues,
                f"$.assets[{index}].id",
                "asset.duplicate_id",
                f"asset_id {asset.id!r} is duplicated",
            )
        else:
            assets[asset.id] = asset

    message_sets: list[tuple[Sequence[Message], str]] = [(sample.messages, "$.messages")]
    if sample.preference:
        message_sets.extend(
            [
                (sample.preference.chosen.messages, "$.preference.chosen.messages"),
                (sample.preference.rejected.messages, "$.preference.rejected.messages"),
            ]
        )
    for messages, path in message_sets:
        _validate_coordinates(messages, path, assets, issues)
        _validate_tool_ids(messages, path, issues)

    if sample.objective == "cpt":
        if sample.preference or sample.rollout:
            _add(
                issues,
                "$",
                "objective.unexpected_fields",
                "cpt samples cannot contain preference or rollout",
            )
        if not any(message.role == "document" for message in sample.messages):
            _add(
                issues,
                "$.messages",
                "cpt.document_required",
                "cpt requires at least one document message",
            )
        if not _has_trainable_target(sample.messages, "cpt"):
            _add(
                issues,
                "$.messages",
                "loss.no_target",
                "cpt sample has no content with positive loss weight",
            )
    elif sample.objective == "sft":
        if sample.preference or sample.rollout:
            _add(
                issues,
                "$",
                "objective.unexpected_fields",
                "sft samples cannot contain preference or rollout",
            )
        if not any(message.role == "assistant" for message in sample.messages):
            _add(
                issues,
                "$.messages",
                "sft.assistant_required",
                "sft requires at least one assistant message",
            )
        if not _has_trainable_target(sample.messages, "sft"):
            _add(
                issues,
                "$.messages",
                "loss.no_target",
                "sft sample has no content with positive loss weight",
            )
    elif sample.objective == "preference":
        if sample.preference is None:
            _add(
                issues,
                "$.preference",
                "preference.required",
                "preference objective requires chosen and rejected candidates",
            )
        if sample.rollout:
            _add(
                issues,
                "$.rollout",
                "objective.unexpected_rollout",
                "preference samples cannot contain rollout",
            )
        if sample.preference:
            for name, candidate in (
                ("chosen", sample.preference.chosen),
                ("rejected", sample.preference.rejected),
            ):
                if not any(message.role == "assistant" for message in candidate.messages):
                    _add(
                        issues,
                        f"$.preference.{name}.messages",
                        "preference.assistant_required",
                        "candidate requires an assistant continuation",
                    )
                if not _has_trainable_target(candidate.messages, "preference"):
                    _add(
                        issues,
                        f"$.preference.{name}.messages",
                        "loss.no_target",
                        "candidate has no content with positive loss weight",
                    )
            if candidate_to_dict(sample.preference.chosen) == candidate_to_dict(
                sample.preference.rejected
            ):
                _add(
                    issues,
                    "$.preference",
                    "preference.identical",
                    "chosen and rejected candidates must differ",
                )
    elif sample.objective == "prompt_only":
        if sample.preference:
            _add(
                issues,
                "$.preference",
                "objective.unexpected_preference",
                "prompt_only samples cannot contain preference",
            )
        if sample.rollout is None:
            _add(
                issues,
                "$.rollout",
                "rollout.required",
                "prompt_only objective requires rollout configuration",
            )

    return issues


def validate_sample_dict(data: Any) -> tuple[ValidationIssue, ...]:
    """Return all currently detectable issues without raising."""

    structural = _validate_structure(data)
    if structural:
        return tuple(structural)
    sample = _parse_validated_sample(data)
    return tuple(_semantic_issues(sample))


def _loss(value: Mapping[str, Any]) -> float | None:
    raw = value.get("loss_weight")
    return float(raw) if raw is not None else None


def _parse_block(value: Mapping[str, Any]) -> ContentBlock:
    block_type = value["type"]
    loss_weight = _loss(value)
    if block_type == "text":
        return TextBlock(text=value["text"], loss_weight=loss_weight)
    if block_type == "media":
        return MediaBlock(asset_id=value["asset_id"], loss_weight=loss_weight)
    if block_type == "bbox":
        return BBoxBlock(
            asset_id=value["asset_id"],
            xyxy=tuple(float(item) for item in value["xyxy"]),  # type: ignore[arg-type]
            coordinate_space=value["coordinate_space"],
            label=value.get("label"),
            loss_weight=loss_weight,
        )
    if block_type == "point":
        return PointBlock(
            asset_id=value["asset_id"],
            xy=tuple(float(item) for item in value["xy"]),  # type: ignore[arg-type]
            coordinate_space=value["coordinate_space"],
            label=value.get("label"),
            loss_weight=loss_weight,
        )
    if block_type == "json":
        return JsonBlock(value=freeze_json(value["value"]), loss_weight=loss_weight)
    if block_type == "tool_call":
        return ToolCallBlock(
            name=value["name"],
            arguments=freeze_json(value["arguments"]),
            call_id=value.get("call_id"),
            loss_weight=loss_weight,
        )
    if block_type == "tool_result":
        return ToolResultBlock(
            value=freeze_json(value["value"]),
            name=value.get("name"),
            call_id=value.get("call_id"),
            loss_weight=loss_weight,
        )
    raise AssertionError(f"unreachable block type: {block_type}")


def _parse_message(value: Mapping[str, Any]) -> Message:
    return Message(
        role=value["role"],
        content=tuple(_parse_block(block) for block in value["content"]),
        name=value.get("name"),
        loss_weight=_loss(value),
    )


def _parse_candidate(value: Mapping[str, Any]) -> Candidate:
    score = value.get("score")
    return Candidate(
        messages=tuple(_parse_message(message) for message in value["messages"]),
        score=float(score) if score is not None else None,
        metadata=freeze_json(value.get("metadata", {})),
    )


def _parse_preference(value: Mapping[str, Any]) -> Preference:
    margin = value.get("margin")
    return Preference(
        chosen=_parse_candidate(value["chosen"]),
        rejected=_parse_candidate(value["rejected"]),
        margin=float(margin) if margin is not None else None,
        judge=value.get("judge"),
    )


def _parse_rollout(value: Mapping[str, Any]) -> Rollout:
    return Rollout(
        verifiers=tuple(
            Verifier(
                type=verifier["type"],
                weight=float(verifier.get("weight", 1.0)),
                spec=freeze_json(verifier.get("spec", {})),
            )
            for verifier in value["verifiers"]
        ),
        environment=value.get("environment"),
        max_completion_tokens=value.get("max_completion_tokens"),
        reference_answer=freeze_json(value.get("reference_answer")),
    )


def _parse_validated_sample(data: Mapping[str, Any]) -> CanonicalSample:
    assets = []
    for value in data.get("assets", []):
        duration = value.get("duration_seconds")
        assets.append(
            Asset(
                id=value["id"],
                modality=value["modality"],
                uri=value["uri"],
                mime_type=value.get("mime_type"),
                sha256=value.get("sha256"),
                width=value.get("width"),
                height=value.get("height"),
                duration_seconds=float(duration) if duration is not None else None,
                num_frames=value.get("num_frames"),
            )
        )
    return CanonicalSample(
        id=data["id"],
        objective=data["objective"],
        messages=tuple(_parse_message(message) for message in data["messages"]),
        assets=tuple(assets),
        tools=tuple(freeze_json(tool) for tool in data.get("tools", [])),
        preference=(
            _parse_preference(data["preference"]) if "preference" in data else None
        ),
        rollout=_parse_rollout(data["rollout"]) if "rollout" in data else None,
        metadata=freeze_json(data.get("metadata", {})),
    )


def parse_sample(data: Any) -> CanonicalSample:
    """Validate and parse a raw JSON-compatible value."""

    issues = validate_sample_dict(data)
    if issues:
        raise SampleValidationError(issues)
    return _parse_validated_sample(data)


def load_sample(path: str | Path) -> CanonicalSample:
    """Load one canonical sample from a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return parse_sample(json.load(handle))


def block_to_dict(block: ContentBlock) -> dict[str, Any]:
    value: dict[str, Any]
    if isinstance(block, TextBlock):
        value = {"type": "text", "text": block.text}
    elif isinstance(block, MediaBlock):
        value = {"type": "media", "asset_id": block.asset_id}
    elif isinstance(block, BBoxBlock):
        value = {
            "type": "bbox",
            "asset_id": block.asset_id,
            "xyxy": list(block.xyxy),
            "coordinate_space": block.coordinate_space,
        }
        if block.label is not None:
            value["label"] = block.label
    elif isinstance(block, PointBlock):
        value = {
            "type": "point",
            "asset_id": block.asset_id,
            "xy": list(block.xy),
            "coordinate_space": block.coordinate_space,
        }
        if block.label is not None:
            value["label"] = block.label
    elif isinstance(block, JsonBlock):
        value = {"type": "json", "value": thaw_json(block.value)}
    elif isinstance(block, ToolCallBlock):
        value = {
            "type": "tool_call",
            "name": block.name,
            "arguments": thaw_json(block.arguments),
        }
        if block.call_id is not None:
            value["call_id"] = block.call_id
    elif isinstance(block, ToolResultBlock):
        value = {"type": "tool_result", "value": thaw_json(block.value)}
        if block.name is not None:
            value["name"] = block.name
        if block.call_id is not None:
            value["call_id"] = block.call_id
    else:
        raise TypeError(f"unknown content block: {type(block).__name__}")
    if block.loss_weight is not None:
        value["loss_weight"] = block.loss_weight
    return value


def message_to_dict(message: Message) -> dict[str, Any]:
    value: dict[str, Any] = {
        "role": message.role,
        "content": [block_to_dict(block) for block in message.content],
    }
    if message.name is not None:
        value["name"] = message.name
    if message.loss_weight is not None:
        value["loss_weight"] = message.loss_weight
    return value


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    value: dict[str, Any] = {
        "messages": [message_to_dict(message) for message in candidate.messages]
    }
    if candidate.score is not None:
        value["score"] = candidate.score
    if candidate.metadata:
        value["metadata"] = thaw_json(candidate.metadata)
    return value


def sample_to_dict(sample: CanonicalSample) -> dict[str, Any]:
    """Return the normalized JSON representation used for hashing."""

    value: dict[str, Any] = {
        "schema_version": sample.schema_version,
        "id": sample.id,
        "objective": sample.objective,
        "messages": [message_to_dict(message) for message in sample.messages],
    }
    if sample.assets:
        assets = []
        for asset in sample.assets:
            item: dict[str, Any] = {
                "id": asset.id,
                "modality": asset.modality,
                "uri": asset.uri,
            }
            for key in (
                "mime_type",
                "sha256",
                "width",
                "height",
                "duration_seconds",
                "num_frames",
            ):
                field_value = getattr(asset, key)
                if field_value is not None:
                    item[key] = field_value
            assets.append(item)
        value["assets"] = assets
    if sample.tools:
        value["tools"] = [thaw_json(tool) for tool in sample.tools]
    if sample.preference:
        preference: dict[str, Any] = {
            "chosen": candidate_to_dict(sample.preference.chosen),
            "rejected": candidate_to_dict(sample.preference.rejected),
        }
        if sample.preference.margin is not None:
            preference["margin"] = sample.preference.margin
        if sample.preference.judge is not None:
            preference["judge"] = sample.preference.judge
        value["preference"] = preference
    if sample.rollout:
        rollout: dict[str, Any] = {
            "verifiers": [
                {
                    "type": verifier.type,
                    "weight": verifier.weight,
                    **(
                        {"spec": thaw_json(verifier.spec)} if verifier.spec else {}
                    ),
                }
                for verifier in sample.rollout.verifiers
            ]
        }
        if sample.rollout.environment is not None:
            rollout["environment"] = sample.rollout.environment
        if sample.rollout.max_completion_tokens is not None:
            rollout["max_completion_tokens"] = sample.rollout.max_completion_tokens
        if sample.rollout.reference_answer is not None:
            rollout["reference_answer"] = thaw_json(sample.rollout.reference_answer)
        value["rollout"] = rollout
    if sample.metadata:
        value["metadata"] = thaw_json(sample.metadata)
    return value


def canonical_json(sample: CanonicalSample) -> str:
    """Serialize a sample deterministically, independent of source key order."""

    return json.dumps(
        sample_to_dict(sample),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(sample: CanonicalSample) -> str:
    """Return a versioned SHA-256 fingerprint of normalized sample semantics."""

    digest = hashlib.sha256(canonical_json(sample).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
