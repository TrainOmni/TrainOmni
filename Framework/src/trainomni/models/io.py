"""Typed boundary between model plugins, planners, objectives and engines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from trainomni.contracts import BatchPlan, CostVector


@dataclass(frozen=True, slots=True)
class SourceSpan:
    field: str
    start: int
    end: int
    source_path: str
    loss_weight: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("source span bounds are invalid")


@dataclass(frozen=True, slots=True)
class EncodedSample:
    sample_id: str
    model_inputs: Mapping[str, Any]
    cost: CostVector
    source_spans: tuple[SourceSpan, ...] = ()
    trace: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("encoded sample_id must not be blank")
        if not self.model_inputs:
            raise ValueError("encoded sample model_inputs must not be empty")
        object.__setattr__(self, "model_inputs", MappingProxyType(dict(self.model_inputs)))
        object.__setattr__(self, "trace", MappingProxyType(dict(self.trace)))


@dataclass(frozen=True, slots=True)
class ModelBatch:
    sample_ids: tuple[str, ...]
    model_inputs: Mapping[str, Any]
    plan: BatchPlan
    trace: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ValueError("model batch sample_ids must not be empty")
        expected = tuple(item.sample_id for item in self.plan.items)
        if self.sample_ids != expected:
            raise ValueError("model batch sample order differs from batch plan")
        if not self.model_inputs:
            raise ValueError("model batch inputs must not be empty")
        object.__setattr__(self, "model_inputs", MappingProxyType(dict(self.model_inputs)))
        object.__setattr__(self, "trace", MappingProxyType(dict(self.trace)))


def summarize_value(value: Any, *, max_sequence_items: int = 16) -> Any:
    """Summarize tensor-like objects without materializing payloads."""

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            normalized_shape = [int(item) for item in shape]
        except (TypeError, ValueError):
            normalized_shape = str(shape)
        result = {
            "kind": "tensor_like",
            "shape": normalized_shape,
            "dtype": str(dtype),
        }
        device = getattr(value, "device", None)
        if device is not None:
            result["device"] = str(device)
        return result
    if isinstance(value, Mapping):
        return {
            str(key): summarize_value(item, max_sequence_items=max_sequence_items)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [
            summarize_value(item, max_sequence_items=max_sequence_items)
            for item in value[:max_sequence_items]
        ]
        if len(value) > max_sequence_items:
            items.append({"truncated_items": len(value) - max_sequence_items})
        return items
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "bytes", "length": len(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"kind": type(value).__name__, "repr": repr(value)[:200]}


def inspect_encoded_sample(encoded: EncodedSample) -> dict[str, Any]:
    return {
        "sample_id": encoded.sample_id,
        "cost": encoded.cost.to_dict(),
        "model_inputs": summarize_value(encoded.model_inputs),
        "source_spans": [
            {
                "field": span.field,
                "start": span.start,
                "end": span.end,
                "source_path": span.source_path,
                "loss_weight": span.loss_weight,
            }
            for span in encoded.source_spans
        ],
        "trace": summarize_value(encoded.trace),
    }


def inspect_model_batch(batch: ModelBatch) -> dict[str, Any]:
    return {
        "sample_ids": list(batch.sample_ids),
        "plan": batch.plan.to_dict(),
        "model_inputs": summarize_value(batch.model_inputs),
        "trace": summarize_value(batch.trace),
    }
