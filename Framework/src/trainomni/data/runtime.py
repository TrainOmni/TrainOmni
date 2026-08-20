"""Stateful encode/plan/collate stream used by execution engines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from trainomni.config import DataSpec
from trainomni.contracts import BatchBudget, CostVector
from trainomni.models import EncodedSample, ModelBatch, SourceSpan

from .batching import BatchPlanningError, GreedyBatchPlanner
from .mixture import MixtureStream

BATCH_STREAM_STATE_VERSION = "trainomni.batch-stream.v1"
_BUDGET_FIELDS = {
    "max_samples",
    "max_text_tokens",
    "max_vision_tokens",
    "max_pixels",
    "max_frames",
    "max_audio_seconds",
    "max_model_units",
}


def batch_budget_from_data(data: DataSpec, *, default_max_samples: int = 1) -> BatchBudget:
    unknown = set(data.config) - (_BUDGET_FIELDS | {"repeat"})
    if unknown:
        raise BatchPlanningError(f"unknown data runtime config fields: {sorted(unknown)}")
    values = {key: data.config[key] for key in _BUDGET_FIELDS if key in data.config}
    values.setdefault("max_samples", default_max_samples)
    return BatchBudget(**values)


class StatefulBatchStream:
    """Make model batches and preserve the one-sample look-ahead exactly."""

    def __init__(
        self,
        mixture: MixtureStream,
        *,
        plugin: Any,
        sample_objective: str,
        stage_id: str,
        budget: BatchBudget,
        packing: bool,
        data_spec: DataSpec | None = None,
    ) -> None:
        self.mixture = mixture
        self.plugin = plugin
        self.sample_objective = sample_objective
        self.stage_id = stage_id
        self.budget = budget
        self.packing = packing
        self.data_spec = data_spec
        self._pending: EncodedSample | None = None
        self._batches = 0
        self._samples = 0

    def __iter__(self) -> StatefulBatchStream:
        return self

    def __next__(self) -> ModelBatch:
        current: list[EncodedSample] = []
        cost = CostVector()
        while True:
            try:
                encoded = self._take_encoded()
            except StopIteration:
                if not current:
                    raise
                break
            exceeded_alone = self.budget.exceeded_by(encoded.cost, 1)
            if exceeded_alone:
                raise BatchPlanningError(
                    f"sample {encoded.sample_id!r} exceeds batch budget: "
                    f"{list(exceeded_alone)}"
                )
            combined = cost + encoded.cost
            if current and self.budget.exceeded_by(combined, len(current) + 1):
                self._pending = encoded
                break
            current.append(encoded)
            cost = combined
            if self.budget.exceeded_by(cost, len(current) + 1):
                break
            if self.budget.max_samples is not None and len(current) >= self.budget.max_samples:
                break
        plans = GreedyBatchPlanner(self.budget, packing=self.packing).plan(current)
        if len(plans) != 1:
            raise BatchPlanningError("stateful batch assembly produced an invalid plan")
        batch = self.plugin.collate(current, plans[0])
        if not isinstance(batch, ModelBatch):
            raise TypeError(
                f"plugin.collate() must return ModelBatch, got {type(batch).__name__}"
            )
        self._batches += 1
        self._samples += len(current)
        return batch

    def _take_encoded(self) -> EncodedSample:
        if self._pending is not None:
            encoded = self._pending
            self._pending = None
            return encoded
        imported = next(self.mixture)
        if self.data_spec is not None:
            validate_sample_against_data(
                imported.sample, self.data_spec, self.sample_objective
            )
        issues = self.plugin.validate_sample(imported.sample, self.sample_objective)
        if issues:
            details = "; ".join(f"{item.code}: {item.message}" for item in issues)
            raise ValueError(
                f"model plugin rejected sample {imported.sample.id!r}: {details}"
            )
        encoded = self.plugin.encode(
            imported.sample,
            {
                "stage_id": self.stage_id,
                "objective": self.sample_objective,
                "inspect": False,
                "source_trace": imported.trace.to_dict(),
            },
        )
        if not isinstance(encoded, EncodedSample):
            raise TypeError(
                f"plugin.encode() must return EncodedSample, got {type(encoded).__name__}"
            )
        if encoded.sample_id != imported.sample.id:
            raise ValueError("plugin.encode() changed the canonical sample ID")
        return encoded

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": BATCH_STREAM_STATE_VERSION,
            "sample_objective": self.sample_objective,
            "stage_id": self.stage_id,
            "budget": self.budget.to_dict(),
            "packing": self.packing,
            "data_spec": (
                self.data_spec.model_dump(mode="json") if self.data_spec else None
            ),
            "batches": self._batches,
            "samples": self._samples,
            "pending": _encoded_state(self._pending) if self._pending else None,
            "mixture": self.mixture.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "state_version": BATCH_STREAM_STATE_VERSION,
            "sample_objective": self.sample_objective,
            "stage_id": self.stage_id,
            "budget": self.budget.to_dict(),
            "packing": self.packing,
            "data_spec": (
                self.data_spec.model_dump(mode="json") if self.data_spec else None
            ),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise BatchPlanningError(f"batch stream {key} mismatch")
        mixture = state.get("mixture")
        if not isinstance(mixture, Mapping):
            raise BatchPlanningError("batch stream mixture state is missing")
        self.mixture.load_state_dict(mixture)
        self._batches = _nonnegative_int(state.get("batches"), "batches")
        self._samples = _nonnegative_int(state.get("samples"), "samples")
        pending = state.get("pending")
        if pending is not None and not isinstance(pending, Mapping):
            raise BatchPlanningError("batch stream pending state must be a mapping")
        self._pending = _encoded_from_state(pending) if pending is not None else None

    @property
    def counters(self) -> Mapping[str, int]:
        return {"batches": self._batches, "samples": self._samples}


def _encoded_state(value: EncodedSample) -> dict[str, Any]:
    return {
        "sample_id": value.sample_id,
        "model_inputs": dict(value.model_inputs),
        "cost": value.cost.to_dict(),
        "source_spans": [asdict(span) for span in value.source_spans],
        "trace": dict(value.trace),
    }


def _encoded_from_state(value: Mapping[str, Any]) -> EncodedSample:
    return EncodedSample(
        sample_id=str(value["sample_id"]),
        model_inputs=value["model_inputs"],
        cost=CostVector(**value["cost"]),
        source_spans=tuple(SourceSpan(**item) for item in value["source_spans"]),
        trace=value.get("trace", {}),
    )


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise BatchPlanningError(f"batch stream {name} is invalid")
    return value


def validate_sample_against_data(
    sample: Any, data: DataSpec, sample_objective: str
) -> None:
    if sample.objective != sample_objective:
        raise ValueError(
            f"sample {sample.id!r} objective {sample.objective!r} does not match "
            f"stage objective {sample_objective!r}"
        )
    modalities = {asset.modality for asset in sample.assets}
    unsupported_modalities = modalities - data.modalities
    if unsupported_modalities:
        raise ValueError(
            f"sample {sample.id!r} has undeclared modalities: "
            f"{sorted(unsupported_modalities)}"
        )
    block_types = {
        block.type for message in sample.messages for block in message.content
    }
    unsupported_blocks = block_types - data.content_blocks
    if unsupported_blocks:
        raise ValueError(
            f"sample {sample.id!r} has undeclared content blocks: "
            f"{sorted(unsupported_blocks)}"
        )
    if len(sample.assets) > data.max_media_per_sample:
        raise ValueError(
            f"sample {sample.id!r} contains {len(sample.assets)} media assets; "
            f"stage limit is {data.max_media_per_sample}"
        )
