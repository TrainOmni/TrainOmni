"""Model-neutral cost vectors and immutable batch plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CostVector:
    text_tokens: int = 0
    vision_tokens: int = 0
    pixels: int = 0
    frames: int = 0
    audio_seconds: float = 0.0
    model_units: float = 0.0

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.text_tokens,
                self.vision_tokens,
                self.pixels,
                self.frames,
                self.audio_seconds,
                self.model_units,
            )
        ):
            raise ValueError("cost values must be non-negative")

    def __add__(self, other: CostVector) -> CostVector:
        return CostVector(
            text_tokens=self.text_tokens + other.text_tokens,
            vision_tokens=self.vision_tokens + other.vision_tokens,
            pixels=self.pixels + other.pixels,
            frames=self.frames + other.frames,
            audio_seconds=self.audio_seconds + other.audio_seconds,
            model_units=self.model_units + other.model_units,
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "text_tokens": self.text_tokens,
            "vision_tokens": self.vision_tokens,
            "pixels": self.pixels,
            "frames": self.frames,
            "audio_seconds": self.audio_seconds,
            "model_units": self.model_units,
        }


@dataclass(frozen=True, slots=True)
class BatchBudget:
    max_samples: int | None = None
    max_text_tokens: int | None = None
    max_vision_tokens: int | None = None
    max_pixels: int | None = None
    max_frames: int | None = None
    max_audio_seconds: float | None = None
    max_model_units: float | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")

    def exceeded_by(self, cost: CostVector, sample_count: int) -> tuple[str, ...]:
        exceeded = []
        pairs = (
            ("samples", sample_count, self.max_samples),
            ("text_tokens", cost.text_tokens, self.max_text_tokens),
            ("vision_tokens", cost.vision_tokens, self.max_vision_tokens),
            ("pixels", cost.pixels, self.max_pixels),
            ("frames", cost.frames, self.max_frames),
            ("audio_seconds", cost.audio_seconds, self.max_audio_seconds),
            ("model_units", cost.model_units, self.max_model_units),
        )
        for name, actual, maximum in pairs:
            if maximum is not None and actual > maximum:
                exceeded.append(name)
        return tuple(exceeded)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BatchItem:
    sample_id: str
    segment_id: int
    cost: CostVector


@dataclass(frozen=True, slots=True)
class BatchPlan:
    items: tuple[BatchItem, ...]
    total_cost: CostVector
    budget: BatchBudget
    packing: bool = False

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("batch plan must contain at least one sample")
        if self.budget.exceeded_by(self.total_cost, len(self.items)):
            raise ValueError("batch plan exceeds its declared budget")

    def to_dict(self) -> dict[str, Any]:
        return {
            "packing": self.packing,
            "total_cost": self.total_cost.to_dict(),
            "budget": self.budget.to_dict(),
            "items": [
                {
                    "sample_id": item.sample_id,
                    "segment_id": item.segment_id,
                    "cost": item.cost.to_dict(),
                }
                for item in self.items
            ],
        }
