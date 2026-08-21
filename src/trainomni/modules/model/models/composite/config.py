"""Composite model component and modal-route binding."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeBranchConfig:
    name: str
    modality: str
    input_key: str
    encoder: str
    connector: str
    positions_key: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        named_values = {
            "name": self.name,
            "modality": self.modality,
            "input_key": self.input_key,
            "encoder": self.encoder,
            "connector": self.connector,
        }
        if any(not value or not _NAME.fullmatch(value) for value in named_values.values()):
            raise ValueError(
                "branch names, modality, input_key, encoder and connector must be "
                "non-empty identifier-like names"
            )
        if self.positions_key is not None and not _NAME.fullmatch(self.positions_key):
            raise ValueError("branch positions_key must be an identifier-like name")
        if self.positions_key == self.input_key:
            raise ValueError("branch input_key and positions_key must differ")
        if not isinstance(self.required, bool):
            raise TypeError("branch required must be a boolean")

    @classmethod
    def from_value(cls, value: Any) -> CompositeBranchConfig:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("composite branches must contain mappings")
        allowed = {
            "name",
            "modality",
            "input_key",
            "encoder",
            "connector",
            "positions_key",
            "required",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                "composite branch contains unknown keys: " + ", ".join(unknown)
            )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeModelConfig:
    branches: tuple[CompositeBranchConfig, ...] = (
        CompositeBranchConfig(
            name="vision",
            modality="image",
            input_key="pixel_values",
            encoder="encoder",
            connector="connector",
            positions_key="modal_positions",
        ),
    )
    fusion: str = "fusion"
    language: str = "language"

    def __post_init__(self) -> None:
        branches = tuple(CompositeBranchConfig.from_value(item) for item in self.branches)
        if not branches:
            raise ValueError("composite model requires at least one modal branch")
        names = [branch.name for branch in branches]
        input_keys = [branch.input_key for branch in branches]
        position_keys = [
            branch.positions_key for branch in branches if branch.positions_key is not None
        ]
        if len(names) != len(set(names)):
            raise ValueError("composite branch names must be unique")
        if len(input_keys) != len(set(input_keys)):
            raise ValueError("composite branch input keys must be unique")
        if len(position_keys) != len(set(position_keys)):
            raise ValueError("composite branch position keys must be unique")
        if not _NAME.fullmatch(self.fusion) or not _NAME.fullmatch(self.language):
            raise ValueError("fusion and language must be identifier-like component names")
        object.__setattr__(self, "branches", branches)
