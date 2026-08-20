"""Typed result of building a model-family plugin.

The core never assumes a Hugging Face class hierarchy. It only needs a primary
model, optional named auxiliary models, and processor/tokenizer handles owned by
the model plugin.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trainomni.contracts import ArtifactRef


@dataclass(frozen=True, slots=True)
class ModelBuildContext(Mapping[str, Any]):
    """Mapping-compatible build request for old and new model plugins."""

    config: Mapping[str, Any]
    stage_id: str
    output_dir: Path
    input_artifacts: Mapping[str, ArtifactRef] = field(default_factory=dict)
    mode: str = "train"

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    model: Any
    processor: Any | None = None
    tokenizer: Any | None = None
    auxiliary_models: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model is None:
            raise ValueError("ModelBundle.model must not be None")
        names = set(self.auxiliary_models)
        if "model" in names or any(not str(name).strip() for name in names):
            raise ValueError(
                "auxiliary model names must be non-blank and cannot be 'model'"
            )
        object.__setattr__(
            self, "auxiliary_models", MappingProxyType(dict(self.auxiliary_models))
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def models(self) -> Mapping[str, Any]:
        return MappingProxyType({"model": self.model, **self.auxiliary_models})

    def parameter_names(self) -> tuple[str, ...]:
        named_parameters = getattr(self.model, "named_parameters", None)
        if not callable(named_parameters):
            return ()
        return tuple(name for name, _ in named_parameters())
