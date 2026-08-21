"""Typed module identities, references, and descriptors."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeVar

from .capability import CapabilitySet
from .errors import SpecError

MODULE_API_VERSION = 1
_MODULE_ID = re.compile(
    r"^(?P<kind>[a-z][a-z0-9_]*)"
    r":(?P<namespace>[a-z][a-z0-9_.-]*)"
    r"/(?P<name>[a-z][a-z0-9_.-]*)"
    r"@(?P<version>[0-9A-Za-z_.+-]+)$"
)


class ModuleKind(StrEnum):
    DATA_SOURCE = "data_source"
    SAMPLE_TRANSFORM = "sample_transform"
    MODEL_IO = "model_io"
    SUPERVISION = "supervision"
    PACKER = "packer"
    COLLATOR = "collator"
    ENCODER = "encoder"
    CONNECTOR = "connector"
    FUSION = "fusion"
    LANGUAGE = "language"
    MODEL = "model"
    ATTENTION_POLICY = "attention_policy"
    OBJECTIVE = "objective"
    PARAMETER_POLICY = "parameter_policy"
    EVALUATOR = "evaluator"
    EXPORTER = "exporter"


@dataclass(frozen=True, slots=True, order=True)
class ModuleId:
    kind: ModuleKind
    namespace: str
    name: str
    version: str

    @classmethod
    def parse(cls, value: str) -> ModuleId:
        match = _MODULE_ID.fullmatch(value)
        if match is None:
            raise SpecError(
                "module id must be '<kind>:<namespace>/<name>@<version>', "
                f"got {value!r}"
            )
        try:
            kind = ModuleKind(match.group("kind"))
        except ValueError as exc:
            raise SpecError(f"unknown module kind in {value!r}") from exc
        return cls(
            kind=kind,
            namespace=match.group("namespace"),
            name=match.group("name"),
            version=match.group("version"),
        )

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.namespace}/{self.name}@{self.version}"


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Recursively freeze mapping/list configuration containers."""

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(inner) for key, inner in item.items()})
        if isinstance(item, list | tuple):
            return tuple(freeze(inner) for inner in item)
        return item

    return freeze(value or {})


@dataclass(frozen=True, slots=True)
class ModuleRef:
    module_id: ModuleId
    config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, field_name: str) -> ModuleRef:
        allowed = {"module", "config"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"{field_name} contains unknown keys: {', '.join(unknown)}")
        raw_id = value.get("module")
        if not isinstance(raw_id, str):
            raise SpecError(f"{field_name}.module must be a string")
        raw_config = value.get("config", {})
        if not isinstance(raw_config, Mapping):
            raise SpecError(f"{field_name}.config must be a mapping")
        return cls(ModuleId.parse(raw_id), freeze_mapping(raw_config))


ConfigT = TypeVar("ConfigT")


def parse_config(config_type: type[ConfigT], payload: Mapping[str, Any]) -> ConfigT:
    """Construct a dataclass config while rejecting misspelled keys."""

    if not hasattr(config_type, "__dataclass_fields__"):
        raise TypeError(f"module config type must be a dataclass: {config_type!r}")
    allowed = {item.name for item in fields(config_type) if item.init}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise SpecError(
            f"{config_type.__qualname__} contains unknown keys: {', '.join(unknown)}"
        )
    try:
        return config_type(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise SpecError(f"invalid {config_type.__qualname__}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ModuleDescriptor:
    module_id: ModuleId
    config_type: type[Any]
    factory: Callable[[Any, Any], Any]
    provides: CapabilitySet = field(default_factory=CapabilitySet)
    requires: CapabilitySet = field(default_factory=CapabilitySet)
    api_version: int = MODULE_API_VERSION

    def __post_init__(self) -> None:
        if self.api_version != MODULE_API_VERSION:
            raise SpecError(
                f"{self.module_id} uses module API {self.api_version}; "
                f"framework requires {MODULE_API_VERSION}"
            )

    def build(self, reference: ModuleRef, context: Any) -> Any:
        if reference.module_id != self.module_id:
            raise SpecError(
                f"descriptor {self.module_id} cannot build reference {reference.module_id}"
            )
        return self.factory(parse_config(self.config_type, reference.config), context)
