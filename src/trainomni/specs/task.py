"""Semantic task specification: what the model learns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from trainomni.core.errors import SpecError
from trainomni.core.module import ModuleId, ModuleKind, ModuleRef

from .digest import canonical_value, identity_digest

_RELOCATABLE_COLUMNAR_MODULES = frozenset(
    {
        "data_source:trainomni/parquet@1",
        "data_source:trainomni/arrow@1",
    }
)
_RELOCATABLE_TRANSFORMERS_MODULES = frozenset(
    {
        "model:trainomni/monolithic_transformers@1",
        "encoder:trainomni/transformers_vision@1",
        "encoder:trainomni/transformers_video@1",
        "language:trainomni/transformers_causal_lm@1",
        "model_io:trainomni/transformers@1",
    }
)
_PHYSICAL_COLUMNAR_PATHS = ("<physical-columnar-paths>",)
_PHYSICAL_TRANSFORMERS_ASSET = "<physical-transformers-asset>"


def _normalize_module_value(value: Mapping[str, Any], module_id: str) -> dict[str, Any]:
    normalized = dict(value)
    normalized["config"] = dict(value["config"])
    if module_id in _RELOCATABLE_COLUMNAR_MODULES:
        normalized["config"]["paths"] = list(_PHYSICAL_COLUMNAR_PATHS)
    elif (
        module_id in _RELOCATABLE_TRANSFORMERS_MODULES
        and normalized["config"].get("asset_manifest_sha256") is not None
    ):
        for field in ("model_name_or_path", "processor_name_or_path"):
            if field in normalized["config"]:
                normalized["config"][field] = _PHYSICAL_TRANSFORMERS_ASSET
    return normalized


def semantic_module_identity(reference: ModuleRef) -> Mapping[str, Any]:
    """Return a module identity with relocatable physical bindings normalized."""

    value = canonical_value(reference)
    return _normalize_module_value(value, str(reference.module_id))


def _semantic_task_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"module_id", "config"}:
            module = value["module_id"]
            if isinstance(module, Mapping):
                module_id = (
                    f"{module.get('kind')}:{module.get('namespace')}/"
                    f"{module.get('name')}@{module.get('version')}"
                )
                if module_id in (
                    _RELOCATABLE_COLUMNAR_MODULES
                    | _RELOCATABLE_TRANSFORMERS_MODULES
                ):
                    return _normalize_module_value(value, module_id)
        return {key: _semantic_task_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_semantic_task_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class LocalModuleSpec:
    module_id: ModuleId
    path: Path
    source_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, field: str) -> LocalModuleSpec:
        allowed = {"module", "path", "source_sha256"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"{field} contains unknown keys: {', '.join(unknown)}")
        raw_module = value.get("module")
        raw_path = value.get("path")
        digest = value.get("source_sha256")
        if not isinstance(raw_module, str):
            raise SpecError(f"{field}.module must be a string")
        if not isinstance(raw_path, str) or not raw_path:
            raise SpecError(f"{field}.path must be a non-empty relative path")
        normalized = PurePosixPath(raw_path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise SpecError(f"{field}.path must stay inside the task root")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SpecError(f"{field}.source_sha256 must be a lowercase SHA-256 digest")
        return cls(ModuleId.parse(raw_module), Path(*normalized.parts), digest)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecError(f"{field} must be a mapping")
    return value


def _ref(value: Any, *, field: str, kind: ModuleKind) -> ModuleRef:
    reference = ModuleRef.from_mapping(_mapping(value, field=field), field_name=field)
    if reference.module_id.kind is not kind:
        raise SpecError(
            f"{field} requires a {kind.value} module, got {reference.module_id.kind.value}"
        )
    return reference


@dataclass(frozen=True, slots=True)
class DataPipelineSpec:
    source: ModuleRef
    adapter: ModuleRef | None
    sources: tuple[tuple[str, ModuleRef], ...]
    transforms: tuple[ModuleRef, ...]
    model_io: ModuleRef
    supervision: ModuleRef
    packer: ModuleRef
    collator: ModuleRef
    drop_last: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DataPipelineSpec:
        allowed = {
            "source",
            "adapter",
            "sources",
            "transforms",
            "model_io",
            "supervision",
            "packer",
            "collator",
            "drop_last",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"data contains unknown keys: {', '.join(unknown)}")
        raw_transforms = value.get("transforms", [])
        if not isinstance(raw_transforms, Sequence) or isinstance(raw_transforms, str | bytes):
            raise SpecError("data.transforms must be a sequence")
        raw_sources = value.get("sources", {})
        if not isinstance(raw_sources, Mapping):
            raise SpecError("data.sources must be a mapping")
        sources = tuple(
            (
                str(name),
                _ref(
                    raw_source,
                    field=f"data.sources.{name}",
                    kind=ModuleKind.DATA_SOURCE,
                ),
            )
            for name, raw_source in sorted(
                raw_sources.items(), key=lambda item: str(item[0])
            )
        )
        if any(not name or name.startswith("__") for name, _ in sources):
            raise SpecError(
                "data source names must be non-empty and cannot start with '__'"
            )
        raw_adapter = value.get("adapter")
        drop_last = value.get("drop_last", False)
        if not isinstance(drop_last, bool):
            raise SpecError("data.drop_last must be a boolean")
        return cls(
            source=_ref(value.get("source"), field="data.source", kind=ModuleKind.DATA_SOURCE),
            adapter=(
                None
                if raw_adapter is None
                else _ref(
                    raw_adapter,
                    field="data.adapter",
                    kind=ModuleKind.DATA_ADAPTER,
                )
            ),
            sources=sources,
            transforms=tuple(
                _ref(item, field=f"data.transforms[{index}]", kind=ModuleKind.SAMPLE_TRANSFORM)
                for index, item in enumerate(raw_transforms)
            ),
            model_io=_ref(value.get("model_io"), field="data.model_io", kind=ModuleKind.MODEL_IO),
            supervision=_ref(
                value.get("supervision"),
                field="data.supervision",
                kind=ModuleKind.SUPERVISION,
            ),
            packer=_ref(value.get("packer"), field="data.packer", kind=ModuleKind.PACKER),
            collator=_ref(
                value.get("collator"), field="data.collator", kind=ModuleKind.COLLATOR
            ),
            drop_last=drop_last,
        )


@dataclass(frozen=True, slots=True)
class ModelAssemblySpec:
    implementation: ModuleRef
    components: tuple[tuple[str, ModuleRef], ...]
    attention_policy: ModuleRef | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelAssemblySpec:
        allowed = {"implementation", "components", "attention_policy"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"model contains unknown keys: {', '.join(unknown)}")
        raw_components = value.get("components", {})
        if not isinstance(raw_components, Mapping):
            raise SpecError("model.components must be a mapping")
        components = tuple(
            (
                str(name),
                ModuleRef.from_mapping(
                    _mapping(raw_ref, field=f"model.components.{name}"),
                    field_name=f"model.components.{name}",
                ),
            )
            for name, raw_ref in sorted(raw_components.items(), key=lambda item: str(item[0]))
        )
        invalid_names = [
            name for name, _ in components if not name or name.startswith("__")
        ]
        if invalid_names:
            raise SpecError(
                "model component names must be non-empty and cannot start with '__'"
            )
        raw_attention = value.get("attention_policy")
        attention = (
            None
            if raw_attention is None
            else _ref(
                raw_attention,
                field="model.attention_policy",
                kind=ModuleKind.ATTENTION_POLICY,
            )
        )
        return cls(
            implementation=_ref(
                value.get("implementation"),
                field="model.implementation",
                kind=ModuleKind.MODEL,
            ),
            components=components,
            attention_policy=attention,
        )


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    data: DataPipelineSpec
    evaluators: tuple[ModuleRef, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluationSpec:
        allowed = {"data", "evaluators"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"evaluation contains unknown keys: {', '.join(unknown)}")
        raw_evaluators = value.get("evaluators", [])
        if not isinstance(raw_evaluators, Sequence) or isinstance(
            raw_evaluators, str | bytes
        ):
            raise SpecError("evaluation.evaluators must be a sequence")
        if not raw_evaluators:
            raise SpecError("evaluation requires at least one evaluator")
        return cls(
            data=DataPipelineSpec.from_mapping(
                _mapping(value.get("data"), field="evaluation.data")
            ),
            evaluators=tuple(
                _ref(
                    item,
                    field=f"evaluation.evaluators[{index}]",
                    kind=ModuleKind.EVALUATOR,
                )
                for index, item in enumerate(raw_evaluators)
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    schema_version: int
    name: str
    data: DataPipelineSpec
    model: ModelAssemblySpec
    objective: ModuleRef
    parameters: ModuleRef
    evaluation: EvaluationSpec | None = None
    exporters: tuple[ModuleRef, ...] = ()
    local_modules: tuple[LocalModuleSpec, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskSpec:
        allowed = {
            "schema_version",
            "name",
            "data",
            "model",
            "objective",
            "parameters",
            "evaluation",
            "exporters",
            "local_modules",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SpecError(f"task contains unknown keys: {', '.join(unknown)}")
        version = value.get("schema_version")
        if version != 1:
            raise SpecError(f"unsupported task schema_version: {version!r}")
        name = value.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SpecError("task.name must be a non-empty string")
        raw_evaluation = value.get("evaluation")
        raw_exporters = value.get("exporters", [])
        if not isinstance(raw_exporters, Sequence) or isinstance(
            raw_exporters, str | bytes
        ):
            raise SpecError("task.exporters must be a sequence")
        raw_local_modules = value.get("local_modules", [])
        if not isinstance(raw_local_modules, Sequence) or isinstance(
            raw_local_modules, str | bytes
        ):
            raise SpecError("task.local_modules must be a sequence")
        local_modules = tuple(
            LocalModuleSpec.from_mapping(
                _mapping(item, field=f"local_modules[{index}]"),
                field=f"local_modules[{index}]",
            )
            for index, item in enumerate(raw_local_modules)
        )
        local_ids = [item.module_id for item in local_modules]
        if len(local_ids) != len(set(local_ids)):
            raise SpecError("task.local_modules contains duplicate module ids")
        return cls(
            schema_version=version,
            name=name.strip(),
            data=DataPipelineSpec.from_mapping(_mapping(value.get("data"), field="data")),
            model=ModelAssemblySpec.from_mapping(_mapping(value.get("model"), field="model")),
            objective=_ref(
                value.get("objective"), field="objective", kind=ModuleKind.OBJECTIVE
            ),
            parameters=_ref(
                value.get("parameters"),
                field="parameters",
                kind=ModuleKind.PARAMETER_POLICY,
            ),
            evaluation=(
                None
                if raw_evaluation is None
                else EvaluationSpec.from_mapping(
                    _mapping(raw_evaluation, field="evaluation")
                )
            ),
            exporters=tuple(
                _ref(item, field=f"exporters[{index}]", kind=ModuleKind.EXPORTER)
                for index, item in enumerate(raw_exporters)
            ),
            local_modules=local_modules,
        )

    @property
    def semantic_identity(self) -> Mapping[str, Any]:
        return _semantic_task_value(canonical_value(self))

    @property
    def digest(self) -> str:
        return identity_digest(self.semantic_identity)

    def module_refs(self) -> tuple[ModuleRef, ...]:
        references = [
            *(reference for _, reference in self.data.sources),
            self.data.source,
            *((self.data.adapter,) if self.data.adapter is not None else ()),
            *self.data.transforms,
            self.data.model_io,
            self.data.supervision,
            self.data.packer,
            self.data.collator,
            self.model.implementation,
            *(reference for _, reference in self.model.components),
            self.objective,
            self.parameters,
            *self.exporters,
        ]
        if self.evaluation is not None:
            references.extend(
                (
                    *(reference for _, reference in self.evaluation.data.sources),
                    self.evaluation.data.source,
                    *(
                        (self.evaluation.data.adapter,)
                        if self.evaluation.data.adapter is not None
                        else ()
                    ),
                    *self.evaluation.data.transforms,
                    self.evaluation.data.model_io,
                    self.evaluation.data.supervision,
                    self.evaluation.data.packer,
                    self.evaluation.data.collator,
                    *self.evaluation.evaluators,
                )
            )
        if self.model.attention_policy is not None:
            references.append(self.model.attention_policy)
        return tuple(references)
