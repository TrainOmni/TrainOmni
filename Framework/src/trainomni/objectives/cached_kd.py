"""Native offline dense-logit knowledge-distillation objective.

The teacher is represented only by an immutable, preflighted BF16 logit cache.
No teacher model is constructed or accepted by this objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trainomni.config import canonical_fingerprint
from trainomni.models import ModelBatch

from .protocol import (
    LossOutput,
    LossTerm,
    ObjectiveManifest,
    ObjectiveRequirements,
    ObjectiveSetup,
)

CACHE_SCHEMA_VERSION = "trainomni.offline-dense-logit-cache.v1"
KD_CONFIG_KEY = "offline_dense_logit_kd"
_SHA256 = r"^[0-9a-f]{64}$"
_KD_BATCH_FIELDS = frozenset(
    {
        "kd_teacher_logits",
        "kd_assistant_positions",
        "kd_position_mask",
        "kd_target_token_ids",
        "kd_cache_identity",
    }
)


class CachedLogitKDError(ValueError):
    """Fail-closed cache, alignment, identity or loss-contract violation."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointIdentity(_StrictModel):
    artifact: str = Field(min_length=1)
    path: str = Field(min_length=1)
    state_file: str = "state.pkl"
    state_sha256: str = Field(pattern=_SHA256)
    manifest_file: str = "manifest.json"
    manifest_sha256: str = Field(pattern=_SHA256)
    run_fingerprint: str = Field(pattern=_SHA256)
    load_semantics: Literal["model_only"] = "model_only"


class AssetFileIdentity(_StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)
    size_bytes: int = Field(gt=0)


class AssetSetIdentity(_StrictModel):
    identity_sha256: str = Field(pattern=_SHA256)
    files: tuple[AssetFileIdentity, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> AssetSetIdentity:
        expected = asset_set_digest(self.files, self.metadata)
        if self.identity_sha256 != expected:
            raise ValueError(
                "asset identity_sha256 does not match canonical files/metadata"
            )
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("asset identity contains duplicate file paths")
        return self


class ModelIdentity(_StrictModel):
    plugin_id: str = Field(min_length=1)
    plugin_version: str = Field(min_length=1)
    assets: AssetSetIdentity


class DataIdentity(_StrictModel):
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_SHA256)
    split_fingerprint: str = Field(pattern=_SHA256)
    identity_sha256: str = Field(pattern=_SHA256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LossPositionIdentity(_StrictModel):
    identity_sha256: str = Field(pattern=_SHA256)
    total_positions: int = Field(gt=0)
    position_semantics: Literal["target_position_select_previous_logit"] = (
        "target_position_select_previous_logit"
    )


class LogitTensorIdentity(_StrictModel):
    file: str = Field(min_length=1)
    shape: tuple[int, int]
    dtype: Literal["bfloat16"] = "bfloat16"
    sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_shape(self) -> LogitTensorIdentity:
        if len(self.shape) != 2 or any(item <= 0 for item in self.shape):
            raise ValueError("teacher logit tensor shape must be [positions, vocab]")
        path = Path(self.file)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("teacher logit tensor file must stay under cache root")
        return self


class CacheSampleIdentity(_StrictModel):
    sample_id: str = Field(min_length=1)
    source_index: int = Field(ge=0)
    canonical_sample_sha256: str = Field(pattern=_SHA256)
    input_ids_sha256: str = Field(pattern=_SHA256)
    labels_sha256: str = Field(pattern=_SHA256)
    assistant_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    teacher_logits: LogitTensorIdentity

    @model_validator(mode="after")
    def validate_alignment(self) -> CacheSampleIdentity:
        if not self.assistant_positions:
            raise ValueError("cache sample must contain assistant loss positions")
        if tuple(sorted(set(self.assistant_positions))) != self.assistant_positions:
            raise ValueError(
                "assistant positions must be unique and strictly increasing"
            )
        if self.assistant_positions[0] <= 0:
            raise ValueError(
                "assistant target positions must be positive because logits use p-1"
            )
        if len(self.target_token_ids) != len(self.assistant_positions):
            raise ValueError("target token count differs from assistant positions")
        if self.teacher_logits.shape[0] != len(self.assistant_positions):
            raise ValueError("teacher logit rows differ from assistant positions")
        return self


class LogitCacheIdentity(_StrictModel):
    dtype: Literal["bfloat16"] = "bfloat16"
    vocab_size: int = Field(gt=1)
    total_positions: int = Field(gt=0)
    total_bytes: int = Field(gt=0)
    content_sha256: str = Field(pattern=_SHA256)


class OfflineDenseLogitCacheManifest(_StrictModel):
    schema_version: Literal["trainomni.offline-dense-logit-cache.v1"] = (
        CACHE_SCHEMA_VERSION
    )
    cache_id: str = Field(min_length=1)
    producer_code_revision: str = Field(min_length=1)
    teacher: CheckpointIdentity
    student: CheckpointIdentity
    model: ModelIdentity
    tokenizer: AssetSetIdentity
    processor: AssetSetIdentity
    data: DataIdentity
    loss_positions: LossPositionIdentity
    logits: LogitCacheIdentity
    samples: tuple[CacheSampleIdentity, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> OfflineDenseLogitCacheManifest:
        if not self.samples:
            raise ValueError("cache manifest samples must not be empty")
        sample_ids = [item.sample_id for item in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("cache manifest sample IDs must be unique")
        files = [item.teacher_logits.file for item in self.samples]
        if len(files) != len(set(files)):
            raise ValueError("cache manifest tensor files must be unique")
        positions = sum(len(item.assistant_positions) for item in self.samples)
        if positions != self.logits.total_positions:
            raise ValueError(
                "cache total_positions differs from sample position count"
            )
        if positions != self.loss_positions.total_positions:
            raise ValueError(
                "loss-position total differs from cache position count"
            )
        expected_bytes = positions * self.logits.vocab_size * 2
        if expected_bytes != self.logits.total_bytes:
            raise ValueError("cache total_bytes is inconsistent with BF16 shape")
        for sample in self.samples:
            if sample.teacher_logits.shape[1] != self.logits.vocab_size:
                raise ValueError(
                    f"cache sample {sample.sample_id!r} vocab size mismatch"
                )
            if any(
                token < 0 or token >= self.logits.vocab_size
                for token in sample.target_token_ids
            ):
                raise ValueError(
                    f"cache sample {sample.sample_id!r} target token is out of range"
                )
        actual_loss_positions = loss_position_digest(self.samples)
        if self.loss_positions.identity_sha256 != actual_loss_positions:
            raise ValueError("loss-position identity digest mismatch")
        actual_data = data_identity_digest(self.data, self.samples)
        if self.data.identity_sha256 != actual_data:
            raise ValueError("data identity digest mismatch")
        return self


class OfflineDenseLogitKDConfig(_StrictModel):
    cache_manifest: str = Field(min_length=1)
    cache_manifest_sha256: str = Field(pattern=_SHA256)
    cache_content_sha256: str = Field(pattern=_SHA256)
    teacher_state_sha256: str = Field(pattern=_SHA256)
    teacher_manifest_sha256: str = Field(pattern=_SHA256)
    teacher_run_fingerprint: str = Field(pattern=_SHA256)
    student_state_sha256: str = Field(pattern=_SHA256)
    student_manifest_sha256: str = Field(pattern=_SHA256)
    student_run_fingerprint: str = Field(pattern=_SHA256)
    model_identity_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    processor_sha256: str = Field(pattern=_SHA256)
    data_sha256: str = Field(pattern=_SHA256)
    loss_positions_sha256: str = Field(pattern=_SHA256)
    temperature: float = Field(default=2.0, gt=0)
    ce_weight: float = Field(default=0.5, ge=0)
    kd_weight: float = Field(default=0.5, ge=0)
    vocab_size: int = Field(gt=1)
    cache_dtype: Literal["bfloat16"] = "bfloat16"
    max_cache_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_weights(self) -> OfflineDenseLogitKDConfig:
        if self.ce_weight + self.kd_weight <= 0:
            raise ValueError("CE and KD weights cannot both be zero")
        return self


@dataclass(frozen=True, slots=True)
class OfflineDenseLogitCache:
    manifest_path: Path
    manifest_sha256: str
    manifest: OfflineDenseLogitCacheManifest
    samples: Mapping[str, CacheSampleIdentity]
    metadata: Mapping[str, Any]

    def load_sample(self, torch: Any, sample_id: str) -> tuple[Any, CacheSampleIdentity]:
        try:
            sample = self.samples[sample_id]
        except KeyError as exc:
            raise CachedLogitKDError(
                f"cache has no teacher logits for sample {sample_id!r}"
            ) from exc
        tensor_path = (self.manifest_path.parent / sample.teacher_logits.file).resolve()
        payload = tensor_path.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != sample.teacher_logits.sha256:
            raise CachedLogitKDError(
                f"teacher logit tensor digest changed for sample {sample_id!r}"
            )
        expected_bytes = math.prod(sample.teacher_logits.shape) * 2
        if len(payload) != expected_bytes:
            raise CachedLogitKDError(
                f"teacher logit tensor byte size changed for sample {sample_id!r}"
            )
        tensor = torch.frombuffer(
            bytearray(payload), dtype=torch.bfloat16
        ).reshape(sample.teacher_logits.shape)
        return tensor, sample


class OfflineDenseLogitKDObjective:
    manifest = ObjectiveManifest(
        objective_id="offline-dense-logit-kd",
        objective_version="1.0.0",
        requirements=ObjectiveRequirements(
            sample_objectives=frozenset({"sft"}),
            requires_teacher_model=False,
        ),
        supported_engines=frozenset({"torch"}),
    )

    def setup(self, context: Any) -> ObjectiveSetup:
        runtime = open_offline_dense_logit_cache(context)
        return ObjectiveSetup(runtime=runtime, metadata=runtime.metadata)

    def prepare(self, batch: ModelBatch, context: Any) -> ModelBatch:
        setup = getattr(context, "objective_setup", None)
        runtime = getattr(setup, "runtime", None)
        if not isinstance(runtime, OfflineDenseLogitCache):
            raise CachedLogitKDError(
                "offline dense-logit KD objective was not preflighted"
            )
        if len(batch.sample_ids) != 1:
            raise CachedLogitKDError(
                "offline dense-logit KD v1 requires batch size 1"
            )
        conflicts = _KD_BATCH_FIELDS.intersection(batch.model_inputs)
        if conflicts:
            raise CachedLogitKDError(
                f"model plugin must not inject reserved KD fields: {sorted(conflicts)}"
            )
        sample_id = batch.sample_ids[0]
        teacher_logits, sample = runtime.load_sample(context.torch, sample_id)
        inputs = dict(batch.model_inputs)
        if "input_ids" not in inputs or "labels" not in inputs:
            raise CachedLogitKDError(
                "offline dense-logit KD requires input_ids and labels"
            )
        input_ids = _single_batch_row(inputs["input_ids"], "input_ids")
        labels = _single_batch_row(inputs["labels"], "labels")
        if integer_tensor_digest(input_ids) != sample.input_ids_sha256:
            raise CachedLogitKDError(
                f"input_ids digest mismatch for sample {sample_id!r}"
            )
        if integer_tensor_digest(labels) != sample.labels_sha256:
            raise CachedLogitKDError(
                f"labels digest mismatch for sample {sample_id!r}"
            )
        label_values = _flatten_ints(labels)
        supervised_positions = tuple(
            index for index, value in enumerate(label_values) if value != -100
        )
        if supervised_positions != sample.assistant_positions:
            raise CachedLogitKDError(
                f"assistant target positions differ from labels loss mask for "
                f"sample {sample_id!r}"
            )
        supervised_targets = tuple(
            label_values[position] for position in supervised_positions
        )
        if supervised_targets != sample.target_token_ids:
            raise CachedLogitKDError(
                f"target token alignment mismatch for sample {sample_id!r}"
            )
        torch = context.torch
        positions = torch.tensor(
            sample.assistant_positions, dtype=torch.long
        ).unsqueeze(0)
        targets = torch.tensor(
            sample.target_token_ids, dtype=torch.long
        ).unsqueeze(0)
        mask = torch.ones_like(positions, dtype=torch.bool)
        inputs.update(
            {
                "kd_teacher_logits": teacher_logits.unsqueeze(0),
                "kd_assistant_positions": positions,
                "kd_position_mask": mask,
                "kd_target_token_ids": targets,
                "kd_cache_identity": runtime.manifest.logits.content_sha256,
            }
        )
        trace = dict(batch.trace)
        trace["offline_dense_logit_kd"] = {
            "cache_id": runtime.manifest.cache_id,
            "cache_content_sha256": runtime.manifest.logits.content_sha256,
            "sample_id": sample_id,
            "teacher_logits_sha256": sample.teacher_logits.sha256,
            "assistant_positions": list(sample.assistant_positions),
            "loss": dict(runtime.metadata["loss"]),
        }
        return ModelBatch(
            sample_ids=batch.sample_ids,
            model_inputs=inputs,
            plan=batch.plan,
            trace=trace,
        )

    def compute(self, models: Any, batch: ModelBatch) -> LossOutput:
        model = models.get("model") if isinstance(models, Mapping) else models
        if model is None or not callable(model):
            raise CachedLogitKDError(
                "offline dense-logit KD requires one callable student model"
            )
        if isinstance(models, Mapping) and "teacher" in models:
            raise CachedLogitKDError(
                "offline dense-logit KD forbids a live teacher model"
            )
        inputs = dict(batch.model_inputs)
        try:
            teacher_logits = inputs.pop("kd_teacher_logits")
            positions = inputs.pop("kd_assistant_positions")
            position_mask = inputs.pop("kd_position_mask")
            targets = inputs.pop("kd_target_token_ids")
            cache_identity = inputs.pop("kd_cache_identity")
        except KeyError as exc:
            raise CachedLogitKDError(
                f"offline dense-logit KD batch is missing {exc.args[0]!r}"
            ) from exc
        output = model(**inputs)
        student_logits = getattr(output, "logits", None)
        if student_logits is None and isinstance(output, Mapping):
            student_logits = output.get("logits")
        if student_logits is None:
            raise CachedLogitKDError("student model output does not expose logits")
        setup = getattr(batch, "trace", {}).get("offline_dense_logit_kd", {})
        expected_identity = setup.get("cache_content_sha256")
        if not isinstance(cache_identity, str) or cache_identity != expected_identity:
            raise CachedLogitKDError("batch cache identity mismatch")
        _validate_kd_tensors(
            student_logits,
            teacher_logits,
            positions,
            position_mask,
            targets,
        )
        selected = _select_prediction_logits(student_logits, positions, position_mask)
        selected_teacher = teacher_logits[position_mask]
        selected_targets = targets[position_mask]
        temperature, ce_weight, kd_weight = _loss_config_from_trace(batch)
        torch = _torch_module(student_logits)
        with torch.autocast(device_type=student_logits.device.type, enabled=False):
            student_fp32 = selected.float()
            teacher_fp32 = selected_teacher.float()
            target_ids = selected_targets.long()
            token_ce = torch.nn.functional.cross_entropy(
                student_fp32, target_ids, reduction="mean"
            )
            teacher_log_probs = torch.nn.functional.log_softmax(
                teacher_fp32 / temperature, dim=-1
            )
            student_log_probs = torch.nn.functional.log_softmax(
                student_fp32 / temperature, dim=-1
            )
            teacher_probs = teacher_log_probs.exp()
            teacher_kl = (
                teacher_probs * (teacher_log_probs - student_log_probs)
            ).sum(dim=-1).mean() * (temperature * temperature)
            weighted_ce = token_ce * ce_weight
            weighted_teacher_kl = teacher_kl * kd_weight
            total = weighted_ce + weighted_teacher_kl
        values = {
            "token_ce": token_ce,
            "teacher_kl": teacher_kl,
            "weighted_ce": weighted_ce,
            "weighted_teacher_kl": weighted_teacher_kl,
            "total": total,
        }
        for name, value in values.items():
            if not bool(torch.isfinite(value).item()):
                raise CachedLogitKDError(f"KD loss term {name!r} is non-finite")
        denominator = int(position_mask.sum().item())
        return LossOutput(
            total=total,
            terms={
                name: LossTerm(value=value, denominator=denominator)
                for name, value in values.items()
            },
            metrics={
                **{
                    name: float(value.detach().item())
                    for name, value in values.items()
                },
                "loss": float(total.detach().item()),
            },
            counts={"loss_tokens": denominator},
        )


def open_offline_dense_logit_cache(context: Any) -> OfflineDenseLogitCache:
    if sys.byteorder != "little":
        raise CachedLogitKDError(
            "offline BF16 raw-logit cache v1 requires a little-endian host"
        )
    resolved = context.context.resolved
    raw_config = resolved.run.metadata.get(KD_CONFIG_KEY)
    if raw_config is None:
        raise CachedLogitKDError(
            f"run metadata must define {KD_CONFIG_KEY!r}"
        )
    try:
        config = OfflineDenseLogitKDConfig.model_validate(raw_config)
    except Exception as exc:
        raise CachedLogitKDError(f"invalid offline KD config: {exc}") from exc
    manifest_path = Path(config.cache_manifest)
    if not manifest_path.is_absolute():
        source = resolved.source
        base = source.parent if source is not None else Path.cwd()
        manifest_path = base / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise CachedLogitKDError(f"cache manifest is missing: {manifest_path}")
    observed_manifest = _file_sha256(manifest_path)
    if observed_manifest != config.cache_manifest_sha256:
        raise CachedLogitKDError("cache manifest SHA-256 mismatch")
    try:
        manifest = OfflineDenseLogitCacheManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise CachedLogitKDError(f"invalid cache manifest: {exc}") from exc
    _validate_expected_identities(config, manifest)
    _validate_plugin_identity(context, manifest)
    _validate_checkpoint_identity(manifest.teacher)
    _validate_checkpoint_identity(manifest.student)
    _validate_student_input(context, manifest.student)
    _validate_asset_set(manifest_path.parent, manifest.model.assets, "model")
    _validate_asset_set(manifest_path.parent, manifest.tokenizer, "tokenizer")
    _validate_asset_set(manifest_path.parent, manifest.processor, "processor")
    _validate_data_source(manifest.data)
    if manifest.logits.total_bytes > config.max_cache_bytes:
        raise CachedLogitKDError(
            "dense-logit cache exceeds configured max_cache_bytes: "
            f"observed={manifest.logits.total_bytes}, limit={config.max_cache_bytes}"
        )
    content = hashlib.sha256()
    observed_bytes = 0
    for sample in manifest.samples:
        tensor_path = (manifest_path.parent / sample.teacher_logits.file).resolve()
        if not tensor_path.is_relative_to(manifest_path.parent):
            raise CachedLogitKDError("cache tensor path escapes cache root")
        digest, size = _stream_file_identity(tensor_path, aggregate=content)
        expected_size = math.prod(sample.teacher_logits.shape) * 2
        if digest != sample.teacher_logits.sha256:
            raise CachedLogitKDError(
                f"teacher logit tensor SHA-256 mismatch for {sample.sample_id!r}"
            )
        if size != expected_size:
            raise CachedLogitKDError(
                f"teacher logit tensor size mismatch for {sample.sample_id!r}"
            )
        observed_bytes += size
    if observed_bytes != manifest.logits.total_bytes:
        raise CachedLogitKDError("cache tensor total byte size mismatch")
    if content.hexdigest() != manifest.logits.content_sha256:
        raise CachedLogitKDError("cache total content SHA-256 mismatch")
    metadata = {
        "schema_version": "trainomni.offline-dense-logit-kd-identity.v1",
        "objective_id": "offline-dense-logit-kd",
        "objective_version": "1.0.0",
        "cache_manifest_path": str(manifest_path),
        "cache_manifest_sha256": observed_manifest,
        "cache_content_sha256": manifest.logits.content_sha256,
        "cache_id": manifest.cache_id,
        "producer_code_revision": manifest.producer_code_revision,
        "sample_count": len(manifest.samples),
        "teacher": manifest.teacher.model_dump(mode="json"),
        "student": manifest.student.model_dump(mode="json"),
        "model": manifest.model.model_dump(mode="json"),
        "tokenizer": manifest.tokenizer.model_dump(mode="json"),
        "processor": manifest.processor.model_dump(mode="json"),
        "data": manifest.data.model_dump(mode="json"),
        "loss_positions": manifest.loss_positions.model_dump(mode="json"),
        "logits": manifest.logits.model_dump(mode="json"),
        "loss": {
            "temperature": config.temperature,
            "ce_weight": config.ce_weight,
            "kd_weight": config.kd_weight,
            "kl_direction": "teacher||student",
            "temperature_squared_inside_kd": True,
            "compute_dtype": "float32",
            "reduction": "mean_over_assistant_loss_positions",
        },
    }
    return OfflineDenseLogitCache(
        manifest_path=manifest_path,
        manifest_sha256=observed_manifest,
        manifest=manifest,
        samples=MappingProxyType({item.sample_id: item for item in manifest.samples}),
        metadata=MappingProxyType(metadata),
    )


def asset_set_digest(
    files: Sequence[AssetFileIdentity], metadata: Mapping[str, Any]
) -> str:
    return canonical_fingerprint(
        {
            "files": [item.model_dump(mode="json") for item in files],
            "metadata": dict(metadata),
        }
    )


def data_identity_digest(
    data: DataIdentity, samples: Sequence[CacheSampleIdentity]
) -> str:
    return canonical_fingerprint(
        {
            "source_path": data.source_path,
            "source_sha256": data.source_sha256,
            "split_fingerprint": data.split_fingerprint,
            "metadata": data.metadata,
            "samples": [
                {
                    "sample_id": item.sample_id,
                    "source_index": item.source_index,
                    "canonical_sample_sha256": item.canonical_sample_sha256,
                    "input_ids_sha256": item.input_ids_sha256,
                    "labels_sha256": item.labels_sha256,
                }
                for item in samples
            ],
        }
    )


def loss_position_digest(samples: Sequence[CacheSampleIdentity]) -> str:
    return canonical_fingerprint(
        [
            {
                "sample_id": item.sample_id,
                "assistant_positions": list(item.assistant_positions),
                "target_token_ids": list(item.target_token_ids),
            }
            for item in samples
        ]
    )


def integer_tensor_digest(value: Any) -> str:
    shape = _shape(value)
    values = _flatten_ints(value)
    return canonical_fingerprint(
        {"dtype": "int64", "shape": shape, "values": values}
    )


def _validate_expected_identities(
    config: OfflineDenseLogitKDConfig,
    manifest: OfflineDenseLogitCacheManifest,
) -> None:
    checks = {
        "cache content": (config.cache_content_sha256, manifest.logits.content_sha256),
        "teacher state": (config.teacher_state_sha256, manifest.teacher.state_sha256),
        "teacher manifest": (
            config.teacher_manifest_sha256,
            manifest.teacher.manifest_sha256,
        ),
        "teacher run fingerprint": (
            config.teacher_run_fingerprint,
            manifest.teacher.run_fingerprint,
        ),
        "student state": (config.student_state_sha256, manifest.student.state_sha256),
        "student manifest": (
            config.student_manifest_sha256,
            manifest.student.manifest_sha256,
        ),
        "student run fingerprint": (
            config.student_run_fingerprint,
            manifest.student.run_fingerprint,
        ),
        "model": (config.model_identity_sha256, manifest.model.assets.identity_sha256),
        "tokenizer": (config.tokenizer_sha256, manifest.tokenizer.identity_sha256),
        "processor": (config.processor_sha256, manifest.processor.identity_sha256),
        "data": (config.data_sha256, manifest.data.identity_sha256),
        "loss positions": (
            config.loss_positions_sha256,
            manifest.loss_positions.identity_sha256,
        ),
    }
    mismatched = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if mismatched:
        raise CachedLogitKDError(
            f"offline KD expected identity mismatch: {sorted(mismatched)}"
        )
    if config.vocab_size != manifest.logits.vocab_size:
        raise CachedLogitKDError("offline KD vocab size mismatch")
    if config.cache_dtype != manifest.logits.dtype:
        raise CachedLogitKDError("offline KD cache dtype mismatch")


def _validate_plugin_identity(
    context: Any, manifest: OfflineDenseLogitCacheManifest
) -> None:
    plugin = context.context.resolved.plugin_manifest
    if (
        plugin.plugin_id != manifest.model.plugin_id
        or plugin.plugin_version != manifest.model.plugin_version
    ):
        raise CachedLogitKDError(
            "cache model plugin identity differs from consumer plugin"
        )


def _validate_checkpoint_identity(identity: CheckpointIdentity) -> None:
    root = Path(identity.path).resolve()
    if not root.is_dir():
        raise CachedLogitKDError(
            f"checkpoint identity path is missing: {root}"
        )
    state_path = (root / identity.state_file).resolve()
    manifest_path = (root / identity.manifest_file).resolve()
    if not state_path.is_relative_to(root) or not manifest_path.is_relative_to(root):
        raise CachedLogitKDError("checkpoint identity file escapes checkpoint root")
    if _file_sha256(state_path) != identity.state_sha256:
        raise CachedLogitKDError(
            f"checkpoint state SHA-256 mismatch: {state_path}"
        )
    if _file_sha256(manifest_path) != identity.manifest_sha256:
        raise CachedLogitKDError(
            f"checkpoint manifest SHA-256 mismatch: {manifest_path}"
        )
    try:
        checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CachedLogitKDError(
            f"checkpoint manifest is unreadable: {manifest_path}"
        ) from exc
    observed_run = checkpoint_manifest.get("metadata", {}).get("run_fingerprint")
    if observed_run != identity.run_fingerprint:
        raise CachedLogitKDError(
            f"checkpoint run fingerprint mismatch: {manifest_path}"
        )


def _validate_student_input(context: Any, student: CheckpointIdentity) -> None:
    artifacts = context.context.input_artifacts
    reference = artifacts.get("model") or artifacts.get("checkpoint")
    if reference is None or reference.uri is None:
        raise CachedLogitKDError(
            "offline dense-logit KD requires a physical student model input"
        )
    if str(reference) != student.artifact:
        raise CachedLogitKDError("student artifact reference mismatch")
    if Path(reference.uri).resolve() != Path(student.path).resolve():
        raise CachedLogitKDError("student physical checkpoint path mismatch")


def _validate_asset_set(root: Path, asset: AssetSetIdentity, name: str) -> None:
    for item in asset.files:
        path = Path(item.path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        digest, size = _stream_file_identity(path)
        if digest != item.sha256 or size != item.size_bytes:
            raise CachedLogitKDError(
                f"{name} asset file identity mismatch: {path}"
            )


def _validate_data_source(data: DataIdentity) -> None:
    path = Path(data.source_path).resolve()
    if _file_sha256(path) != data.source_sha256:
        raise CachedLogitKDError(f"data source SHA-256 mismatch: {path}")


def _single_batch_row(value: Any, name: str) -> Any:
    shape = _shape(value)
    if len(shape) == 2:
        if shape[0] != 1:
            raise CachedLogitKDError(f"{name} batch dimension must be 1")
        return value[0]
    if len(shape) == 1:
        return value
    raise CachedLogitKDError(f"{name} must be rank 1 or [1, sequence]")


def _validate_kd_tensors(
    student_logits: Any,
    teacher_logits: Any,
    positions: Any,
    position_mask: Any,
    targets: Any,
) -> None:
    torch = _torch_module(student_logits)
    if student_logits.ndim != 3 or student_logits.shape[0] != 1:
        raise CachedLogitKDError("student logits must have shape [1, sequence, vocab]")
    if teacher_logits.ndim != 3 or teacher_logits.shape[0] != 1:
        raise CachedLogitKDError("teacher logits must have shape [1, positions, vocab]")
    if teacher_logits.dtype != torch.bfloat16:
        raise CachedLogitKDError("teacher logits must remain raw BF16")
    if teacher_logits.requires_grad:
        raise CachedLogitKDError("teacher logits must not require gradients")
    if positions.dtype != torch.long or targets.dtype != torch.long:
        raise CachedLogitKDError("KD positions and targets must be int64")
    if position_mask.dtype != torch.bool:
        raise CachedLogitKDError("KD position mask must be bool")
    if positions.shape != targets.shape or positions.shape != position_mask.shape:
        raise CachedLogitKDError("KD position/target/mask shapes differ")
    if positions.shape[:1] != (1,) or positions.shape[1] <= 0:
        raise CachedLogitKDError("KD positions must have shape [1, positive]")
    if teacher_logits.shape[:2] != positions.shape:
        raise CachedLogitKDError("teacher logit rows differ from KD positions")
    if teacher_logits.shape[-1] != student_logits.shape[-1]:
        raise CachedLogitKDError("teacher/student vocab size mismatch")
    if not bool(position_mask.all().item()):
        raise CachedLogitKDError("KD v1 batch-size-1 position mask must be all true")
    if int(positions.min().item()) <= 0:
        raise CachedLogitKDError("assistant target positions must be positive")
    if int(positions.max().item()) >= student_logits.shape[1]:
        raise CachedLogitKDError("assistant target position exceeds student sequence")
    if int(targets.min().item()) < 0 or int(targets.max().item()) >= student_logits.shape[-1]:
        raise CachedLogitKDError("KD target token is outside student vocab")


def _select_prediction_logits(
    student_logits: Any, positions: Any, mask: Any
) -> Any:
    prediction_positions = positions - 1
    batch_indices = (
        _torch_module(student_logits)
        .arange(student_logits.shape[0], device=student_logits.device)
        .unsqueeze(1)
        .expand_as(prediction_positions)
    )
    selected = student_logits[batch_indices, prediction_positions]
    return selected[mask]


def _loss_config_from_trace(batch: ModelBatch) -> tuple[float, float, float]:
    value = batch.trace.get("offline_dense_logit_kd")
    if not isinstance(value, Mapping):
        raise CachedLogitKDError("KD batch trace identity is missing")
    config = value.get("loss")
    if not isinstance(config, Mapping):
        raise CachedLogitKDError("KD batch trace loss config is missing")
    return (
        float(config["temperature"]),
        float(config["ce_weight"]),
        float(config["kd_weight"]),
    )


def _torch_module(value: Any) -> Any:
    module = type(value).__module__.split(".", 1)[0]
    if module != "torch":
        raise CachedLogitKDError("offline dense-logit KD requires torch tensors")
    import torch

    return torch


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(item) for item in shape]
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        child = _shape(value[0])
        if any(_shape(item) != child for item in value):
            raise CachedLogitKDError("ragged integer tensor identity is unsupported")
        return [len(value), *child]
    return []


def _flatten_ints(value: Any) -> list[int]:
    detached = value.detach().cpu() if hasattr(value, "detach") else value
    if hasattr(detached, "reshape") and hasattr(detached, "tolist"):
        values = detached.reshape(-1).tolist()
        return [int(item) for item in values]
    if isinstance(detached, (list, tuple)):
        result: list[int] = []
        for item in detached:
            result.extend(_flatten_ints(item))
        return result
    return [int(detached)]


def _file_sha256(path: Path) -> str:
    return _stream_file_identity(path)[0]


def _stream_file_identity(
    path: Path, *, aggregate: Any | None = None
) -> tuple[str, int]:
    if not path.is_file():
        raise CachedLogitKDError(f"identity file is missing: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            if aggregate is not None:
                aggregate.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size
