"""Native offline-reference sigmoid DPO objective.

The reference policy is represented only by immutable, preflighted FP32
per-token log probabilities. Training performs two policy forwards and never
constructs or accepts a live reference model.
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

from .cached_kd import (
    AssetSetIdentity,
    CheckpointIdentity,
    ModelIdentity,
    integer_tensor_digest,
)
from .protocol import (
    LossOutput,
    LossTerm,
    ObjectiveManifest,
    ObjectiveRequirements,
    ObjectiveSetup,
)

DPO_CACHE_SCHEMA_VERSION = "trainomni.offline-reference-dpo-cache.v1"
DPO_PREFERENCE_SCHEMA_VERSION = "trainomni.offline-dpo-preference.v1"
DPO_CONFIG_KEY = "offline_reference_dpo"
_SHA256 = r"^[0-9a-f]{64}$"
_PAIR_INPUT_KEYS = frozenset({"chosen", "rejected", "dpo_pair_identity"})


class OfflineReferenceDPOError(ValueError):
    """Fail-closed preference, cache, alignment, identity or loss violation."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DPODataIdentity(_StrictModel):
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_SHA256)
    split_fingerprints: dict[str, str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    identity_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_identity(self) -> DPODataIdentity:
        if not self.split_fingerprints or any(
            not key.strip() or not _is_sha256(value)
            for key, value in self.split_fingerprints.items()
        ):
            raise ValueError("DPO data split fingerprints must be named SHA-256 values")
        if self.identity_sha256 != dpo_data_identity_digest(self):
            raise ValueError("DPO data identity digest mismatch")
        return self


class PreferencePairIdentity(_StrictModel):
    sample_id: str = Field(min_length=1)
    source_index: int = Field(ge=0)
    split: str = Field(min_length=1)
    order_index: int = Field(ge=0)
    canonical_pair_sha256: str = Field(pattern=_SHA256)
    common_prompt_sha256: str = Field(pattern=_SHA256)
    media_sha256: str = Field(pattern=_SHA256)
    chosen_canonical_sha256: str = Field(pattern=_SHA256)
    rejected_canonical_sha256: str = Field(pattern=_SHA256)
    construction_rule: str = Field(min_length=1)
    judge: str = Field(min_length=1)
    chosen_score: Literal[1.0] = 1.0
    rejected_score: Literal[0.0] = 0.0
    margin: Literal[1.0] = 1.0
    identity_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_pair(self) -> PreferencePairIdentity:
        if self.chosen_canonical_sha256 == self.rejected_canonical_sha256:
            raise ValueError("chosen and rejected canonical identities must differ")
        if self.identity_sha256 != preference_pair_digest(self):
            raise ValueError("preference pair identity digest mismatch")
        return self


class OfflineDPOPreferenceManifest(_StrictModel):
    schema_version: Literal["trainomni.offline-dpo-preference.v1"] = (
        DPO_PREFERENCE_SCHEMA_VERSION
    )
    preference_id: str = Field(min_length=1)
    producer_code_revision: str = Field(min_length=1)
    data: DPODataIdentity
    pairs: tuple[PreferencePairIdentity, ...]
    identity_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_manifest(self) -> OfflineDPOPreferenceManifest:
        if not self.pairs:
            raise ValueError("preference manifest must contain at least one pair")
        sample_ids = [pair.sample_id for pair in self.pairs]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("preference sample IDs must be unique")
        order = [pair.order_index for pair in self.pairs]
        if order != list(range(len(self.pairs))):
            raise ValueError("preference order_index must be contiguous manifest order")
        if self.identity_sha256 != preference_manifest_digest(self):
            raise ValueError("preference manifest identity digest mismatch")
        return self


class PreferenceManifestIdentity(_StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)
    identity_sha256: str = Field(pattern=_SHA256)


class ReferenceLogProbTensorIdentity(_StrictModel):
    file: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: Literal["float32"] = "float32"
    sha256: str = Field(pattern=_SHA256)
    sequence_logp: float

    @model_validator(mode="after")
    def validate_tensor(self) -> ReferenceLogProbTensorIdentity:
        if len(self.shape) != 1 or self.shape[0] <= 0:
            raise ValueError("reference log-prob tensor shape must be [positions]")
        if not math.isfinite(self.sequence_logp):
            raise ValueError("reference sequence log-prob must be finite")
        path = Path(self.file)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("reference tensor file must stay under cache root")
        return self


class DPOBranchIdentity(_StrictModel):
    canonical_sha256: str = Field(pattern=_SHA256)
    input_ids_sha256: str = Field(pattern=_SHA256)
    labels_sha256: str = Field(pattern=_SHA256)
    assistant_positions: tuple[int, ...]
    causal_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    reference_logps: ReferenceLogProbTensorIdentity

    @model_validator(mode="after")
    def validate_alignment(self) -> DPOBranchIdentity:
        if not self.assistant_positions:
            raise ValueError("DPO branch must contain assistant loss positions")
        if tuple(sorted(set(self.assistant_positions))) != self.assistant_positions:
            raise ValueError("assistant positions must be unique and increasing")
        if self.assistant_positions[0] <= 0:
            raise ValueError("assistant target positions must be positive")
        if self.causal_positions != tuple(
            position - 1 for position in self.assistant_positions
        ):
            raise ValueError("causal positions must equal assistant positions minus one")
        count = len(self.assistant_positions)
        if len(self.target_token_ids) != count:
            raise ValueError("DPO target-token count differs from positions")
        if self.reference_logps.shape != (count,):
            raise ValueError("reference log-prob rows differ from assistant positions")
        return self


class CachedDPOPairIdentity(_StrictModel):
    sample_id: str = Field(min_length=1)
    source_index: int = Field(ge=0)
    split: str = Field(min_length=1)
    order_index: int = Field(ge=0)
    preference_pair_sha256: str = Field(pattern=_SHA256)
    canonical_pair_sha256: str = Field(pattern=_SHA256)
    common_prompt_sha256: str = Field(pattern=_SHA256)
    media_sha256: str = Field(pattern=_SHA256)
    common_model_inputs_sha256: str = Field(pattern=_SHA256)
    chosen: DPOBranchIdentity
    rejected: DPOBranchIdentity

    @model_validator(mode="after")
    def validate_pair(self) -> CachedDPOPairIdentity:
        if self.chosen.canonical_sha256 == self.rejected.canonical_sha256:
            raise ValueError("cached chosen and rejected branches must differ")
        if len(self.chosen.assistant_positions) != len(
            self.rejected.assistant_positions
        ):
            raise ValueError("DPO v1 requires equal chosen/rejected target counts")
        if self.chosen.assistant_positions != self.rejected.assistant_positions:
            raise ValueError("DPO v1 requires equal chosen/rejected target positions")
        return self


class DPOCacheIdentity(_StrictModel):
    dtype: Literal["float32"] = "float32"
    vocab_size: int = Field(gt=1)
    pair_count: int = Field(gt=0)
    total_positions: int = Field(gt=0)
    total_bytes: int = Field(gt=0)
    content_sha256: str = Field(pattern=_SHA256)


class DPOAlgorithmIdentity(_StrictModel):
    beta: Literal[0.1] = 0.1
    loss_variant: Literal["sigmoid"] = "sigmoid"
    label_smoothing: Literal[0.0] = 0.0
    sequence_reduction: Literal["sum"] = "sum"
    pair_reduction: Literal["mean"] = "mean"
    compute_dtype: Literal["float32"] = "float32"
    reference_free: Literal[False] = False
    auxiliary_ce: Literal[False] = False
    causal_shift: Literal["target_position_select_previous_logit"] = (
        "target_position_select_previous_logit"
    )


class OfflineReferenceDPOCacheManifest(_StrictModel):
    schema_version: Literal["trainomni.offline-reference-dpo-cache.v1"] = (
        DPO_CACHE_SCHEMA_VERSION
    )
    cache_id: str = Field(min_length=1)
    producer_code_revision: str = Field(min_length=1)
    reference: CheckpointIdentity
    policy: CheckpointIdentity
    model: ModelIdentity
    tokenizer: AssetSetIdentity
    processor: AssetSetIdentity
    preference_manifest: PreferenceManifestIdentity
    data_identity_sha256: str = Field(pattern=_SHA256)
    pair_identity_sha256: str = Field(pattern=_SHA256)
    algorithm: DPOAlgorithmIdentity
    logps: DPOCacheIdentity
    pairs: tuple[CachedDPOPairIdentity, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> OfflineReferenceDPOCacheManifest:
        if not self.pairs:
            raise ValueError("DPO cache must contain at least one pair")
        sample_ids = [pair.sample_id for pair in self.pairs]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("DPO cache sample IDs must be unique")
        order = [pair.order_index for pair in self.pairs]
        if order != list(range(len(self.pairs))):
            raise ValueError("DPO cache order_index must be contiguous manifest order")
        total_positions = sum(
            len(pair.chosen.assistant_positions)
            + len(pair.rejected.assistant_positions)
            for pair in self.pairs
        )
        if self.logps.pair_count != len(self.pairs):
            raise ValueError("DPO cache pair_count mismatch")
        if self.logps.total_positions != total_positions:
            raise ValueError("DPO cache total_positions mismatch")
        if self.logps.total_bytes != total_positions * 4:
            raise ValueError("DPO cache total_bytes is inconsistent with FP32 tensors")
        pair_digest = canonical_fingerprint(
            [pair.preference_pair_sha256 for pair in self.pairs]
        )
        if self.pair_identity_sha256 != pair_digest:
            raise ValueError("DPO cache aggregate pair identity mismatch")
        return self


class OfflineReferenceDPOConfig(_StrictModel):
    cache_manifest: str = Field(min_length=1)
    cache_manifest_sha256: str = Field(pattern=_SHA256)
    cache_content_sha256: str = Field(pattern=_SHA256)
    preference_manifest_sha256: str = Field(pattern=_SHA256)
    preference_identity_sha256: str = Field(pattern=_SHA256)
    pair_identity_sha256: str = Field(pattern=_SHA256)
    data_identity_sha256: str = Field(pattern=_SHA256)
    reference_state_sha256: str = Field(pattern=_SHA256)
    reference_manifest_sha256: str = Field(pattern=_SHA256)
    reference_run_fingerprint: str = Field(pattern=_SHA256)
    policy_state_sha256: str = Field(pattern=_SHA256)
    policy_manifest_sha256: str = Field(pattern=_SHA256)
    policy_run_fingerprint: str = Field(pattern=_SHA256)
    model_identity_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    processor_sha256: str = Field(pattern=_SHA256)
    beta: Literal[0.1]
    loss_variant: Literal["sigmoid"]
    label_smoothing: Literal[0.0]
    sequence_reduction: Literal["sum"]
    pair_reduction: Literal["mean"]
    compute_dtype: Literal["float32"]
    reference_free: Literal[False]
    auxiliary_ce: Literal[False]
    expected_pair_count: int = Field(gt=0)
    expected_total_positions: int = Field(gt=0)
    vocab_size: int = Field(gt=1)
    max_cache_bytes: int = Field(gt=0)


class DPOBatchPairIdentity(_StrictModel):
    sample_id: str = Field(min_length=1)
    preference_pair_sha256: str = Field(pattern=_SHA256)
    canonical_pair_sha256: str = Field(pattern=_SHA256)
    common_prompt_sha256: str = Field(pattern=_SHA256)
    media_sha256: str = Field(pattern=_SHA256)
    common_model_inputs_sha256: str = Field(pattern=_SHA256)
    chosen_canonical_sha256: str = Field(pattern=_SHA256)
    rejected_canonical_sha256: str = Field(pattern=_SHA256)


@dataclass(frozen=True, slots=True)
class OfflineReferenceDPOCache:
    manifest_path: Path
    manifest_sha256: str
    manifest: OfflineReferenceDPOCacheManifest
    preference: OfflineDPOPreferenceManifest
    pairs: Mapping[str, CachedDPOPairIdentity]
    metadata: Mapping[str, Any]

    def load_branch(self, torch: Any, pair: CachedDPOPairIdentity, name: str) -> Any:
        branch = pair.chosen if name == "chosen" else pair.rejected
        tensor_path = (self.manifest_path.parent / branch.reference_logps.file).resolve()
        payload = tensor_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != branch.reference_logps.sha256:
            raise OfflineReferenceDPOError(
                f"reference {name} tensor digest changed for {pair.sample_id!r}"
            )
        expected_bytes = branch.reference_logps.shape[0] * 4
        if len(payload) != expected_bytes:
            raise OfflineReferenceDPOError(
                f"reference {name} tensor size changed for {pair.sample_id!r}"
            )
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(
            branch.reference_logps.shape
        )
        if not bool(torch.isfinite(tensor).all().item()):
            raise OfflineReferenceDPOError("reference log-prob tensor is non-finite")
        if float(tensor.sum(dtype=torch.float32).item()) != (
            branch.reference_logps.sequence_logp
        ):
            raise OfflineReferenceDPOError("reference sequence log-prob sum changed")
        return tensor


class OfflineReferenceDPOObjective:
    manifest = ObjectiveManifest(
        objective_id="offline-reference-dpo",
        objective_version="1.0.0",
        requirements=ObjectiveRequirements(
            sample_objectives=frozenset({"preference"}),
            requires_reference_model=False,
        ),
        supported_engines=frozenset({"torch"}),
    )

    def setup(self, context: Any) -> ObjectiveSetup:
        runtime = open_offline_reference_dpo_cache(context)
        return ObjectiveSetup(
            runtime=runtime,
            metadata=runtime.metadata,
            state_count_keys=(
                "preference_pairs",
                "chosen_loss_tokens",
                "rejected_loss_tokens",
            ),
        )

    def prepare(self, batch: ModelBatch, context: Any) -> ModelBatch:
        setup = getattr(context, "objective_setup", None)
        runtime = getattr(setup, "runtime", None)
        if not isinstance(runtime, OfflineReferenceDPOCache):
            raise OfflineReferenceDPOError(
                "offline-reference DPO objective was not preflighted"
            )
        if len(batch.sample_ids) != 1:
            raise OfflineReferenceDPOError("offline-reference DPO v1 requires batch size 1")
        if set(batch.model_inputs) != _PAIR_INPUT_KEYS:
            raise OfflineReferenceDPOError(
                "DPO model batch must contain exactly chosen, rejected and "
                "dpo_pair_identity"
            )
        sample_id = batch.sample_ids[0]
        try:
            pair = runtime.pairs[sample_id]
        except KeyError as exc:
            raise OfflineReferenceDPOError(
                f"DPO cache has no pair for sample {sample_id!r}"
            ) from exc
        try:
            batch_identity = DPOBatchPairIdentity.model_validate(
                batch.model_inputs["dpo_pair_identity"]
            )
        except Exception as exc:
            raise OfflineReferenceDPOError(f"invalid DPO batch pair identity: {exc}") from exc
        _validate_batch_pair_identity(batch_identity, pair)
        chosen_inputs, chosen_positions, chosen_targets = _validate_branch_inputs(
            "chosen", batch.model_inputs["chosen"], pair.chosen
        )
        rejected_inputs, rejected_positions, rejected_targets = _validate_branch_inputs(
            "rejected", batch.model_inputs["rejected"], pair.rejected
        )
        _validate_common_branch_inputs(
            chosen_inputs,
            rejected_inputs,
            chosen_positions,
            rejected_positions,
            pair.common_model_inputs_sha256,
        )
        torch = context.torch
        chosen_reference = runtime.load_branch(torch, pair, "chosen")
        rejected_reference = runtime.load_branch(torch, pair, "rejected")
        prepared_inputs = {
            "dpo_chosen_inputs": chosen_inputs,
            "dpo_rejected_inputs": rejected_inputs,
            "dpo_chosen_positions": torch.tensor(
                chosen_positions, dtype=torch.long
            ).unsqueeze(0),
            "dpo_rejected_positions": torch.tensor(
                rejected_positions, dtype=torch.long
            ).unsqueeze(0),
            "dpo_chosen_targets": torch.tensor(
                chosen_targets, dtype=torch.long
            ).unsqueeze(0),
            "dpo_rejected_targets": torch.tensor(
                rejected_targets, dtype=torch.long
            ).unsqueeze(0),
            "dpo_reference_chosen_logps": chosen_reference.unsqueeze(0),
            "dpo_reference_rejected_logps": rejected_reference.unsqueeze(0),
            "dpo_cache_identity": runtime.manifest.logps.content_sha256,
            "dpo_pair_identity": pair.preference_pair_sha256,
        }
        trace = dict(batch.trace)
        trace["offline_reference_dpo"] = {
            "cache_id": runtime.manifest.cache_id,
            "cache_content_sha256": runtime.manifest.logps.content_sha256,
            "sample_id": sample_id,
            "preference_pair_sha256": pair.preference_pair_sha256,
            "algorithm": runtime.manifest.algorithm.model_dump(mode="json"),
        }
        return ModelBatch(
            sample_ids=batch.sample_ids,
            model_inputs=prepared_inputs,
            plan=batch.plan,
            trace=trace,
        )

    def compute(self, models: Any, batch: ModelBatch) -> LossOutput:
        model = models.get("model") if isinstance(models, Mapping) else models
        if model is None or not callable(model):
            raise OfflineReferenceDPOError("offline-reference DPO requires one policy model")
        if isinstance(models, Mapping) and any(
            key in models for key in ("reference", "reference_model")
        ):
            raise OfflineReferenceDPOError(
                "offline-reference DPO forbids a live reference model"
            )
        values = dict(batch.model_inputs)
        required = {
            "dpo_chosen_inputs",
            "dpo_rejected_inputs",
            "dpo_chosen_positions",
            "dpo_rejected_positions",
            "dpo_chosen_targets",
            "dpo_rejected_targets",
            "dpo_reference_chosen_logps",
            "dpo_reference_rejected_logps",
            "dpo_cache_identity",
            "dpo_pair_identity",
        }
        if set(values) != required:
            raise OfflineReferenceDPOError("prepared DPO batch fields are incomplete")
        trace = batch.trace.get("offline_reference_dpo")
        if not isinstance(trace, Mapping):
            raise OfflineReferenceDPOError("DPO batch trace identity is missing")
        if values["dpo_cache_identity"] != trace.get("cache_content_sha256"):
            raise OfflineReferenceDPOError("DPO batch cache identity mismatch")
        if values["dpo_pair_identity"] != trace.get("preference_pair_sha256"):
            raise OfflineReferenceDPOError("DPO batch pair identity mismatch")
        chosen_output = model(**dict(values["dpo_chosen_inputs"]))
        rejected_output = model(**dict(values["dpo_rejected_inputs"]))
        chosen_logits = _output_logits(chosen_output, "chosen")
        rejected_logits = _output_logits(rejected_output, "rejected")
        torch = _torch_module(chosen_logits)
        if _torch_module(rejected_logits) is not torch:
            raise OfflineReferenceDPOError("chosen/rejected logits use different runtimes")
        beta = float(trace.get("algorithm", {}).get("beta", -1))
        if beta != 0.1:
            raise OfflineReferenceDPOError("offline-reference DPO beta identity mismatch")
        with torch.autocast(device_type=chosen_logits.device.type, enabled=False):
            policy_chosen_tokens = _policy_token_logps(
                chosen_logits,
                values["dpo_chosen_positions"],
                values["dpo_chosen_targets"],
            )
            policy_rejected_tokens = _policy_token_logps(
                rejected_logits,
                values["dpo_rejected_positions"],
                values["dpo_rejected_targets"],
            )
            reference_chosen_tokens = _validate_reference_tensor(
                values["dpo_reference_chosen_logps"], policy_chosen_tokens
            )
            reference_rejected_tokens = _validate_reference_tensor(
                values["dpo_reference_rejected_logps"], policy_rejected_tokens
            )
            policy_chosen_logp = policy_chosen_tokens.sum(dtype=torch.float32)
            policy_rejected_logp = policy_rejected_tokens.sum(dtype=torch.float32)
            reference_chosen_logp = reference_chosen_tokens.sum(dtype=torch.float32)
            reference_rejected_logp = reference_rejected_tokens.sum(dtype=torch.float32)
            policy_log_ratio = policy_chosen_logp - policy_rejected_logp
            reference_log_ratio = reference_chosen_logp - reference_rejected_logp
            delta = policy_log_ratio - reference_log_ratio
            dpo_logit = delta * beta
            loss = torch.nn.functional.softplus(-dpo_logit).mean()
            reward_chosen = beta * (policy_chosen_logp - reference_chosen_logp)
            reward_rejected = beta * (policy_rejected_logp - reference_rejected_logp)
            reward_margin = reward_chosen - reward_rejected
            preference_accuracy = (delta > 0).to(torch.float32).mean()
        metrics_tensors = {
            "policy_chosen_logp": policy_chosen_logp,
            "policy_rejected_logp": policy_rejected_logp,
            "reference_chosen_logp": reference_chosen_logp,
            "reference_rejected_logp": reference_rejected_logp,
            "policy_log_ratio": policy_log_ratio,
            "reference_log_ratio": reference_log_ratio,
            "delta": delta,
            "dpo_logit": dpo_logit,
            "reward_chosen": reward_chosen,
            "reward_rejected": reward_rejected,
            "reward_margin": reward_margin,
            "preference_accuracy": preference_accuracy,
            "loss": loss,
        }
        for name, value in metrics_tensors.items():
            if not bool(torch.isfinite(value).item()):
                raise OfflineReferenceDPOError(f"DPO metric {name!r} is non-finite")
        chosen_count = int(policy_chosen_tokens.numel())
        rejected_count = int(policy_rejected_tokens.numel())
        metrics = {
            name: float(value.detach().item())
            for name, value in metrics_tensors.items()
        }
        metrics.update(
            {
                "chosen_target_tokens": float(chosen_count),
                "rejected_target_tokens": float(rejected_count),
                "pair_count": 1.0,
            }
        )
        return LossOutput(
            total=loss,
            terms={"dpo_loss": LossTerm(value=loss, denominator=1)},
            metrics=metrics,
            counts={
                "loss_tokens": chosen_count + rejected_count,
                "chosen_loss_tokens": chosen_count,
                "rejected_loss_tokens": rejected_count,
                "preference_pairs": 1,
            },
        )


def open_offline_reference_dpo_cache(context: Any) -> OfflineReferenceDPOCache:
    if sys.byteorder != "little":
        raise OfflineReferenceDPOError(
            "offline-reference DPO cache v1 requires a little-endian host"
        )
    resolved = context.context.resolved
    if resolved.run.stage.stage_type != "offline_preference":
        raise OfflineReferenceDPOError(
            "offline-reference DPO requires stage_type='offline_preference'"
        )
    raw_config = resolved.run.metadata.get(DPO_CONFIG_KEY)
    if raw_config is None:
        raise OfflineReferenceDPOError(
            f"run metadata must define {DPO_CONFIG_KEY!r}"
        )
    try:
        config = OfflineReferenceDPOConfig.model_validate(raw_config)
    except Exception as exc:
        raise OfflineReferenceDPOError(f"invalid offline DPO config: {exc}") from exc
    manifest_path = _resolve_from_run(config.cache_manifest, resolved.source)
    if not manifest_path.is_file():
        raise OfflineReferenceDPOError(f"DPO cache manifest is missing: {manifest_path}")
    manifest_sha = _file_sha256(manifest_path)
    if manifest_sha != config.cache_manifest_sha256:
        raise OfflineReferenceDPOError("DPO cache manifest SHA-256 mismatch")
    try:
        manifest = OfflineReferenceDPOCacheManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise OfflineReferenceDPOError(f"invalid DPO cache manifest: {exc}") from exc
    _validate_expected_identities(config, manifest)
    _validate_plugin_identity(context, manifest)
    _validate_checkpoint_pair(manifest.policy, manifest.reference)
    _validate_policy_input(context, manifest.policy)
    _validate_asset_set(manifest_path.parent, manifest.model.assets, "model")
    _validate_asset_set(manifest_path.parent, manifest.tokenizer, "tokenizer")
    _validate_asset_set(manifest_path.parent, manifest.processor, "processor")
    preference_path = Path(manifest.preference_manifest.path)
    if not preference_path.is_absolute():
        preference_path = manifest_path.parent / preference_path
    preference_path = preference_path.resolve()
    if _file_sha256(preference_path) != manifest.preference_manifest.sha256:
        raise OfflineReferenceDPOError("preference manifest SHA-256 mismatch")
    try:
        preference = OfflineDPOPreferenceManifest.model_validate_json(
            preference_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise OfflineReferenceDPOError(f"invalid preference manifest: {exc}") from exc
    if preference.identity_sha256 != manifest.preference_manifest.identity_sha256:
        raise OfflineReferenceDPOError("preference manifest identity mismatch")
    if preference.data.identity_sha256 != manifest.data_identity_sha256:
        raise OfflineReferenceDPOError("DPO data identity mismatch")
    _validate_data_source(preference.data)
    _validate_cached_pairs(manifest.pairs, preference.pairs)
    if manifest.logps.total_bytes > config.max_cache_bytes:
        raise OfflineReferenceDPOError(
            "reference cache exceeds configured max_cache_bytes: "
            f"observed={manifest.logps.total_bytes}, limit={config.max_cache_bytes}"
        )
    content = hashlib.sha256()
    observed_bytes = 0
    for pair in manifest.pairs:
        for name, branch in (("chosen", pair.chosen), ("rejected", pair.rejected)):
            path = (manifest_path.parent / branch.reference_logps.file).resolve()
            if not path.is_relative_to(manifest_path.parent):
                raise OfflineReferenceDPOError("reference tensor path escapes cache root")
            digest, size = _stream_file_identity(path, aggregate=content)
            if digest != branch.reference_logps.sha256:
                raise OfflineReferenceDPOError(
                    f"reference {name} tensor SHA-256 mismatch for {pair.sample_id!r}"
                )
            if size != branch.reference_logps.shape[0] * 4:
                raise OfflineReferenceDPOError(
                    f"reference {name} tensor size mismatch for {pair.sample_id!r}"
                )
            payload = path.read_bytes()
            tensor = context.torch.frombuffer(
                bytearray(payload), dtype=context.torch.float32
            )
            if not bool(context.torch.isfinite(tensor).all().item()):
                raise OfflineReferenceDPOError("reference log-prob tensor is non-finite")
            if float(tensor.sum(dtype=context.torch.float32).item()) != (
                branch.reference_logps.sequence_logp
            ):
                raise OfflineReferenceDPOError("reference sequence log-prob sum mismatch")
            observed_bytes += size
    if observed_bytes != manifest.logps.total_bytes:
        raise OfflineReferenceDPOError("reference cache total byte size mismatch")
    if content.hexdigest() != manifest.logps.content_sha256:
        raise OfflineReferenceDPOError("reference cache total content SHA-256 mismatch")
    metadata = {
        "schema_version": "trainomni.offline-reference-dpo-identity.v1",
        "objective_id": "offline-reference-dpo",
        "objective_version": "1.0.0",
        "cache_manifest_path": str(manifest_path),
        "cache_manifest_sha256": manifest_sha,
        "cache_content_sha256": manifest.logps.content_sha256,
        "cache_id": manifest.cache_id,
        "producer_code_revision": manifest.producer_code_revision,
        "reference": manifest.reference.model_dump(mode="json"),
        "policy": manifest.policy.model_dump(mode="json"),
        "model": manifest.model.model_dump(mode="json"),
        "tokenizer": manifest.tokenizer.model_dump(mode="json"),
        "processor": manifest.processor.model_dump(mode="json"),
        "preference_manifest": manifest.preference_manifest.model_dump(mode="json"),
        "data": preference.data.model_dump(mode="json"),
        "pair_identity_sha256": manifest.pair_identity_sha256,
        "logps": manifest.logps.model_dump(mode="json"),
        "algorithm": manifest.algorithm.model_dump(mode="json"),
    }
    return OfflineReferenceDPOCache(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        preference=preference,
        pairs=MappingProxyType({pair.sample_id: pair for pair in manifest.pairs}),
        metadata=MappingProxyType(metadata),
    )


def dpo_data_identity_digest(data: DPODataIdentity) -> str:
    return canonical_fingerprint(
        {
            "source_path": data.source_path,
            "source_sha256": data.source_sha256,
            "split_fingerprints": data.split_fingerprints,
            "metadata": data.metadata,
        }
    )


def preference_pair_digest(pair: PreferencePairIdentity) -> str:
    value = pair.model_dump(mode="json", exclude={"identity_sha256"})
    return canonical_fingerprint(value)


def preference_manifest_digest(manifest: OfflineDPOPreferenceManifest) -> str:
    return canonical_fingerprint(
        {
            "schema_version": manifest.schema_version,
            "preference_id": manifest.preference_id,
            "producer_code_revision": manifest.producer_code_revision,
            "data": manifest.data.model_dump(mode="json"),
            "pairs": [pair.model_dump(mode="json") for pair in manifest.pairs],
        }
    )


def model_inputs_digest(value: Any) -> str:
    return canonical_fingerprint(_model_input_identity(value))


def _validate_expected_identities(
    config: OfflineReferenceDPOConfig,
    manifest: OfflineReferenceDPOCacheManifest,
) -> None:
    checks = {
        "cache content": (config.cache_content_sha256, manifest.logps.content_sha256),
        "preference manifest": (
            config.preference_manifest_sha256,
            manifest.preference_manifest.sha256,
        ),
        "preference identity": (
            config.preference_identity_sha256,
            manifest.preference_manifest.identity_sha256,
        ),
        "pair identity": (config.pair_identity_sha256, manifest.pair_identity_sha256),
        "data identity": (config.data_identity_sha256, manifest.data_identity_sha256),
        "reference state": (config.reference_state_sha256, manifest.reference.state_sha256),
        "reference manifest": (
            config.reference_manifest_sha256,
            manifest.reference.manifest_sha256,
        ),
        "reference run": (
            config.reference_run_fingerprint,
            manifest.reference.run_fingerprint,
        ),
        "policy state": (config.policy_state_sha256, manifest.policy.state_sha256),
        "policy manifest": (
            config.policy_manifest_sha256,
            manifest.policy.manifest_sha256,
        ),
        "policy run": (config.policy_run_fingerprint, manifest.policy.run_fingerprint),
        "model": (config.model_identity_sha256, manifest.model.assets.identity_sha256),
        "tokenizer": (config.tokenizer_sha256, manifest.tokenizer.identity_sha256),
        "processor": (config.processor_sha256, manifest.processor.identity_sha256),
    }
    mismatched = [name for name, (expected, actual) in checks.items() if expected != actual]
    if mismatched:
        raise OfflineReferenceDPOError(
            f"offline DPO expected identity mismatch: {sorted(mismatched)}"
        )
    algorithm = manifest.algorithm
    config_algorithm = {
        "beta": config.beta,
        "loss_variant": config.loss_variant,
        "label_smoothing": config.label_smoothing,
        "sequence_reduction": config.sequence_reduction,
        "pair_reduction": config.pair_reduction,
        "compute_dtype": config.compute_dtype,
        "reference_free": config.reference_free,
        "auxiliary_ce": config.auxiliary_ce,
    }
    for key, expected in config_algorithm.items():
        if getattr(algorithm, key) != expected:
            raise OfflineReferenceDPOError(f"offline DPO algorithm mismatch: {key}")
    if config.expected_pair_count != manifest.logps.pair_count:
        raise OfflineReferenceDPOError("offline DPO expected pair count mismatch")
    if config.expected_total_positions != manifest.logps.total_positions:
        raise OfflineReferenceDPOError("offline DPO expected position count mismatch")
    if config.vocab_size != manifest.logps.vocab_size:
        raise OfflineReferenceDPOError("offline DPO vocab size mismatch")


def _validate_plugin_identity(context: Any, manifest: OfflineReferenceDPOCacheManifest) -> None:
    plugin = context.context.resolved.plugin_manifest
    if (
        plugin.plugin_id != manifest.model.plugin_id
        or plugin.plugin_version != manifest.model.plugin_version
    ):
        raise OfflineReferenceDPOError("DPO cache model plugin identity mismatch")


def _validate_checkpoint_pair(
    policy: CheckpointIdentity, reference: CheckpointIdentity
) -> None:
    seen: dict[Path, CheckpointIdentity] = {}
    for identity in (policy, reference):
        root = Path(identity.path).resolve()
        previous = seen.get(root)
        if previous is not None:
            if previous.model_dump(mode="json") != identity.model_dump(mode="json"):
                raise OfflineReferenceDPOError(
                    "policy/reference identities disagree for the same checkpoint path"
                )
            continue
        _validate_checkpoint_identity(identity)
        seen[root] = identity


def _validate_checkpoint_identity(identity: CheckpointIdentity) -> None:
    root = Path(identity.path).resolve()
    if not root.is_dir():
        raise OfflineReferenceDPOError(f"checkpoint path is missing: {root}")
    state_path = (root / identity.state_file).resolve()
    manifest_path = (root / identity.manifest_file).resolve()
    if not state_path.is_relative_to(root) or not manifest_path.is_relative_to(root):
        raise OfflineReferenceDPOError("checkpoint identity file escapes root")
    if _file_sha256(state_path) != identity.state_sha256:
        raise OfflineReferenceDPOError(f"checkpoint state SHA-256 mismatch: {state_path}")
    if _file_sha256(manifest_path) != identity.manifest_sha256:
        raise OfflineReferenceDPOError(
            f"checkpoint manifest SHA-256 mismatch: {manifest_path}"
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineReferenceDPOError("checkpoint manifest is unreadable") from exc
    if value.get("metadata", {}).get("run_fingerprint") != identity.run_fingerprint:
        raise OfflineReferenceDPOError("checkpoint run fingerprint mismatch")


def _validate_policy_input(context: Any, policy: CheckpointIdentity) -> None:
    artifacts = context.context.input_artifacts
    reference = artifacts.get("model") or artifacts.get("checkpoint")
    if reference is None or reference.uri is None:
        raise OfflineReferenceDPOError(
            "offline-reference DPO requires a physical policy model input"
        )
    if str(reference) != policy.artifact:
        raise OfflineReferenceDPOError("policy artifact reference mismatch")
    if Path(reference.uri).resolve() != Path(policy.path).resolve():
        raise OfflineReferenceDPOError("policy physical checkpoint path mismatch")


def _validate_asset_set(root: Path, asset: AssetSetIdentity, name: str) -> None:
    for item in asset.files:
        path = Path(item.path)
        if not path.is_absolute():
            path = root / path
        digest, size = _stream_file_identity(path.resolve())
        if digest != item.sha256 or size != item.size_bytes:
            raise OfflineReferenceDPOError(f"{name} asset identity mismatch: {path}")


def _validate_data_source(data: DPODataIdentity) -> None:
    if _file_sha256(Path(data.source_path).resolve()) != data.source_sha256:
        raise OfflineReferenceDPOError("DPO source data SHA-256 mismatch")


def _validate_cached_pairs(
    cached: Sequence[CachedDPOPairIdentity],
    preference: Sequence[PreferencePairIdentity],
) -> None:
    if len(cached) != len(preference):
        raise OfflineReferenceDPOError("cache/preference pair count mismatch")
    for cache_pair, preference_pair in zip(cached, preference, strict=True):
        checks = {
            "sample_id": (cache_pair.sample_id, preference_pair.sample_id),
            "source_index": (cache_pair.source_index, preference_pair.source_index),
            "split": (cache_pair.split, preference_pair.split),
            "order_index": (cache_pair.order_index, preference_pair.order_index),
            "pair identity": (
                cache_pair.preference_pair_sha256,
                preference_pair.identity_sha256,
            ),
            "canonical pair": (
                cache_pair.canonical_pair_sha256,
                preference_pair.canonical_pair_sha256,
            ),
            "common prompt": (
                cache_pair.common_prompt_sha256,
                preference_pair.common_prompt_sha256,
            ),
            "media": (cache_pair.media_sha256, preference_pair.media_sha256),
            "chosen": (
                cache_pair.chosen.canonical_sha256,
                preference_pair.chosen_canonical_sha256,
            ),
            "rejected": (
                cache_pair.rejected.canonical_sha256,
                preference_pair.rejected_canonical_sha256,
            ),
        }
        mismatch = [name for name, pair in checks.items() if pair[0] != pair[1]]
        if mismatch:
            raise OfflineReferenceDPOError(
                f"cache/preference pair mismatch for {cache_pair.sample_id!r}: "
                f"{sorted(mismatch)}"
            )


def _validate_batch_pair_identity(
    actual: DPOBatchPairIdentity, expected: CachedDPOPairIdentity
) -> None:
    checks = {
        "sample_id": (actual.sample_id, expected.sample_id),
        "preference pair": (
            actual.preference_pair_sha256,
            expected.preference_pair_sha256,
        ),
        "canonical pair": (actual.canonical_pair_sha256, expected.canonical_pair_sha256),
        "common prompt": (actual.common_prompt_sha256, expected.common_prompt_sha256),
        "media": (actual.media_sha256, expected.media_sha256),
        "common model inputs": (
            actual.common_model_inputs_sha256,
            expected.common_model_inputs_sha256,
        ),
        "chosen": (actual.chosen_canonical_sha256, expected.chosen.canonical_sha256),
        "rejected": (
            actual.rejected_canonical_sha256,
            expected.rejected.canonical_sha256,
        ),
    }
    mismatch = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if mismatch:
        raise OfflineReferenceDPOError(f"DPO batch pair identity mismatch: {mismatch}")


def _validate_branch_inputs(
    name: str, value: Any, expected: DPOBranchIdentity
) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise OfflineReferenceDPOError(f"DPO {name} branch inputs must be a mapping")
    inputs = dict(value)
    if "input_ids" not in inputs or "labels" not in inputs:
        raise OfflineReferenceDPOError(f"DPO {name} branch requires input_ids and labels")
    input_ids = _single_batch_row(inputs["input_ids"], f"{name} input_ids")
    labels = _single_batch_row(inputs["labels"], f"{name} labels")
    if integer_tensor_digest(input_ids) != expected.input_ids_sha256:
        raise OfflineReferenceDPOError(f"DPO {name} input_ids digest mismatch")
    if integer_tensor_digest(labels) != expected.labels_sha256:
        raise OfflineReferenceDPOError(f"DPO {name} labels digest mismatch")
    label_values = _flatten_ints(labels)
    positions = tuple(index for index, token in enumerate(label_values) if token != -100)
    if positions != expected.assistant_positions:
        raise OfflineReferenceDPOError(
            f"DPO {name} positions differ from real labels loss mask"
        )
    targets = tuple(label_values[position] for position in positions)
    if targets != expected.target_token_ids:
        raise OfflineReferenceDPOError(f"DPO {name} target-token alignment mismatch")
    return inputs, positions, targets


def _validate_common_branch_inputs(
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
    chosen_positions: tuple[int, ...],
    rejected_positions: tuple[int, ...],
    expected_common_inputs_sha256: str,
) -> None:
    if chosen_positions != rejected_positions:
        raise OfflineReferenceDPOError("chosen/rejected target positions differ")
    chosen_ids = _flatten_ints(_single_batch_row(chosen["input_ids"], "chosen input_ids"))
    rejected_ids = _flatten_ints(
        _single_batch_row(rejected["input_ids"], "rejected input_ids")
    )
    if len(chosen_ids) != len(rejected_ids):
        raise OfflineReferenceDPOError("chosen/rejected sequence lengths differ")
    target_positions = set(chosen_positions)
    if any(
        chosen_token != rejected_token
        for index, (chosen_token, rejected_token) in enumerate(
            zip(chosen_ids, rejected_ids, strict=True)
        )
        if index not in target_positions
    ):
        raise OfflineReferenceDPOError(
            "chosen/rejected common prompt token inputs differ"
        )
    chosen_common = {
        key: value for key, value in chosen.items() if key not in {"input_ids", "labels"}
    }
    rejected_common = {
        key: value for key, value in rejected.items() if key not in {"input_ids", "labels"}
    }
    if not _values_equal(chosen_common, rejected_common):
        raise OfflineReferenceDPOError("chosen/rejected common media/model inputs differ")
    if model_inputs_digest(chosen_common) != expected_common_inputs_sha256:
        raise OfflineReferenceDPOError("common media/model input identity mismatch")


def _policy_token_logps(logits: Any, positions: Any, targets: Any) -> Any:
    torch = _torch_module(logits)
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise OfflineReferenceDPOError("policy logits must have shape [1, sequence, vocab]")
    if positions.ndim != 2 or positions.shape[0] != 1 or positions.shape[1] <= 0:
        raise OfflineReferenceDPOError("DPO positions must have shape [1, positive]")
    if positions.dtype != torch.long or targets.dtype != torch.long:
        raise OfflineReferenceDPOError("DPO positions and targets must be int64")
    if positions.shape != targets.shape:
        raise OfflineReferenceDPOError("DPO position/target shapes differ")
    if int(positions.min().item()) <= 0 or int(positions.max().item()) >= logits.shape[1]:
        raise OfflineReferenceDPOError("DPO assistant target position is out of range")
    if int(targets.min().item()) < 0 or int(targets.max().item()) >= logits.shape[-1]:
        raise OfflineReferenceDPOError("DPO target token is outside policy vocab")
    prediction_positions = positions - 1
    selected = logits[0, prediction_positions[0]].float()
    return torch.nn.functional.log_softmax(selected, dim=-1).gather(
        1, targets[0].unsqueeze(1)
    ).squeeze(1)


def _validate_reference_tensor(value: Any, policy_tokens: Any) -> Any:
    torch = _torch_module(policy_tokens)
    if _torch_module(value) is not torch:
        raise OfflineReferenceDPOError("reference log-probs require torch tensors")
    if value.dtype != torch.float32:
        raise OfflineReferenceDPOError("reference log-probs must remain FP32")
    if value.ndim != 2 or value.shape[0] != 1:
        raise OfflineReferenceDPOError("reference log-probs must have shape [1, positions]")
    selected = value[0]
    if selected.shape != policy_tokens.shape:
        raise OfflineReferenceDPOError("reference/policy token log-prob shapes differ")
    if selected.requires_grad:
        raise OfflineReferenceDPOError("reference log-probs must not require gradients")
    if not bool(torch.isfinite(selected).all().item()):
        raise OfflineReferenceDPOError("reference log-probs are non-finite")
    return selected


def _output_logits(output: Any, branch: str) -> Any:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if logits is None:
        raise OfflineReferenceDPOError(f"policy {branch} output does not expose logits")
    return logits


def _model_input_identity(value: Any) -> Any:
    if _is_torch_tensor(value):
        tensor = value.detach().cpu().contiguous()
        byte_view = tensor.view(_torch_module(tensor).uint8).reshape(-1)
        payload = byte_view.numpy().tobytes()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": [int(item) for item in tensor.shape],
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _model_input_identity(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_model_input_identity(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "kind": "bytes",
            "sha256": hashlib.sha256(bytes(value)).hexdigest(),
            "size": len(value),
        }
    raise OfflineReferenceDPOError(
        f"unsupported common model input identity type: {type(value).__name__}"
    )


def _values_equal(left: Any, right: Any) -> bool:
    if _is_torch_tensor(left) or _is_torch_tensor(right):
        return (
            _is_torch_tensor(left)
            and _is_torch_tensor(right)
            and bool(_torch_module(left).equal(left, right))
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _single_batch_row(value: Any, name: str) -> Any:
    shape = _shape(value)
    if len(shape) == 2:
        if shape[0] != 1:
            raise OfflineReferenceDPOError(f"{name} batch dimension must be 1")
        return value[0]
    if len(shape) == 1:
        return value
    raise OfflineReferenceDPOError(f"{name} must be rank 1 or [1, sequence]")


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(item) for item in shape]
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        child = _shape(value[0])
        if any(_shape(item) != child for item in value):
            raise OfflineReferenceDPOError("ragged tensor identity is unsupported")
        return [len(value), *child]
    return []


def _flatten_ints(value: Any) -> list[int]:
    detached = value.detach().cpu() if hasattr(value, "detach") else value
    if hasattr(detached, "reshape") and hasattr(detached, "tolist"):
        return [int(item) for item in detached.reshape(-1).tolist()]
    if isinstance(detached, (list, tuple)):
        result: list[int] = []
        for item in detached:
            result.extend(_flatten_ints(item))
        return result
    return [int(detached)]


def _resolve_from_run(value: str, source: Path | None) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (source.parent if source is not None else Path.cwd()) / path
    return path.resolve()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


def _torch_module(value: Any) -> Any:
    if not _is_torch_tensor(value):
        raise OfflineReferenceDPOError("offline-reference DPO requires torch tensors")
    import torch

    return torch


def _file_sha256(path: Path) -> str:
    return _stream_file_identity(path)[0]


def _stream_file_identity(
    path: Path, *, aggregate: Any | None = None
) -> tuple[str, int]:
    if not path.is_file():
        raise OfflineReferenceDPOError(f"identity file is missing: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            if aggregate is not None:
                aggregate.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size
