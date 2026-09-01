"""Strict per-sample identity checks for offline supervision caches."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch

from trainomni.contracts.cache import (
    current_model_inputs_field,
    digest_tensor,
)
from trainomni.core.errors import ObjectiveError


def value_digest(value: torch.Tensor) -> str:
    normalized = value.detach().to(device="cpu", dtype=torch.int64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(normalized.shape)).encode("ascii"))
    digest.update(normalized.numpy().tobytes())
    return digest.hexdigest()


def _expected_rows(inputs: Mapping, labels: torch.Tensor, *, ignore_index: int):
    if labels.ndim != 2:
        raise ObjectiveError("cache identity labels must be [batch, sequence]")
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.shape != labels.shape:
        raise ObjectiveError("cache identity requires input_ids aligned with labels")
    attention = inputs.get("attention_mask")
    if attention is None:
        attention = torch.ones_like(labels, dtype=torch.int64)
    elif not isinstance(attention, torch.Tensor) or attention.shape != labels.shape:
        raise ObjectiveError("cache identity attention_mask must align with labels")
    if not bool(torch.logical_or(attention.eq(0), attention.eq(1)).all().item()):
        raise ObjectiveError("cache identity attention_mask must be binary")
    rows = []
    for index in range(labels.shape[0]):
        valid = attention[index].bool()
        supervised = labels[index].ne(ignore_index)
        if bool(torch.logical_and(supervised, torch.logical_not(valid)).any().item()):
            raise ObjectiveError(
                "cache identity has supervised targets outside the attention mask"
            )
        positions = torch.nonzero(supervised, as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ObjectiveError("cache identity sample has no supervised target positions")
        targets = labels[index].index_select(0, positions)
        rows.append(
            (
                value_digest(input_ids[index]),
                value_digest(attention[index]),
                value_digest(positions),
                value_digest(targets),
            )
        )
    return tuple(rows)


def validate_cache_binding(
    *,
    batch,
    cache_field: str,
    inputs: Mapping,
    labels: torch.Tensor,
    ignore_index: int,
    branch_code: int,
    producer_identity_sha256: str,
) -> None:
    prefix = f"__cache_identity__{cache_field}__"
    expected_fields = {
        "input_ids": prefix + "input_ids_sha256",
        "attention_mask": prefix + "attention_mask_sha256",
        "positions": prefix + "supervised_positions_sha256",
        "targets": prefix + "target_token_ids_sha256",
        "model_inputs": prefix + "model_inputs_sha256",
        "producer": prefix + "producer_identity_sha256",
        "branch": prefix + "branch",
        "current_model_inputs": current_model_inputs_field(cache_field),
    }
    missing = sorted(set(expected_fields.values()) - set(batch.supervision))
    if missing:
        raise ObjectiveError(
            "offline cache is missing immutable identity fields: " + ", ".join(missing)
        )
    rows = _expected_rows(inputs, labels, ignore_index=ignore_index)
    producer = digest_tensor(producer_identity_sha256)
    for index, (
        input_digest,
        attention_digest,
        position_digest,
        target_digest,
    ) in enumerate(rows):
        expected = {
            "input_ids": digest_tensor(input_digest),
            "attention_mask": digest_tensor(attention_digest),
            "positions": digest_tensor(position_digest),
            "targets": digest_tensor(target_digest),
            "producer": producer,
        }
        for name, value in expected.items():
            observed = batch.supervision[expected_fields[name]]
            if (
                not isinstance(observed, torch.Tensor)
                or observed.ndim != 2
                or observed.shape[0] != labels.shape[0]
                or observed.shape[1] != 32
                or not torch.equal(observed[index].detach().cpu(), value)
            ):
                raise ObjectiveError(
                    f"offline cache {cache_field!r} {name} identity mismatch"
                )
        cached_model_inputs = batch.supervision[expected_fields["model_inputs"]]
        current_model_inputs = batch.supervision[
            expected_fields["current_model_inputs"]
        ]
        if (
            not isinstance(cached_model_inputs, torch.Tensor)
            or not isinstance(current_model_inputs, torch.Tensor)
            or cached_model_inputs.ndim != 2
            or current_model_inputs.ndim != 2
            or cached_model_inputs.shape != (labels.shape[0], 32)
            or current_model_inputs.shape != (labels.shape[0], 32)
            or not torch.equal(
                cached_model_inputs[index].detach().cpu(),
                current_model_inputs[index].detach().cpu(),
            )
        ):
            raise ObjectiveError(
                f"offline cache {cache_field!r} model_inputs identity mismatch"
            )
        branches = batch.supervision[expected_fields["branch"]]
        if (
            not isinstance(branches, torch.Tensor)
            or branches.ndim != 1
            or branches.shape[0] != labels.shape[0]
            or int(branches[index].item()) != branch_code
        ):
            raise ObjectiveError(
                f"offline cache {cache_field!r} branch identity mismatch"
            )
