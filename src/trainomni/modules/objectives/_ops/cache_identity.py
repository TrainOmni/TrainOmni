"""Strict per-sample identity checks for offline supervision caches."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch

from trainomni.core.errors import ObjectiveError


def value_digest(value: torch.Tensor) -> str:
    normalized = value.detach().to(device="cpu", dtype=torch.int64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(normalized.shape)).encode("ascii"))
    digest.update(normalized.numpy().tobytes())
    return digest.hexdigest()


def digest_tensor(value: str, *, device=None) -> torch.Tensor:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("cache identity must be a lowercase SHA-256 digest")
    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8, device=device)


def _expected_rows(inputs: Mapping, labels: torch.Tensor, *, ignore_index: int):
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.shape != labels.shape:
        raise ObjectiveError("cache identity requires input_ids aligned with labels")
    attention = inputs.get("attention_mask")
    if attention is not None and (
        not isinstance(attention, torch.Tensor) or attention.shape != labels.shape
    ):
        raise ObjectiveError("cache identity attention_mask must align with labels")
    rows = []
    for index in range(labels.shape[0]):
        valid = (
            torch.ones_like(labels[index], dtype=torch.bool)
            if attention is None
            else attention[index].bool()
        )
        ids = input_ids[index][valid]
        row_labels = labels[index][valid]
        positions = torch.nonzero(row_labels.ne(ignore_index), as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ObjectiveError("cache identity sample has no supervised target positions")
        targets = row_labels.index_select(0, positions)
        rows.append(
            (
                value_digest(ids),
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
        "positions": prefix + "supervised_positions_sha256",
        "targets": prefix + "target_token_ids_sha256",
        "producer": prefix + "producer_identity_sha256",
        "branch": prefix + "branch",
    }
    missing = sorted(set(expected_fields.values()) - set(batch.supervision))
    if missing:
        raise ObjectiveError(
            "offline cache is missing immutable identity fields: " + ", ".join(missing)
        )
    rows = _expected_rows(inputs, labels, ignore_index=ignore_index)
    producer = digest_tensor(producer_identity_sha256)
    for index, (input_digest, position_digest, target_digest) in enumerate(rows):
        expected = {
            "input_ids": digest_tensor(input_digest),
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
