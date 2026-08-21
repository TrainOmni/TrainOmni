"""Deterministic actual-parameter update evidence for optimizer groups."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from trainomni.core.errors import OptimizationError


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _group_digest(values: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, tensor_digest in values:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _TensorSnapshot:
    name: str
    digest: str
    sample_indices: torch.Tensor
    sample_values: torch.Tensor


@dataclass(frozen=True, slots=True)
class _GroupSnapshot:
    name: str
    gradient_norm: float
    parameter_count: int
    parameter_numel: int
    parameter_dtypes: tuple[str, ...]
    digest: str
    tensors: tuple[_TensorSnapshot, ...]


@dataclass(frozen=True, slots=True)
class UpdateSnapshot:
    groups: tuple[_GroupSnapshot, ...]


def _gradient_norm(parameters: list[torch.Tensor]) -> float:
    device = parameters[0].device
    squared = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum()
    return float(squared.sqrt().item())


def _sample_indices(*, numel: int, count: int) -> torch.Tensor:
    """Return deterministic, in-bounds, evenly distributed CPU indices.

    CUDA ``linspace(..., dtype=int64)`` can round large tensor endpoints out of
    bounds because its implementation passes through floating-point arithmetic.
    Update evidence must never turn a valid optimizer step into a device assert,
    so index construction stays on CPU and uses exact integer arithmetic.
    """

    if numel <= 0 or count <= 0:
        return torch.empty(0, dtype=torch.int64)
    count = min(numel, count)
    if count == 1:
        return torch.zeros(1, dtype=torch.int64)
    positions = torch.arange(count, dtype=torch.int64)
    return positions.mul(numel - 1).div(count - 1, rounding_mode="floor")


def capture_update_snapshot(
    model: Any,
    optimizer: Any,
    *,
    sample_elements_per_group: int,
) -> UpdateSnapshot:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for index, raw_group in enumerate(optimizer.param_groups):
        group_name = str(raw_group.get("group_name", f"group-{index}"))
        parameters = list(raw_group["params"])
        if not parameters:
            raise OptimizationError(f"optimizer group {group_name!r} is empty")
        tensor_budget = max(1, sample_elements_per_group // len(parameters))
        snapshots = []
        digests = []
        for parameter_index, parameter in enumerate(parameters):
            name = names.get(id(parameter), f"<unnamed:{parameter_index}>")
            tensor_digest = _tensor_digest(parameter)
            digests.append((name, tensor_digest))
            indices = _sample_indices(
                numel=parameter.numel(),
                count=tensor_budget,
            )
            if indices.numel():
                values = (
                    parameter.detach()
                    .reshape(-1)
                    .index_select(0, indices.to(parameter.device))
                    .float()
                    .cpu()
                )
            else:
                values = torch.empty(0, dtype=torch.float32)
            snapshots.append(
                _TensorSnapshot(name, tensor_digest, indices, values)
            )
        groups.append(
            _GroupSnapshot(
                name=group_name,
                gradient_norm=_gradient_norm(parameters),
                parameter_count=len(parameters),
                parameter_numel=sum(parameter.numel() for parameter in parameters),
                parameter_dtypes=tuple(
                    sorted({str(parameter.dtype) for parameter in parameters})
                ),
                digest=_group_digest(digests),
                tensors=tuple(snapshots),
            )
        )
    return UpdateSnapshot(tuple(groups))


def finalize_update_evidence(
    snapshot: UpdateSnapshot,
    model: Any,
    *,
    required_groups: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    parameters = dict(model.named_parameters())
    evidence = {}
    available = {group.name for group in snapshot.groups}
    missing = sorted(set(required_groups) - available)
    if missing:
        raise OptimizationError(
            "required update-evidence groups do not exist: " + ", ".join(missing)
        )
    for group in snapshot.groups:
        after_digests = []
        changed_tensors = []
        changed_sampled_elements = 0
        sampled_elements = 0
        max_abs_sampled_update = 0.0
        for tensor in group.tensors:
            try:
                parameter = parameters[tensor.name]
            except KeyError as exc:
                raise OptimizationError(
                    f"parameter disappeared during optimizer step: {tensor.name}"
                ) from exc
            after_digest = _tensor_digest(parameter)
            after_digests.append((tensor.name, after_digest))
            if after_digest != tensor.digest:
                changed_tensors.append(tensor.name)
            if tensor.sample_indices.numel():
                after_values = (
                    parameter.detach()
                    .reshape(-1)
                    .index_select(0, tensor.sample_indices.to(parameter.device))
                    .float()
                    .cpu()
                )
                difference = (after_values - tensor.sample_values).abs()
                changed_sampled_elements += int(difference.ne(0).sum().item())
                sampled_elements += int(difference.numel())
                if difference.numel():
                    max_abs_sampled_update = max(
                        max_abs_sampled_update,
                        float(difference.max().item()),
                    )
        after_digest = _group_digest(after_digests)
        group_evidence = {
            "gradient_norm": group.gradient_norm,
            "parameter_count": group.parameter_count,
            "parameter_numel": group.parameter_numel,
            "parameter_dtypes": list(group.parameter_dtypes),
            "before_sha256": group.digest,
            "after_sha256": after_digest,
            "changed_tensor_count": len(changed_tensors),
            "changed_tensors": changed_tensors,
            "sampled_elements": sampled_elements,
            "changed_sampled_elements": changed_sampled_elements,
            "max_abs_sampled_update": max_abs_sampled_update,
        }
        evidence[group.name] = group_evidence
        if group.name in required_groups and not changed_tensors:
            raise OptimizationError(
                f"required optimizer group {group.name!r} had no actual parameter update"
            )
    return evidence
