"""Typed modal feature boundaries between encoders, connectors, and fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ModalFeatures:
    embeddings: Any
    mask: Any | None = None
    positions: Any | None = None
    grid: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ModalFeatureBranch:
    """One named and typed connector output supplied to a fusion module."""

    name: str
    modality: str
    features: ModalFeatures

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("modal feature branch name must not be empty")
        if not self.modality:
            raise ValueError("modal feature branch modality must not be empty")


@dataclass(frozen=True, slots=True)
class ModalFeatureSet:
    """Ordered modal branches; fusion decides how their features interact."""

    branches: tuple[ModalFeatureBranch, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(branch.name for branch in self.branches)
        if len(names) != len(set(names)):
            raise ValueError("modal feature branch names must be unique")

    @classmethod
    def coerce(cls, value: ModalFeatureSet | ModalFeatures) -> ModalFeatureSet:
        if isinstance(value, cls):
            return value
        if isinstance(value, ModalFeatures):
            return cls((ModalFeatureBranch("default", "unknown", value),))
        raise TypeError("fusion requires ModalFeatureSet or ModalFeatures")

    def require(self, name: str) -> ModalFeatures:
        for branch in self.branches:
            if branch.name == name:
                return branch.features
        raise KeyError(f"modal feature branch does not exist: {name}")

    def concatenate(self) -> ModalFeatures:
        """Concatenate branches in declared order while preserving boundaries."""

        if not self.branches:
            raise ValueError("cannot concatenate an empty modal feature set")
        embeddings = []
        batch_size = None
        hidden_size = None
        for branch in self.branches:
            value = branch.features.embeddings
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError(
                    f"branch {branch.name!r} embeddings must be [batch, tokens, hidden]"
                )
            current_batch, _, current_hidden = value.shape
            if batch_size is None:
                batch_size, hidden_size = current_batch, current_hidden
            elif (current_batch, current_hidden) != (batch_size, hidden_size):
                raise ValueError("modal branches have incompatible batch or hidden sizes")
            embeddings.append(value)

        masks = [branch.features.mask for branch in self.branches]
        merged_mask = None
        if any(mask is not None for mask in masks):
            normalized_masks = []
            for branch, mask in zip(self.branches, masks, strict=True):
                tokens = branch.features.embeddings.shape[1]
                if mask is None:
                    mask = torch.ones(
                        (batch_size, tokens),
                        dtype=torch.bool,
                        device=branch.features.embeddings.device,
                    )
                if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (
                    batch_size,
                    tokens,
                ):
                    raise ValueError(
                        f"branch {branch.name!r} mask must match [batch, tokens]"
                    )
                normalized_masks.append(mask)
            merged_mask = torch.cat(normalized_masks, dim=1)

        positions = [branch.features.positions for branch in self.branches]
        merged_positions = None
        if any(position is not None for position in positions):
            if any(position is None for position in positions):
                raise ValueError(
                    "modal positions must be provided for every branch or no branch"
                )
            normalized_positions = []
            for branch, position in zip(self.branches, positions, strict=True):
                tokens = branch.features.embeddings.shape[1]
                if not isinstance(position, torch.Tensor) or tuple(position.shape) != (
                    batch_size,
                    tokens,
                ):
                    raise ValueError(
                        f"branch {branch.name!r} positions must match [batch, tokens]"
                    )
                normalized_positions.append(position)
            merged_positions = torch.cat(normalized_positions, dim=1)

        offset = 0
        branch_slices = {}
        branch_metadata = {}
        grids = {}
        modalities = {}
        for branch in self.branches:
            next_offset = offset + branch.features.embeddings.shape[1]
            branch_slices[branch.name] = (offset, next_offset)
            branch_metadata[branch.name] = branch.features.metadata
            grids[branch.name] = branch.features.grid
            modalities[branch.name] = branch.modality
            offset = next_offset
        return ModalFeatures(
            embeddings=torch.cat(embeddings, dim=1),
            mask=merged_mask,
            positions=merged_positions,
            grid=MappingProxyType(grids),
            metadata={
                "branch_order": tuple(branch.name for branch in self.branches),
                "branch_slices": MappingProxyType(branch_slices),
                "branch_modalities": MappingProxyType(modalities),
                "branch_metadata": MappingProxyType(branch_metadata),
            },
        )
