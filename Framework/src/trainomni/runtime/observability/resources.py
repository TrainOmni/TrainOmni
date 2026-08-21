"""Device resource snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    cuda_max_allocated_bytes: int
    cuda_max_reserved_bytes: int


def reset_peak_resources(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def snapshot_resources(device: torch.device) -> ResourceSnapshot:
    if device.type != "cuda":
        return ResourceSnapshot(0, 0)
    return ResourceSnapshot(
        cuda_max_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        cuda_max_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )
