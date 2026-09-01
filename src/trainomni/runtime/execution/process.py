"""Launcher-neutral process-group and rank contract."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch

from trainomni.core.errors import SpecError
from trainomni.specs.run import ExecutionSpec

_RANK_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


@dataclass(slots=True)
class ProcessContext:
    backend_name: str
    rank: int
    local_rank: int
    world_size: int
    process_group_backend: str | None
    owns_process_group: bool
    _store: Any | None = None
    _injected_environment: tuple[str, ...] = ()

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.backend_name != "single"

    @classmethod
    def create(cls, spec: ExecutionSpec, *, requested_device: str) -> ProcessContext:
        try:
            import torch.distributed as dist
        except ImportError as exc:
            if spec.backend == "single":
                return cls("single", 0, 0, 1, None, False)
            raise SpecError("this PyTorch build has no distributed package") from exc

        present = {name: os.environ.get(name) for name in _RANK_ENV}
        populated = {name for name, value in present.items() if value is not None}
        if populated and populated != set(_RANK_ENV):
            missing = sorted(set(_RANK_ENV) - populated)
            raise SpecError(
                "distributed launch environment is incomplete; missing "
                + ", ".join(missing)
            )
        if populated:
            try:
                rank = int(present["RANK"] or "")
                local_rank = int(present["LOCAL_RANK"] or "")
                world_size = int(present["WORLD_SIZE"] or "")
            except ValueError as exc:
                raise SpecError("distributed rank environment must contain integers") from exc
        else:
            rank = 0
            local_rank = 0
            world_size = 1
        if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
            raise SpecError("distributed rank environment is out of range")
        if spec.expected_world_size is not None and world_size != spec.expected_world_size:
            raise SpecError(
                "execution world-size mismatch: expected "
                f"{spec.expected_world_size}, launcher supplied {world_size}"
            )
        if spec.backend == "single":
            if world_size != 1 or rank != 0:
                raise SpecError(
                    "execution.backend=single cannot run under a multi-rank launcher"
                )
            return cls("single", 0, local_rank, 1, None, False)
        if not dist.is_available():
            raise SpecError("selected execution backend requires torch.distributed")

        group_backend = cls._resolve_group_backend(
            spec.process_group_backend,
            requested_device=requested_device,
            world_size=world_size,
        )
        owns_group = False
        store = None
        injected = []
        if dist.is_initialized():
            observed_rank = dist.get_rank()
            observed_world_size = dist.get_world_size()
            if (observed_rank, observed_world_size) != (rank, world_size):
                raise SpecError(
                    "initialized process group disagrees with launcher rank/world size"
                )
            observed_backend = str(dist.get_backend())
            if observed_backend != group_backend:
                raise SpecError(
                    "initialized process group backend mismatch: "
                    f"expected {group_backend}, got {observed_backend}"
                )
        else:
            timeout = timedelta(seconds=spec.timeout_seconds)
            if populated:
                if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
                    raise SpecError(
                        "multi-process launch requires MASTER_ADDR and MASTER_PORT"
                    )
                dist.init_process_group(
                    group_backend,
                    init_method="env://",
                    rank=rank,
                    world_size=world_size,
                    timeout=timeout,
                )
            else:
                # Standalone world-size-one execution is a real distributed
                # backend probe without imposing launcher environment on callers.
                try:
                    store = dist.TCPStore(
                        "127.0.0.1",
                        0,
                        1,
                        True,
                        timeout,
                        use_libuv=sys.platform != "win32",
                    )
                except TypeError:
                    store = dist.TCPStore("127.0.0.1", 0, 1, True, timeout)
                dist.init_process_group(
                    group_backend,
                    store=store,
                    rank=0,
                    world_size=1,
                    timeout=timeout,
                )
                for name, value in (
                    ("RANK", "0"),
                    ("LOCAL_RANK", "0"),
                    ("WORLD_SIZE", "1"),
                ):
                    if name not in os.environ:
                        os.environ[name] = value
                        injected.append(name)
            owns_group = True
        return cls(
            spec.backend,
            rank,
            local_rank,
            world_size,
            group_backend,
            owns_group,
            store,
            tuple(injected),
        )

    @staticmethod
    def _resolve_group_backend(
        configured: str, *, requested_device: str, world_size: int
    ) -> str:
        if configured != "auto":
            selected = configured
        else:
            device_type = torch.device(requested_device).type
            if device_type == "cuda":
                selected = "nccl" if torch.distributed.is_nccl_available() else "gloo"
            elif device_type == "npu":
                selected = "hccl"
            else:
                selected = "gloo"
        available = {
            "gloo": torch.distributed.is_gloo_available,
            "nccl": torch.distributed.is_nccl_available,
        }
        checker = available.get(selected)
        if checker is not None and not checker():
            raise SpecError(f"PyTorch distributed backend {selected!r} is unavailable")
        if (
            world_size > 1
            and torch.device(requested_device).type == "cuda"
            and selected == "gloo"
            and configured == "auto"
        ):
            raise SpecError(
                "CUDA multi-rank execution requires NCCL; select gloo explicitly "
                "only for a deliberate compatibility probe"
            )
        return selected

    def resolve_device(self, requested: str) -> str:
        device = torch.device(requested)
        if device.type in {"cuda", "npu"}:
            if device.index is not None and device.index != self.local_rank:
                raise SpecError(
                    f"requested device index {device.index} disagrees with LOCAL_RANK "
                    f"{self.local_rank}"
                )
            return f"{device.type}:{self.local_rank}"
        return str(device)

    def barrier(self) -> None:
        if self.distributed and self.world_size > 1:
            torch.distributed.barrier()

    def reduce_float(self, value: float, *, reduction: str, device: torch.device) -> float:
        if self.world_size == 1:
            return float(value)
        if reduction not in {"sum", "mean", "max"}:
            raise ValueError(f"unsupported reduction: {reduction}")
        tensor_device = (
            device if self.process_group_backend in {"nccl", "hccl"} else torch.device("cpu")
        )
        tensor = torch.tensor(float(value), dtype=torch.float64, device=tensor_device)
        operation = {
            "sum": torch.distributed.ReduceOp.SUM,
            "mean": torch.distributed.ReduceOp.SUM,
            "max": torch.distributed.ReduceOp.MAX,
        }[reduction]
        torch.distributed.all_reduce(tensor, op=operation)
        if reduction == "mean":
            tensor /= self.world_size
        return float(tensor.cpu().item())

    def all_gather_metrics(
        self, value: Mapping[str, int | float]
    ) -> tuple[dict[str, int | float], ...]:
        local = dict(value)
        if any(
            not isinstance(name, str)
            or not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(float(metric))
            for name, metric in local.items()
        ):
            raise SpecError("data metrics must contain finite numeric values")
        if self.world_size == 1:
            return (local,)
        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, local)
        if any(not isinstance(item, Mapping) for item in gathered):
            raise SpecError("distributed data metrics are invalid")
        return tuple(dict(item) for item in gathered)

    def close(self) -> None:
        if self.owns_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        for name in self._injected_environment:
            os.environ.pop(name, None)
        self.owns_process_group = False
        self._store = None
