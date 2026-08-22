"""Direct PyTorch single-process, DDP and FSDP2 execution backends."""

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from trainomni.contracts.distribution import distribution_hints
from trainomni.core.errors import SpecError
from trainomni.runtime.kernels.compilation import compile_forward
from trainomni.runtime.optimization.gradients import clip_gradients
from trainomni.runtime.optimization.optimizer import build_optimizer
from trainomni.runtime.optimization.scheduler import build_scheduler
from trainomni.specs.run import RunSpec

from .fsdp_state import FSDP2StateAdapter
from .process import ProcessContext
from .selection import remap_selection, selection_names


class TorchExecutionBackend:
    def __init__(
        self,
        *,
        name: str,
        canonical_model: Any,
        execution_model: Any,
        optimizer: Any,
        scheduler: Any | None,
        process: ProcessContext,
    ) -> None:
        self.name = name
        self.canonical_model = canonical_model
        self.execution_model = execution_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.process = process
        self.state_adapter = None

    def accumulation_context(self, *, final_microbatch: bool):
        del final_microbatch
        return nullcontext()

    def backward(self, loss: Any, scaler: Any | None) -> None:
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

    def unscale_gradients(self, scaler: Any | None) -> None:
        if scaler is not None:
            scaler.unscale_(self.optimizer)

    def clip_grad_norm(self, max_norm: float | None) -> float:
        return clip_gradients(self.canonical_model.parameters(), max_norm)

    def step(self, scaler: Any | None) -> None:
        if scaler is None:
            self.optimizer.step()
        else:
            scaler.step(self.optimizer)
            scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "foundation": "torch",
            "torch_version": torch.__version__,
            "rank": self.process.rank,
            "local_rank": self.process.local_rank,
            "world_size": self.process.world_size,
            "process_group_backend": self.process.process_group_backend,
        }

    def close(self) -> None:
        self.process.close()


class DDPExecutionBackend(TorchExecutionBackend):
    def accumulation_context(self, *, final_microbatch: bool):
        if final_microbatch:
            return nullcontext()
        return self.execution_model.no_sync()


class FSDP2ExecutionBackend(TorchExecutionBackend):
    @contextmanager
    def accumulation_context(self, *, final_microbatch: bool):
        if final_microbatch:
            yield
            return
        self.canonical_model.set_requires_gradient_sync(False, recurse=True)
        self.canonical_model.set_reshard_after_backward(False, recurse=True)
        try:
            yield
        finally:
            self.canonical_model.set_requires_gradient_sync(True, recurse=True)
            self.canonical_model.set_reshard_after_backward(True, recurse=True)


def _single(model, selection, run: RunSpec, process: ProcessContext):
    optimizer = build_optimizer(run.optimizer, selection)
    scheduler = build_scheduler(run.scheduler, optimizer, total_steps=run.max_steps)
    return TorchExecutionBackend(
        name="single",
        canonical_model=model,
        execution_model=compile_forward(model, run.compile),
        optimizer=optimizer,
        scheduler=scheduler,
        process=process,
    )


def _ddp(model, selection, run: RunSpec, process: ProcessContext):
    if run.execution.process_group_backend == "hccl":
        device_ids = [process.local_rank]
    else:
        device = next(model.parameters()).device
        device_ids = [process.local_rank] if device.type == "cuda" else None
    optimizer = build_optimizer(run.optimizer, selection)
    scheduler = build_scheduler(run.scheduler, optimizer, total_steps=run.max_steps)
    options = run.execution.ddp
    ddp_kwargs = {
        "device_ids": device_ids,
        "output_device": None if device_ids is None else process.local_rank,
        "find_unused_parameters": options.find_unused_parameters,
        "gradient_as_bucket_view": options.gradient_as_bucket_view,
        "static_graph": options.static_graph,
    }
    ddp_parameters = inspect.signature(
        torch.nn.parallel.DistributedDataParallel
    ).parameters
    buffer_option = (
        "forward_sync_buffers"
        if "forward_sync_buffers" in ddp_parameters
        else "broadcast_buffers"
    )
    ddp_kwargs[buffer_option] = options.broadcast_buffers
    wrapped = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    return DDPExecutionBackend(
        name="torch_ddp",
        canonical_model=model,
        execution_model=compile_forward(wrapped, run.compile),
        optimizer=optimizer,
        scheduler=scheduler,
        process=process,
    )


def _fsdp2(model, selection, run: RunSpec, process: ProcessContext):
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
    except ImportError as exc:
        raise SpecError("selected backend requires PyTorch FSDP2 fully_shard") from exc
    if run.compile.enabled and run.compile.fullgraph:
        raise SpecError("FSDP2 does not support TrainOmni compile.fullgraph")
    options = run.execution.fsdp2
    group_names = selection_names(model, selection)
    hints = distribution_hints(model)
    if hints.is_moe:
        raise SpecError(
            "generic FSDP2 does not claim expert parallelism; choose a delegated MoE backend"
        )
    if hints.replicated_modules or hints.tied_parameter_groups:
        raise SpecError(
            "generic FSDP2 cannot yet preserve explicitly replicated modules or "
            "cross-unit tied parameters; provide a topology-aware backend adapter"
        )
    modules = dict(model.named_modules())
    if options.wrap_policy == "model_declared":
        if not hints.fsdp_units:
            raise SpecError(
                "FSDP2 model_declared wrapping requires non-empty model distribution hints"
            )
        unit_paths = hints.fsdp_units
    else:
        unit_paths = ()
    mesh = init_device_mesh(
        next(model.parameters()).device.type,
        (process.world_size,),
        mesh_dim_names=("dp_shard",),
    )
    offload_policy = CPUOffloadPolicy() if options.cpu_offload else None
    kwargs = {
        "mesh": mesh,
        "reshard_after_forward": options.reshard_after_forward,
    }
    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy
    for path in sorted(unit_paths, key=lambda item: item.count("."), reverse=True):
        fully_shard(modules[path], **kwargs)
    root_kwargs = dict(kwargs)
    # The root owns embeddings/heads and orchestrates nested units. Keeping the
    # root materialized follows the FSDP2 transformer recipe and avoids an
    # unnecessary final all-gather before backward.
    root_kwargs["reshard_after_forward"] = False
    fully_shard(model, **root_kwargs)
    remapped = remap_selection(model, selection, group_names)
    optimizer = build_optimizer(run.optimizer, remapped)
    scheduler = build_scheduler(run.scheduler, optimizer, total_steps=run.max_steps)
    backend = FSDP2ExecutionBackend(
        name="torch_fsdp2",
        canonical_model=model,
        execution_model=compile_forward(model, run.compile),
        optimizer=optimizer,
        scheduler=scheduler,
        process=process,
    )
    backend._fsdp_unit_paths = tuple(unit_paths)
    backend.state_adapter = FSDP2StateAdapter()
    return backend


def build_torch_backend(
    *, model, selection, run: RunSpec, process: ProcessContext
) -> TorchExecutionBackend:
    if run.execution.backend == "single":
        return _single(model, selection, run, process)
    if run.execution.backend == "torch_ddp":
        return _ddp(model, selection, run, process)
    if run.execution.backend == "torch_fsdp2":
        return _fsdp2(model, selection, run, process)
    raise SpecError(f"not a direct torch backend: {run.execution.backend}")
