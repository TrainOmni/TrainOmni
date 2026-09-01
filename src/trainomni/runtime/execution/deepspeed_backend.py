"""Thin optional DeepSpeed engine adapter.

DeepSpeed owns backward, optimizer step and ZeRO partitioning. TrainOmni retains
task/data/objective identity. Native ZeRO checkpoint bridging remains fail-closed.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import Any

import torch

from trainomni.contracts.distribution import distribution_hints
from trainomni.core.errors import SpecError
from trainomni.runtime.optimization.gradients import clip_gradients, scale_gradients
from trainomni.runtime.optimization.optimizer import build_optimizer
from trainomni.runtime.optimization.scheduler import build_scheduler
from trainomni.specs.run import RunSpec

from .process import ProcessContext


class DeepSpeedExecutionBackend:
    name = "deepspeed"

    def __init__(
        self,
        *,
        engine,
        canonical_model,
        optimizer,
        scheduler,
        process: ProcessContext,
        version: str,
        config: dict[str, Any],
    ) -> None:
        self.engine = engine
        self.canonical_model = canonical_model
        self.execution_model = engine
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.process = process
        self.version = version
        self.config = config
        self.state_adapter = None

    def accumulation_context(self, *, final_microbatch: bool):
        del final_microbatch
        # TrainOmni performs numerical accumulation outside DeepSpeed. ZeRO
        # synchronization remains upstream-owned; this is correct though less
        # communication-efficient until a multi-rank optimization gate runs.
        return nullcontext()

    def backward(self, loss: Any, scaler: Any | None) -> None:
        if scaler is not None:
            raise SpecError("DeepSpeed owns loss scaling; external GradScaler is invalid")
        self.engine.backward(loss)

    def unscale_gradients(self, scaler: Any | None) -> None:
        if scaler is not None:
            raise SpecError("DeepSpeed owns loss scaling; external GradScaler is invalid")

    def normalize_gradients(self, global_denominator: float) -> None:
        scale_gradients(
            self.canonical_model.parameters(),
            self.process.world_size / global_denominator,
        )

    def clip_grad_norm(self, max_norm: float | None) -> float:
        # DeepSpeed applies configured clipping in engine.step(). This local
        # observation is exact for the certified world-size-one path. A later
        # multi-rank gate must compare it with DeepSpeed's cached global norm.
        return clip_gradients(
            self.canonical_model.parameters(),
            None,
        )

    def step(self, scaler: Any | None) -> None:
        if scaler is not None:
            raise SpecError("DeepSpeed owns loss scaling; external GradScaler is invalid")
        self.engine.step()

    def zero_grad(self) -> None:
        self.engine.zero_grad()

    def metadata(self) -> dict[str, Any]:
        zero = self.config["zero_optimization"]
        return {
            "backend": self.name,
            "foundation": "deepspeed",
            "deepspeed_version": self.version,
            "torch_version": torch.__version__,
            "rank": self.process.rank,
            "local_rank": self.process.local_rank,
            "world_size": self.process.world_size,
            "process_group_backend": self.process.process_group_backend,
            "zero_stage": zero["stage"],
            "offload_optimizer": zero.get("offload_optimizer", {}).get(
                "device", "none"
            ),
            "offload_parameters": zero.get("offload_param", {}).get(
                "device", "none"
            ),
        }

    def close(self) -> None:
        self.process.close()


def _config(run: RunSpec, process: ProcessContext) -> dict[str, Any]:
    options = run.execution.deepspeed
    zero: dict[str, Any] = {
        "stage": options.zero_stage,
        "overlap_comm": options.overlap_comm,
        "contiguous_gradients": options.contiguous_gradients,
    }
    if options.offload_optimizer != "none":
        zero["offload_optimizer"] = {"device": options.offload_optimizer}
    if options.offload_parameters != "none":
        zero["offload_param"] = {"device": options.offload_parameters}
    return {
        "train_micro_batch_size_per_gpu": run.per_device_batch_size,
        # Numerical accumulation is owned by TrainOmni's objective loop. DS is
        # presented one already-normalized optimizer step at a time.
        "gradient_accumulation_steps": 1,
        "train_batch_size": run.per_device_batch_size * process.world_size,
        "gradient_clipping": 0.0 if run.max_grad_norm is None else run.max_grad_norm,
        "bf16": {"enabled": run.precision in {"bf16_mixed", "bf16_true"}},
        "fp16": {"enabled": run.precision == "fp16_mixed"},
        "zero_optimization": zero,
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 0,
        "wall_clock_breakdown": False,
    }


def build_deepspeed_backend(
    *, model, selection, run: RunSpec, process: ProcessContext
) -> DeepSpeedExecutionBackend:
    if sys.platform == "win32":
        raise SpecError(
            "DeepSpeed training is fail-closed on native Windows: the current "
            "upstream 0.19.x engine crashes in CUDA backward with the Windows "
            "Gloo-only PyTorch build. Use the same RunSpec on Linux/NCCL."
        )
    if run.compile.enabled:
        raise SpecError(
            "TrainOmni torch.compile cannot wrap DeepSpeed; use an upstream "
            "DeepSpeed compile option only after a separate capability gate"
        )
    if run.checkpoint.enabled:
        raise SpecError(
            "DeepSpeed checkpointing is not yet bridged into TrainOmni's atomic "
            "identity format; set checkpoint.enabled=false for a Linux execution "
            "probe or use torch_fsdp2 for checkpointed distributed training"
        )
    if distribution_hints(model).is_moe:
        raise SpecError(
            "generic DeepSpeed ZeRO does not claim expert parallelism; use a "
            "model/backend adapter that owns expert and router process groups"
        )
    try:
        import deepspeed
    except ImportError as exc:
        raise SpecError(
            "execution.backend=deepspeed requires the optional deepspeed package"
        ) from exc
    optimizer = build_optimizer(run.optimizer, selection)
    scheduler = build_scheduler(run.scheduler, optimizer, total_steps=run.max_steps)
    config = _config(run, process)
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
        dist_init_required=False,
    )
    return DeepSpeedExecutionBackend(
        engine=engine,
        canonical_model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        process=process,
        version=str(deepspeed.__version__),
        config=config,
    )
