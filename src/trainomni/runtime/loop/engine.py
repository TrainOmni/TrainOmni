"""Small task-agnostic PyTorch training engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from trainomni import __version__
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import CheckpointError, OptimizationError, SpecError
from trainomni.runtime.checkpoint.manager import CheckpointManager
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.kernels.activation_checkpointing import (
    apply_activation_checkpointing,
)
from trainomni.runtime.kernels.attention import apply_attention_kernel
from trainomni.runtime.kernels.compilation import compile_forward
from trainomni.runtime.observability import (
    JsonlEventWriter,
    reset_peak_resources,
    snapshot_resources,
)
from trainomni.runtime.optimization.evidence import (
    capture_update_snapshot,
    finalize_update_evidence,
)
from trainomni.runtime.optimization.gradients import clip_gradients
from trainomni.runtime.optimization.optimizer import optimizer_metadata
from trainomni.runtime.random import seed_everything
from trainomni.specs.run import RunSpec

from .step import execute_forward_plan


@dataclass(frozen=True, slots=True)
class StepMetrics:
    global_step: int
    loss: float
    grad_norm: float
    micro_batches: int
    learning_rate: float
    cuda_max_allocated_bytes: int
    cuda_max_reserved_bytes: int
    loss_terms: dict[str, float]
    objective_metrics: dict[str, float]
    data_metrics: dict[str, int | float]
    parameter_evidence: dict[str, dict[str, Any]]


class TrainEngine:
    def __init__(
        self,
        *,
        model: Any,
        objective: Any,
        optimizer: Any,
        scheduler: Any | None,
        stream: Any,
        run: RunSpec,
        task_digest: str,
        module_lock: dict[str, str],
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stream = stream
        self.run = run
        self.device = DeviceContext(run.device, run.precision)
        self.attention_kernel_modules = apply_attention_kernel(
            self.model, run.attention_kernel
        )
        self.activation_checkpoint_components = apply_activation_checkpointing(
            self.model, run.activation_checkpointing
        )
        self.device.prepare_model(self.model)
        self.execution_model = compile_forward(self.model, run.compile)
        self.global_step = 0
        self.micro_step = 0
        self.scaler = self._build_scaler()
        self.checkpoints = CheckpointManager(
            root=run.checkpoint.directory,
            task_digest=task_digest,
            run_digest=run.digest,
            module_lock=module_lock,
            framework_version=__version__,
        )
        self.events = JsonlEventWriter(
            run.checkpoint.directory.parent / "metrics" / "events.jsonl"
        )
        seed_everything(run.seed, deterministic=run.deterministic)
        reset_peak_resources(self.device.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.last_parameter_evidence: dict[str, dict[str, Any]] = {}
        self.events.write(
            "engine_initialized",
            {
                "task_digest": task_digest,
                "run_digest": run.digest,
                "device": str(self.device.device),
                "precision": run.precision,
                "attention_kernel": run.attention_kernel,
                "attention_kernel_modules": self.attention_kernel_modules,
                "compile": {
                    "enabled": run.compile.enabled,
                    "backend": run.compile.backend,
                    "mode": run.compile.mode,
                    "fullgraph": run.compile.fullgraph,
                    "dynamic": run.compile.dynamic,
                },
                "global_step": 0,
            },
        )

    def _build_scaler(self):
        if self.run.precision != "fp16_mixed":
            return None
        if self.device.device.type != "cuda":
            raise SpecError("fp16 training requires a CUDA device")
        return torch.amp.GradScaler("cuda", enabled=True)

    def train(self, *, stop_after_steps: int | None = None) -> tuple[StepMetrics, ...]:
        target = self.run.max_steps
        if stop_after_steps is not None:
            if stop_after_steps < self.global_step:
                raise SpecError("stop_after_steps precedes the current global step")
            target = min(target, stop_after_steps)
        records = []
        starting_step = self.global_step
        self.model.train()
        while self.global_step < target:
            records.append(self._optimizer_step())
        if (
            self.global_step > starting_step
            and self.global_step % self.run.checkpoint.every_steps != 0
        ):
            self.save_checkpoint()
        return tuple(records)

    def _optimizer_step(self) -> StepMetrics:
        accumulated_loss = 0.0
        term_totals: dict[str, list[float]] = {}
        objective_metric_totals: dict[str, float] = {}
        accumulation = self.run.gradient_accumulation_steps
        for index in range(accumulation):
            self.micro_step = index
            batch = self.device.move_batch(
                self.stream.next_batch(self.run.per_device_batch_size)
            )
            context = ObjectiveContext(global_step=self.global_step, micro_step=index)
            bundle = execute_forward_plan(
                model=self.execution_model,
                objective=self.objective,
                batch=batch,
                context=context,
                device=self.device,
            )
            normalized = bundle.total / accumulation
            if self.scaler is None:
                normalized.backward()
            else:
                self.scaler.scale(normalized).backward()
            accumulated_loss += float(bundle.total.detach().float().item())
            for name, term in bundle.terms.items():
                values = term_totals.setdefault(name, [0.0, 0.0])
                values[0] += float(term.numerator.detach().float().item())
                values[1] += float(term.denominator.detach().float().item())
            for name, value in bundle.metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    objective_metric_totals[name] = objective_metric_totals.get(
                        name, 0.0
                    ) + float(value.detach().float().item())

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = clip_gradients(self.model.parameters(), self.run.max_grad_norm)
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise OptimizationError("gradient norm is non-finite; optimizer step aborted")
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        evidence_snapshot = None
        evidence_spec = self.run.update_evidence
        if evidence_spec.enabled and (
            (self.global_step + 1) % evidence_spec.every_steps == 0
        ):
            evidence_snapshot = capture_update_snapshot(
                self.model,
                self.optimizer,
                sample_elements_per_group=evidence_spec.sample_elements_per_group,
            )
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        self.micro_step = 0
        parameter_evidence = {}
        if evidence_snapshot is not None:
            parameter_evidence = finalize_update_evidence(
                evidence_snapshot,
                self.model,
                required_groups=evidence_spec.required_groups,
            )
            self.last_parameter_evidence = parameter_evidence
        resources = snapshot_resources(self.device.device)
        loss_terms = {
            name: numerator / denominator
            for name, (numerator, denominator) in term_totals.items()
        }
        objective_metrics = {
            name: value / accumulation
            for name, value in objective_metric_totals.items()
        }
        data_metric_hook = getattr(self.stream, "metrics", None)
        data_metrics = {} if not callable(data_metric_hook) else dict(data_metric_hook())
        record = StepMetrics(
            global_step=self.global_step,
            loss=accumulated_loss / accumulation,
            grad_norm=grad_norm,
            micro_batches=accumulation,
            learning_rate=learning_rate,
            cuda_max_allocated_bytes=resources.cuda_max_allocated_bytes,
            cuda_max_reserved_bytes=resources.cuda_max_reserved_bytes,
            loss_terms=loss_terms,
            objective_metrics=objective_metrics,
            data_metrics=data_metrics,
            parameter_evidence=parameter_evidence,
        )
        self.events.write(
            "optimizer_step",
            {
                "global_step": record.global_step,
                "loss": record.loss,
                "grad_norm": record.grad_norm,
                "micro_batches": record.micro_batches,
                "learning_rate": record.learning_rate,
                "cuda_max_allocated_bytes": record.cuda_max_allocated_bytes,
                "cuda_max_reserved_bytes": record.cuda_max_reserved_bytes,
                "loss_terms": record.loss_terms,
                "objective_metrics": record.objective_metrics,
                "data_metrics": record.data_metrics,
                "parameter_evidence": record.parameter_evidence,
            },
        )
        if self.global_step % self.run.checkpoint.every_steps == 0:
            self.save_checkpoint()
        return record

    def save_checkpoint(self) -> Path:
        return self.checkpoints.save(
            global_step=self.global_step,
            micro_step=self.micro_step,
            model=self.model,
            optimizer=self.optimizer,
            objective=self.objective,
            stream=self.stream,
            scheduler=self.scheduler,
            scaler=self.scaler,
            runtime_metadata={
                "optimizer": optimizer_metadata(self.optimizer, self.run.optimizer),
                "parameter_evidence": self.last_parameter_evidence,
            },
        )

    def resume(self, checkpoint: Path) -> None:
        if self.global_step != 0 or self.micro_step != 0:
            raise CheckpointError("resume requires a newly constructed engine")
        global_step, micro_step = self.checkpoints.load(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            objective=self.objective,
            stream=self.stream,
            map_location=self.device.device,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        self.global_step = global_step
        self.micro_step = micro_step
        loaded_evidence = self.checkpoints.loaded_runtime_metadata.get(
            "parameter_evidence", {}
        )
        if not isinstance(loaded_evidence, dict):
            raise CheckpointError("checkpoint parameter evidence is invalid")
        self.last_parameter_evidence = loaded_evidence
        self.events.write(
            "checkpoint_resumed",
            {
                "checkpoint": str(Path(checkpoint).resolve()),
                "global_step": global_step,
            },
        )
