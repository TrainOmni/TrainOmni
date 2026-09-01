"""Small task-agnostic PyTorch training engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from trainomni import __version__
from trainomni.contracts.loss import ObjectiveMetric
from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import CheckpointError, OptimizationError, SpecError
from trainomni.runtime.checkpoint.manager import CheckpointManager
from trainomni.runtime.execution import build_execution_backend
from trainomni.runtime.kernels.activation_checkpointing import (
    apply_activation_checkpointing,
)
from trainomni.runtime.kernels.attention import apply_attention_kernel
from trainomni.runtime.observability import (
    JsonlEventWriter,
    NullEventWriter,
    reset_peak_resources,
    snapshot_resources,
)
from trainomni.runtime.optimization.evidence import (
    capture_update_snapshot,
    finalize_update_evidence,
)
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
    data_metrics_by_rank: tuple[dict[str, int | float], ...]
    parameter_evidence: dict[str, dict[str, Any]]


class TrainEngine:
    def __init__(
        self,
        *,
        model: Any,
        objective: Any,
        parameter_selection: Any,
        stream: Any,
        run: RunSpec,
        task_digest: str,
        module_lock: dict[str, str],
        reproducible: bool = True,
        provenance_issues: tuple[str, ...] = (),
    ) -> None:
        if run.checkpoint.enabled and not reproducible:
            raise SpecError(
                "exact-resume checkpointing requires immutable external asset identity: "
                + "; ".join(provenance_issues)
            )
        self.objective = objective
        self.stream = stream
        self.run = run
        self.attention_kernel_modules = apply_attention_kernel(
            model, run.attention_kernel
        )
        self.activation_checkpoint_components = apply_activation_checkpointing(
            model, run.activation_checkpointing
        )
        self.execution = build_execution_backend(
            model=model,
            selection=parameter_selection,
            run=run,
        )
        self.model = self.execution.canonical_model
        self.execution_model = self.execution.execution_model
        self.optimizer = self.execution.optimizer
        self.scheduler = self.execution.scheduler
        self.process = self.execution.process
        self.device = self.execution.device
        if self.process.world_size > 1:
            shard = getattr(self.stream, "shard", None)
            if not callable(shard):
                self.execution.close()
                raise SpecError(
                    "multi-rank execution requires a rank-shardable batch stream"
                )
            shard(rank=self.process.rank, world_size=self.process.world_size)
        self.global_step = 0
        self.micro_step = 0
        self.scaler = self._build_scaler()
        self.checkpoints = CheckpointManager(
            root=run.checkpoint.directory,
            task_digest=task_digest,
            run_digest=run.digest,
            module_lock=module_lock,
            compatible_run_digests=(run.legacy_path_bound_digest,),
            framework_version=__version__,
            process=self.process,
            state_adapter=getattr(self.execution, "state_adapter", None),
        )
        self.events = (
            JsonlEventWriter(
                run.checkpoint.directory.parent / "metrics" / "events.jsonl"
            )
            if self.process.is_primary
            else NullEventWriter()
        )
        seed_everything(run.seed, deterministic=run.deterministic)
        reset_peak_resources(self.device.device)
        self.execution.zero_grad()
        self.last_parameter_evidence: dict[str, dict[str, Any]] = {}
        self.events.write(
            "engine_initialized",
            {
                "task_digest": task_digest,
                "run_digest": run.digest,
                "reproducible": reproducible,
                "provenance_issues": provenance_issues,
                "device": str(self.device.device),
                "precision": run.precision,
                "execution": self.execution.metadata(),
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
        if self.execution.name == "deepspeed":
            return None
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
            self.run.checkpoint.enabled
            and self.global_step > starting_step
            and self.global_step % self.run.checkpoint.every_steps != 0
        ):
            self.save_checkpoint()
        return tuple(records)

    def _optimizer_step(self) -> StepMetrics:
        term_totals: dict[str, list[float]] = {}
        term_weights: dict[str, float] = {}
        objective_metric_totals: dict[str, list[Any]] = {}
        local_normalization_denominator = 0.0
        accumulation = self.run.gradient_accumulation_steps
        for index in range(accumulation):
            self.micro_step = index
            with self.execution.accumulation_context(
                final_microbatch=index == accumulation - 1
            ):
                batch = self.device.move_batch(
                    self.stream.next_batch(self.run.per_device_batch_size)
                )
                context = ObjectiveContext(
                    global_step=self.global_step, micro_step=index
                )
                bundle = execute_forward_plan(
                    model=self.execution_model,
                    objective=self.objective,
                    batch=batch,
                    context=context,
                    device=self.device,
                )
                denominator = next(iter(bundle.terms.values())).denominator
                denominator_value = float(denominator.detach().float().item())
                # Accumulate unnormalized local numerators.  A single effective-
                # batch denominator is applied after all microbatches and ranks.
                self.execution.backward(
                    bundle.total * denominator.detach(),
                    self.scaler,
                )
            local_normalization_denominator += denominator_value
            for name, term in bundle.terms.items():
                values = term_totals.setdefault(name, [0.0, 0.0])
                values[0] += float(term.numerator.detach().float().item())
                values[1] += float(term.denominator.detach().float().item())
                weight = float(term.weight)
                if name in term_weights and term_weights[name] != weight:
                    raise OptimizationError(
                        f"loss term {name!r} changed weight across microbatches"
                    )
                term_weights[name] = weight
            for name, metric in bundle.metrics.items():
                if not isinstance(metric, ObjectiveMetric):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise OptimizationError(
                        f"objective metric {name!r} has no aggregation contract"
                    )
                values = objective_metric_totals.setdefault(
                    name,
                    [metric.aggregation, 0.0, 0.0],
                )
                values[1] += float(metric.numerator.detach().float().item())
                if metric.denominator is not None:
                    values[2] += float(metric.denominator.detach().float().item())

        self.execution.unscale_gradients(self.scaler)
        global_normalization_denominator = self.process.reduce_float(
            local_normalization_denominator,
            reduction="sum",
            device=self.device.device,
        )
        if not math.isfinite(global_normalization_denominator) or (
            global_normalization_denominator <= 0
        ):
            self.optimizer.zero_grad(set_to_none=True)
            raise OptimizationError("global loss denominator must be finite and positive")
        self.execution.normalize_gradients(global_normalization_denominator)
        grad_norm = self.execution.clip_grad_norm(self.run.max_grad_norm)
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise OptimizationError(
                "gradient norm is non-finite; optimizer step aborted"
            )
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
        self.execution.step(self.scaler)
        self.execution.zero_grad()
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
            name: self.process.reduce_float(
                numerator, reduction="sum", device=self.device.device
            )
            / self.process.reduce_float(
                denominator, reduction="sum", device=self.device.device
            )
            for name, (numerator, denominator) in term_totals.items()
        }
        accumulated_loss = sum(
            term_weights[name] * value for name, value in loss_terms.items()
        )
        objective_metrics = {}
        for name, (aggregation, numerator, denominator) in objective_metric_totals.items():
            global_numerator = self.process.reduce_float(
                numerator,
                reduction="sum",
                device=self.device.device,
            )
            if aggregation == "sum":
                objective_metrics[name] = global_numerator
                continue
            global_denominator = self.process.reduce_float(
                denominator,
                reduction="sum",
                device=self.device.device,
            )
            if not math.isfinite(global_denominator) or global_denominator <= 0:
                self.optimizer.zero_grad(set_to_none=True)
                raise OptimizationError(
                    f"objective metric {name!r} global denominator must be positive"
                )
            objective_metrics[name] = global_numerator / global_denominator
        data_metric_hook = getattr(self.stream, "metrics", None)
        data_metrics = (
            {} if not callable(data_metric_hook) else dict(data_metric_hook())
        )
        data_metrics_by_rank = self.process.all_gather_metrics(data_metrics)
        record = StepMetrics(
            global_step=self.global_step,
            loss=accumulated_loss,
            grad_norm=self.process.reduce_float(
                grad_norm, reduction="mean", device=self.device.device
            ),
            micro_batches=accumulation,
            learning_rate=learning_rate,
            cuda_max_allocated_bytes=int(
                self.process.reduce_float(
                    resources.cuda_max_allocated_bytes,
                    reduction="max",
                    device=self.device.device,
                )
            ),
            cuda_max_reserved_bytes=int(
                self.process.reduce_float(
                    resources.cuda_max_reserved_bytes,
                    reduction="max",
                    device=self.device.device,
                )
            ),
            loss_terms=loss_terms,
            objective_metrics=objective_metrics,
            data_metrics=data_metrics,
            data_metrics_by_rank=data_metrics_by_rank,
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
                "data_metrics_by_rank": [
                    {"rank": rank, "metrics": metrics}
                    for rank, metrics in enumerate(record.data_metrics_by_rank)
                ],
                "parameter_evidence": record.parameter_evidence,
            },
        )
        if (
            self.run.checkpoint.enabled
            and self.global_step % self.run.checkpoint.every_steps == 0
        ):
            self.save_checkpoint()
        return record

    def save_checkpoint(self) -> Path:
        if not self.run.checkpoint.enabled:
            raise SpecError("checkpointing is disabled for this run")
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
                "execution": self.execution.metadata(),
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

    def close(self) -> None:
        self.execution.close()
