"""Objective-driven held-out evaluation using the training forward ABI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from trainomni.core.context import ObjectiveContext
from trainomni.core.errors import SpecError
from trainomni.runtime.device.context import DeviceContext
from trainomni.runtime.loop.step import execute_forward_plan


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    batches: int
    samples: int
    metrics: dict[str, Any]


def evaluate_batches(
    *,
    model,
    objective,
    stream,
    evaluators,
    device: DeviceContext,
    batches: int,
    batch_size: int,
    execution_model=None,
) -> EvaluationResult:
    if batches <= 0:
        raise SpecError("evaluation batches must be positive")
    if batch_size <= 0:
        raise SpecError("evaluation batch_size must be positive")
    if not evaluators:
        raise SpecError("evaluation requires at least one evaluator")
    for evaluator in evaluators:
        evaluator.reset()
    was_training = model.training
    if execution_model is None:
        execution_model = model
    model.eval()
    sample_count = 0
    try:
        with torch.inference_mode():
            for index in range(batches):
                batch = device.move_batch(stream.next_batch(batch_size))
                sample_count += len(batch.sample_ids)
                loss = execute_forward_plan(
                    model=execution_model,
                    objective=objective,
                    batch=batch,
                    context=ObjectiveContext(
                        global_step=0,
                        micro_step=index,
                        training=False,
                    ),
                    device=device,
                )
                for evaluator in evaluators:
                    evaluator.update(batch, loss)
    finally:
        model.train(was_training)
    metrics = {}
    for evaluator in evaluators:
        values = evaluator.compute()
        overlap = sorted(set(metrics) & set(values))
        if overlap:
            raise SpecError(
                "evaluators produced duplicate metric names: " + ", ".join(overlap)
            )
        metrics.update(values)
    return EvaluationResult(batches=batches, samples=sample_count, metrics=metrics)
