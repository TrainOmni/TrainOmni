"""Evaluator registry with a built-in normalized loss evaluator."""

from __future__ import annotations

from typing import Any

from .protocol import EvaluationManifest, EvaluationRequest, EvaluationResult, Evaluator


class EvaluationError(RuntimeError):
    pass


class EvaluatorRegistry:
    def __init__(self, *, include_builtins: bool = True) -> None:
        self._items: dict[str, Evaluator] = {}
        if include_builtins:
            self.register(LossEvaluator())
            from .command import CommandEvaluator

            self.register(CommandEvaluator())

    def register(self, evaluator: Evaluator) -> None:
        manifest = getattr(evaluator, "manifest", None)
        if not isinstance(manifest, EvaluationManifest):
            raise EvaluationError("evaluator must define EvaluationManifest")
        if not callable(getattr(evaluator, "evaluate", None)):
            raise EvaluationError("evaluator must implement evaluate()")
        if manifest.evaluator_id in self._items:
            raise EvaluationError(
                f"evaluator {manifest.evaluator_id!r} is already registered"
            )
        self._items[manifest.evaluator_id] = evaluator

    def get(self, evaluator_id: str) -> Evaluator:
        try:
            return self._items[evaluator_id]
        except KeyError as exc:
            raise EvaluationError(
                f"unknown evaluator {evaluator_id!r}; available: {sorted(self._items)}"
            ) from exc


class LossEvaluator:
    manifest = EvaluationManifest(
        evaluator_id="loss",
        evaluator_version="1.0.0",
        modalities=frozenset({"text", "image", "video", "audio"}),
    )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if request.objective is None:
            raise EvaluationError("loss evaluator requires an objective")
        max_batches = request.config.get("max_batches")
        if not isinstance(max_batches, int) or max_batches <= 0:
            raise EvaluationError("loss evaluator max_batches must be positive")
        models = request.model_bundle.models()
        model = request.model_bundle.model
        if callable(getattr(model, "eval", None)):
            model.eval()
        numerators: dict[str, float] = {}
        denominators: dict[str, int | float] = {}
        batches = 0
        samples = 0
        context = _no_grad_context()
        with context:
            iterator = iter(request.batches)
            while batches < max_batches:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                prepared = request.objective.prepare(batch, request)
                output = request.objective.compute(models, prepared)
                for name, term in output.terms.items():
                    scalar = _scalar(term.value)
                    numerators[name] = numerators.get(name, 0.0) + scalar * term.denominator
                    denominators[name] = denominators.get(name, 0) + term.denominator
                batches += 1
                samples += len(batch.sample_ids)
        if not batches:
            raise EvaluationError("evaluation dataset produced no batches")
        metrics = {
            f"loss/{name}": numerators[name] / denominators[name]
            for name in sorted(numerators)
        }
        return EvaluationResult(
            evaluator_id=self.manifest.evaluator_id,
            metrics=metrics,
            counts={"batches": batches, "samples": samples},
        )


def _scalar(value: Any) -> float:
    detached = value.detach() if hasattr(value, "detach") else value
    if hasattr(detached, "item"):
        return float(detached.item())
    return float(detached)


def _no_grad_context() -> Any:
    try:
        import torch
    except ImportError:
        from contextlib import nullcontext

        return nullcontext()
    return torch.no_grad()
