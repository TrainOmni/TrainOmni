"""Explicit shell-free adapter for delegated benchmark harnesses."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .protocol import EvaluationManifest, EvaluationRequest, EvaluationResult
from .registry import EvaluationError


class CommandEvaluator:
    manifest = EvaluationManifest(
        evaluator_id="command",
        evaluator_version="1.0.0",
        modalities=frozenset({"text", "image", "video", "audio"}),
        requires_generation=True,
        delegated=True,
    )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        command = request.config.get("command")
        if not isinstance(command, (list, tuple)) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise EvaluationError("command evaluator requires a non-empty argv list")
        if request.config.get("allow_external_command") is not True:
            raise EvaluationError("external evaluator command was not explicitly allowed")
        timeout = request.config.get("timeout_seconds", 3600)
        completed = subprocess.run(
            list(command),
            cwd=Path(request.output_dir),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        (Path(request.output_dir) / "evaluator.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (Path(request.output_dir) / "evaluator.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            raise EvaluationError(
                f"evaluator command exited with code {completed.returncode}"
            )
        try:
            payload = json.loads(completed.stdout)
            metrics = payload["metrics"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EvaluationError(
                "evaluator stdout must be JSON with a metrics mapping"
            ) from exc
        if not isinstance(metrics, dict):
            raise EvaluationError("evaluator metrics must be a mapping")
        return EvaluationResult(
            evaluator_id="command",
            metrics={str(key): float(value) for key, value in metrics.items()},
            artifacts={
                "stdout": str(Path(request.output_dir) / "evaluator.stdout.log"),
                "stderr": str(Path(request.output_dir) / "evaluator.stderr.log"),
            },
        )
