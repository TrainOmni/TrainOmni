"""Deterministic stage gate evaluation."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_type: str
    passed: bool
    message: str
    details: Mapping[str, Any]


_OPERATORS = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}


def evaluate_gate(
    spec: Mapping[str, Any], *, metrics: Mapping[str, float], artifacts: set[str]
) -> GateResult:
    gate_type = spec.get("type")
    if gate_type == "metric":
        expected = {"type", "metric", "op", "value"}
        _check_fields(spec, expected)
        metric = spec["metric"]
        op_name = spec["op"]
        if metric not in metrics:
            return GateResult(
                gate_type="metric",
                passed=False,
                message=f"metric {metric!r} is missing",
                details={"metric": metric},
            )
        if op_name not in _OPERATORS:
            raise ValueError(f"unsupported gate operator {op_name!r}")
        actual = metrics[metric]
        target = spec["value"]
        passed = bool(_OPERATORS[op_name](actual, target))
        return GateResult(
            gate_type="metric",
            passed=passed,
            message=f"{metric}={actual} {op_name} {target}: {passed}",
            details={"metric": metric, "actual": actual, "op": op_name, "target": target},
        )
    if gate_type == "artifact":
        expected = {"type", "artifact"}
        _check_fields(spec, expected)
        artifact = spec["artifact"]
        passed = artifact in artifacts
        return GateResult(
            gate_type="artifact",
            passed=passed,
            message=f"artifact {artifact!r} present: {passed}",
            details={"artifact": artifact},
        )
    if gate_type == "manual":
        expected = {"type", "approved", "reason"}
        _check_fields(spec, expected)
        passed = spec.get("approved") is True
        return GateResult(
            gate_type="manual",
            passed=passed,
            message=spec.get("reason", "manual approval required"),
            details={"approved": passed},
        )
    raise ValueError(f"unsupported gate type {gate_type!r}")


def _check_fields(spec: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"unknown gate fields: {sorted(unknown)}")
