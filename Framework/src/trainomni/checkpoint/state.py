"""Versioned state registry for exact local resume."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

STATE_REGISTRY_VERSION = "trainomni.state-registry.v1"


class Stateful(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class StateRegistryError(ValueError):
    pass


class StateRegistry:
    def __init__(self) -> None:
        self._objects: dict[str, Stateful] = {}

    def register(self, name: str, value: Stateful) -> None:
        if not name or any(part in name for part in ("/", "\\", "..")):
            raise StateRegistryError(f"invalid state name {name!r}")
        if name in self._objects:
            raise StateRegistryError(f"state object {name!r} is already registered")
        if not callable(getattr(value, "state_dict", None)) or not callable(
            getattr(value, "load_state_dict", None)
        ):
            raise StateRegistryError(
                f"state object {name!r} must implement state_dict/load_state_dict"
            )
        self._objects[name] = value

    def state_dict(self) -> dict[str, Any]:
        return {
            "registry_version": STATE_REGISTRY_VERSION,
            "objects": {
                name: dict(value.state_dict())
                for name, value in sorted(self._objects.items())
            },
        }

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        if state.get("registry_version") != STATE_REGISTRY_VERSION:
            raise StateRegistryError("state registry version mismatch")
        objects = state.get("objects")
        if not isinstance(objects, Mapping):
            raise StateRegistryError("state registry objects must be a mapping")
        expected = set(self._objects)
        actual = set(objects)
        if strict and actual != expected:
            raise StateRegistryError(
                f"state registry names mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        for name in sorted(expected & actual):
            value = objects[name]
            if not isinstance(value, Mapping):
                raise StateRegistryError(f"state for {name!r} must be a mapping")
            self._objects[name].load_state_dict(value)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._objects))


class PythonRandomState:
    """Capture the global Python random generator."""

    def state_dict(self) -> Mapping[str, Any]:
        return {"state": random.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "state" not in state:
            raise StateRegistryError("Python RNG state is missing 'state'")
        random.setstate(state["state"])


@dataclass(slots=True)
class ScalarState:
    """Small mutable scalar used for steps, microsteps and consumed budgets."""

    value: int | float = 0

    def state_dict(self) -> Mapping[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        value = state.get("value")
        if not isinstance(value, (int, float)):
            raise StateRegistryError("scalar state value must be numeric")
        self.value = value


class TorchRandomState:
    """Optional PyTorch CPU/device RNG state without importing torch at package load."""

    def state_dict(self) -> Mapping[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise StateRegistryError("Torch RNG state requires the 'torch' extra") from exc
        value: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            value["cuda"] = torch.cuda.get_rng_state_all()
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            value["npu"] = npu.get_rng_state_all()
        return value

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise StateRegistryError("Torch RNG state requires the 'torch' extra") from exc
        if "cpu" not in state:
            raise StateRegistryError("Torch RNG state is missing CPU state")
        torch.random.set_rng_state(state["cpu"])
        if "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
        if "npu" in state:
            torch.npu.set_rng_state_all(state["npu"])


@dataclass(slots=True)
class ObjectState:
    """Register any PyTorch-style object exposing state_dict/load_state_dict."""

    value: Any

    def state_dict(self) -> Mapping[str, Any]:
        state = self.value.state_dict()
        if not isinstance(state, Mapping):
            raise StateRegistryError("object state_dict() must return a mapping")
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.value.load_state_dict(state)
