"""Execution-engine registry."""

from __future__ import annotations

from .protocol import EngineAdapter, EngineManifest


class EngineRegistryError(ValueError):
    pass


class EngineRegistry:
    def __init__(self, *, include_builtins: bool = True) -> None:
        self._engines: dict[str, EngineAdapter] = {}
        if include_builtins:
            from .delegated import DelegatedCommandEngine, VeOmniCommandEngine
            from .torch_engine import TorchEngine

            self.register(TorchEngine())
            for engine_id in ("delegated", "trl", "verl", "nemo"):
                self.register(DelegatedCommandEngine(engine_id))
            self.register(VeOmniCommandEngine())

    def register(self, engine: EngineAdapter) -> None:
        manifest = getattr(engine, "manifest", None)
        if not isinstance(manifest, EngineManifest):
            raise EngineRegistryError("engine must define EngineManifest")
        for method in ("validate", "prepare", "run", "checkpoint", "collect"):
            if not callable(getattr(engine, method, None)):
                raise EngineRegistryError(f"engine must implement {method}()")
        if manifest.engine_id in self._engines:
            raise EngineRegistryError(
                f"engine {manifest.engine_id!r} is already registered"
            )
        self._engines[manifest.engine_id] = engine

    def get(self, engine_id: str) -> EngineAdapter:
        try:
            return self._engines[engine_id]
        except KeyError as exc:
            raise EngineRegistryError(
                f"unknown engine {engine_id!r}; available: {sorted(self._engines)}"
            ) from exc

    def manifests(self) -> tuple[EngineManifest, ...]:
        return tuple(self._engines[key].manifest for key in sorted(self._engines))
