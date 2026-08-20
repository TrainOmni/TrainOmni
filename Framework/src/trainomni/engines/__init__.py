"""Swappable execution-engine contracts."""

from .delegated import (
    VEOMNI_BRIDGE_API_VERSION,
    DelegatedCommandEngine,
    DelegatedEngineError,
    DelegatedStageContext,
    VeOmniCommandEngine,
)
from .protocol import (
    ENGINE_API_VERSION,
    EngineAdapter,
    EngineCapabilities,
    EngineKind,
    EngineManifest,
    EngineRequirements,
    PreparedStage,
    StageResult,
    negotiate_engine,
)
from .registry import EngineRegistry, EngineRegistryError
from .torch_engine import TorchEngine, TorchEngineError, TorchStageContext

__all__ = [
    "ENGINE_API_VERSION",
    "VEOMNI_BRIDGE_API_VERSION",
    "DelegatedCommandEngine",
    "DelegatedEngineError",
    "DelegatedStageContext",
    "EngineAdapter",
    "EngineCapabilities",
    "EngineKind",
    "EngineManifest",
    "EngineRegistry",
    "EngineRegistryError",
    "EngineRequirements",
    "PreparedStage",
    "StageResult",
    "TorchEngine",
    "TorchEngineError",
    "TorchStageContext",
    "VeOmniCommandEngine",
    "negotiate_engine",
]
