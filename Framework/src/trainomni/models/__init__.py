"""Model-family plugin contracts and checkpoint inspection."""

from .bundle import ModelBuildContext, ModelBundle
from .conformance import (
    REQUIRED_PLUGIN_METHODS,
    validate_plugin_components,
    validate_plugin_shape,
)
from .io import (
    EncodedSample,
    ModelBatch,
    SourceSpan,
    inspect_encoded_sample,
    inspect_model_batch,
    summarize_value,
)
from .probe import (
    CheckpointProbe,
    CompositeCompatibility,
    ProbeError,
    TensorInfo,
    analyze_composite,
    probe_checkpoint,
)
from .protocol import (
    MODEL_PLUGIN_API_VERSION,
    CapabilityIssue,
    CapabilityReport,
    ComponentCatalog,
    ComponentRule,
    ModelCapabilities,
    ModelFamilyPlugin,
    ModelPluginManifest,
    ModelRequirements,
    negotiate_capabilities,
)

__all__ = [
    "MODEL_PLUGIN_API_VERSION",
    "REQUIRED_PLUGIN_METHODS",
    "CapabilityIssue",
    "CapabilityReport",
    "CheckpointProbe",
    "ComponentCatalog",
    "ComponentRule",
    "CompositeCompatibility",
    "EncodedSample",
    "ModelBatch",
    "ModelBuildContext",
    "ModelBundle",
    "ModelCapabilities",
    "ModelFamilyPlugin",
    "ModelPluginManifest",
    "ModelRequirements",
    "ProbeError",
    "SourceSpan",
    "TensorInfo",
    "analyze_composite",
    "inspect_encoded_sample",
    "inspect_model_batch",
    "negotiate_capabilities",
    "probe_checkpoint",
    "summarize_value",
    "validate_plugin_components",
    "validate_plugin_shape",
]
