"""Stable, backend-neutral TrainOmni contracts."""

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    RESUME_LEVELS,
    ArtifactManifest,
    ArtifactRef,
)
from .batching import BatchBudget, BatchItem, BatchPlan, CostVector
from .issues import IssueSeverity, ValidationIssue, ValidationReport

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "RESUME_LEVELS",
    "ArtifactManifest",
    "ArtifactRef",
    "BatchBudget",
    "BatchItem",
    "BatchPlan",
    "CostVector",
    "IssueSeverity",
    "ValidationIssue",
    "ValidationReport",
]
