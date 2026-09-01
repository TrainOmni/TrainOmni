"""Stable cross-module value objects."""

from .artifact import ArtifactIdentity
from .batch import EncodedSample, OmniBatch, SupervisedExample
from .cache import current_model_inputs_field, digest_tensor, model_inputs_digest
from .data import DataRecord
from .distribution import DistributionHints, distribution_hints
from .features import ModalFeatureBranch, ModalFeatures, ModalFeatureSet
from .forward import ForwardPlan, ForwardRequest, ForwardResult, OutputRequirements
from .loss import LossBundle, LossTerm, ObjectiveMetric
from .sample import ContentBlock, Message, OmniSample

__all__ = [
    "ArtifactIdentity",
    "ContentBlock",
    "DataRecord",
    "DistributionHints",
    "EncodedSample",
    "ForwardPlan",
    "ForwardRequest",
    "ForwardResult",
    "LossBundle",
    "LossTerm",
    "Message",
    "ModalFeatureBranch",
    "ModalFeatureSet",
    "ModalFeatures",
    "ObjectiveMetric",
    "OmniBatch",
    "OmniSample",
    "OutputRequirements",
    "SupervisedExample",
    "current_model_inputs_field",
    "digest_tensor",
    "distribution_hints",
    "model_inputs_digest",
]
