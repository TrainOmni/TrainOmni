"""Stable cross-module value objects."""

from .artifact import ArtifactIdentity
from .batch import EncodedSample, OmniBatch, SupervisedExample
from .distribution import DistributionHints, distribution_hints
from .features import ModalFeatureBranch, ModalFeatures, ModalFeatureSet
from .forward import ForwardPlan, ForwardRequest, ForwardResult, OutputRequirements
from .loss import LossBundle, LossTerm
from .sample import ContentBlock, Message, OmniSample

__all__ = [
    "ArtifactIdentity",
    "ContentBlock",
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
    "OmniBatch",
    "OmniSample",
    "OutputRequirements",
    "SupervisedExample",
    "distribution_hints",
]
