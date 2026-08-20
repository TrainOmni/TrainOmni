"""TrainOmni framework core package."""

from .data import (
    CanonicalSample,
    SampleValidationError,
    canonical_hash,
    canonical_json,
    load_sample,
    parse_sample,
    validate_sample_dict,
)

__all__ = [
    "CanonicalSample",
    "SampleValidationError",
    "canonical_hash",
    "canonical_json",
    "load_sample",
    "parse_sample",
    "validate_sample_dict",
]

__version__ = "1.0.0"
