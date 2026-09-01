"""Framework error hierarchy."""


class TrainOmniError(Exception):
    """Base class for expected framework failures."""


class SpecError(TrainOmniError):
    """Raised when task or run configuration is invalid."""


class RegistryError(TrainOmniError):
    """Raised for duplicate, missing, or incompatible modules."""


class CapabilityError(TrainOmniError):
    """Raised when a resolved module graph cannot satisfy its requirements."""


class ObjectiveError(TrainOmniError):
    """Raised when supervision or loss computation is invalid."""


class OptimizationError(TrainOmniError):
    """Raised when gradients or optimizer state are invalid."""


class CheckpointError(TrainOmniError):
    """Raised when checkpoint identity or state validation fails."""
