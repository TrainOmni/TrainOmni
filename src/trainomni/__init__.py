"""TrainOmni modular multimodal-training framework."""

__version__ = "0.1.1"

# A committed semantic provenance identifier for builtin Framework code.  It is
# deliberately independent of a Git checkout or dirty working tree so packaged
# wheels and source installs produce the same identity.  Any builtin behavior
# change that can affect assembly, training, checkpointing, evaluation, or
# export must advance both this identifier and the package version.
BUILTIN_CODE_PROVENANCE = "trainomni-builtin-core-v1.1-20260901"
