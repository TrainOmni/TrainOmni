from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class MergerConfig:
    input_dim: int = 768
    output_dim: int = 1536
    spatial_merge_size: int = 2
    seed: int = 20260904
    initializer_range: float = 0.02
