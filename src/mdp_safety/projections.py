"""Public weight-strategy API; implementations live in strategies."""

from .strategies import (
    GaugeStrategy, NormProjectionStrategy,
    ProjectionStrategy, Separation, inner_approximation_contains,
    make_projection, projection_names, register_projection,
    LammelKueferStrategy, LammelKueferGaugeStrategy,
)
from .strategies.registry import _STRATEGIES

__all__ = [
    "GaugeStrategy", "NormProjectionStrategy",
    "ProjectionStrategy", "Separation",
    "inner_approximation_contains", "make_projection", "projection_names",
    "register_projection",
    "LammelKueferStrategy", "LammelKueferGaugeStrategy",
]
