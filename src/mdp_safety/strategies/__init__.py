"""Pluggable weight-selection strategies."""

from .base import ProjectionStrategy, Separation
from .gauge import GaugeStrategy
from .norm import NormProjectionStrategy
from .registry import make_projection, projection_names, register_projection
from .sandwich import inner_approximation_contains
from .sandwich_lk import LammelKueferStrategy, LammelKueferGaugeStrategy

__all__ = [
    "GaugeStrategy", "NormProjectionStrategy",
    "ProjectionStrategy", "Separation",
    "inner_approximation_contains", "make_projection", "projection_names",
    "register_projection",
    "LammelKueferStrategy", "LammelKueferGaugeStrategy",
]
