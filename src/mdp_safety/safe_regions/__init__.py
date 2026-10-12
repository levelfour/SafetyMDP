"""Safe-region types and JSON construction."""

from .base import ProjectionPoint, RegionError, SafeRegion, Vector
from .factory import region_from_dict
from .polynomial import FormulaRegion
from .polytope import PolytopeRegion
from .quadratic import QuadraticRegion

__all__ = [
    "FormulaRegion", "PolytopeRegion", "ProjectionPoint", "QuadraticRegion",
    "RegionError", "SafeRegion", "Vector", "region_from_dict",
]
