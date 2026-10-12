"""Public safe-region API; implementations live in safe_regions."""

from .formula import FormulaError, PolynomialFormula, parse_polynomial_formula
from .safe_regions import (
    FormulaRegion, PolytopeRegion, ProjectionPoint, QuadraticRegion,
    RegionError, SafeRegion, Vector, region_from_dict,
)

__all__ = [
    "FormulaError", "FormulaRegion", "PolynomialFormula", "PolytopeRegion",
    "ProjectionPoint", "QuadraticRegion", "RegionError", "SafeRegion", "Vector",
    "parse_polynomial_formula", "region_from_dict",
]
