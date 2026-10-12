"""Construct a safe-region implementation from the public JSON schema."""

from __future__ import annotations

from typing import Any

from ..formula import FormulaError, parse_polynomial_formula
from .base import RegionError, SafeRegion
from .polytope import PolytopeRegion
from .polynomial import FormulaRegion
from .quadratic import QuadraticRegion

def region_from_dict(
    specification: dict[str, Any],
    tolerance: float = 1e-9,
    dimension_hint: int | None = None,
) -> SafeRegion:
    kind = str(specification.get("type", "")).lower()
    if kind == "polytope":
        return PolytopeRegion(specification.get("A"), specification.get("b"), tolerance)
    if kind in {"quadratic", "quadratic_sublevel"}:
        matrix = specification.get("Q")
        if matrix is None:
            raise RegionError("quadratic region requires Q")
        dimension = len(matrix)
        return QuadraticRegion(
            matrix,
            specification.get("linear", [0.0] * dimension),
            float(specification.get("constant", 0.0)),
            float(specification["threshold"]),
            tolerance,
        )
    if kind in {"formula", "polynomial", "polynomial_formula"}:
        expression = specification.get("formula", specification.get("expression"))
        if not isinstance(expression, str) or not expression.strip():
            raise RegionError("formula region requires a non-empty formula string")
        variables = specification.get("variables")
        if variables is None:
            if dimension_hint is None:
                raise RegionError(
                    "formula region requires variables or a dimension supplied by terminal labels"
                )
            variables = [f"x{index + 1}" for index in range(dimension_hint)]
        elif not isinstance(variables, (list, tuple)):
            raise RegionError("formula variables must be a JSON array of names")
        try:
            formula = parse_polynomial_formula(
                expression,
                variables,
                parameters=specification.get("parameters", {}),
                threshold=specification.get("threshold"),
                assume_convex=bool(specification.get("assume_convex", False)),
            )
        except FormulaError as error:
            raise RegionError(str(error)) from error
        return FormulaRegion(formula, tolerance)
    raise RegionError(
        f"unknown safe-region type {kind!r}; expected 'polytope', 'quadratic', or 'formula'"
    )
