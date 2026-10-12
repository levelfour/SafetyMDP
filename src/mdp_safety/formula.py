"""Safe parser and evaluator for polynomial safe-region formulae."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


class FormulaError(ValueError):
    """Raised when a formula is outside the supported safe subset."""


Exponent = tuple[int, ...]


def _clean(terms: dict[Exponent, float]) -> dict[Exponent, float]:
    return {powers: value for powers, value in terms.items() if abs(value) > 1e-15}


@dataclass(frozen=True)
class PolynomialFormula:
    variable_names: tuple[str, ...]
    terms: dict[Exponent, float]
    threshold: float
    expression: str
    convexity_assumed: bool = False

    @property
    def dimension(self) -> int:
        return len(self.variable_names)

    @property
    def degree(self) -> int:
        return max((sum(powers) for powers in self.terms), default=0)

    def value(self, point: np.ndarray) -> float:
        x = np.asarray(point, dtype=float)
        total = 0.0
        for powers, coefficient in self.terms.items():
            monomial = coefficient
            for coordinate, power in zip(x, powers):
                if power:
                    monomial *= coordinate**power
            total += monomial
        return float(total)

    def gradient(self, point: np.ndarray) -> np.ndarray:
        x = np.asarray(point, dtype=float)
        result = np.zeros(self.dimension)
        for powers, coefficient in self.terms.items():
            for index, power in enumerate(powers):
                if power == 0:
                    continue
                term = coefficient * power
                for other, other_power in enumerate(powers):
                    exponent = other_power - (1 if other == index else 0)
                    if exponent:
                        term *= x[other] ** exponent
                result[index] += term
        return result

    def hessian(self, point: np.ndarray) -> np.ndarray:
        x = np.asarray(point, dtype=float)
        result = np.zeros((self.dimension, self.dimension))
        for powers, coefficient in self.terms.items():
            for row in range(self.dimension):
                for column in range(self.dimension):
                    multiplier = powers[row] * (
                        powers[column] - (1 if row == column else 0)
                    )
                    if multiplier == 0:
                        continue
                    term = coefficient * multiplier
                    for index, power in enumerate(powers):
                        exponent = power
                        if index == row:
                            exponent -= 1
                        if index == column:
                            exponent -= 1
                        if exponent:
                            term *= x[index] ** exponent
                    result[row, column] += term
        return result

    def validate_shape_assumptions(self, tolerance: float = 1e-10) -> None:
        zero = (0,) * self.dimension
        for powers, coefficient in self.terms.items():
            if powers != zero and coefficient < -tolerance:
                raise FormulaError(
                    "downward closure requires nonnegative nonconstant polynomial "
                    "coefficients in the supported formula fragment"
                )
        if self.value(np.zeros(self.dimension)) > self.threshold + tolerance:
            raise FormulaError("the formula safe region must contain the origin")
        if self.convexity_assumed:
            return
        if self.degree <= 1:
            return
        if self.degree == 2:
            eigenvalues = np.linalg.eigvalsh(self.hessian(np.zeros(self.dimension)))
            if float(np.min(eigenvalues)) >= -tolerance:
                return
            raise FormulaError(
                "the quadratic formula is not convex (its Hessian is not positive semidefinite)"
            )
        separable = all(
            (powers == zero or coefficient >= -tolerance)
            and sum(power > 0 for power in powers) <= 1
            and (sum(powers) == 0 or sum(powers) >= 1)
            for powers, coefficient in self.terms.items()
        )
        if not separable:
            raise FormulaError(
                "convexity is not automatically decidable for this higher-degree "
                "polynomial; set assume_convex=true only when convexity is known"
            )


class _PolynomialParser:
    def __init__(
        self,
        variables: tuple[str, ...],
        parameters: Mapping[str, float],
        maximum_degree: int = 12,
    ) -> None:
        self.variables = variables
        self.variable_index = {name: index for index, name in enumerate(variables)}
        self.parameters = {}
        for name, value in parameters.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise FormulaError(f"parameter {name!r} must be finite")
            self.parameters[name] = numeric
        self.maximum_degree = maximum_degree
        overlap = set(self.variables) & set(self.parameters)
        if overlap:
            raise FormulaError(f"variables and parameters overlap: {sorted(overlap)}")

    def parse(self, node: ast.AST) -> dict[Exponent, float]:
        zero = (0,) * len(self.variables)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = float(node.value)
            if not math.isfinite(value):
                raise FormulaError("formula contains a non-finite number")
            return {zero: value}
        if isinstance(node, ast.Name):
            if node.id in self.parameters:
                return {zero: self.parameters[node.id]}
            if node.id not in self.variable_index:
                raise FormulaError(f"unknown name {node.id!r} in formula")
            powers = [0] * len(self.variables)
            powers[self.variable_index[node.id]] = 1
            return {tuple(powers): 1.0}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.parse(node.operand)
            if isinstance(node.op, ast.USub):
                value = {powers: -coefficient for powers, coefficient in value.items()}
            return _clean(value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = self.parse(node.left)
            right = self.parse(node.right)
            sign = -1.0 if isinstance(node.op, ast.Sub) else 1.0
            result = dict(left)
            for powers, coefficient in right.items():
                result[powers] = result.get(powers, 0.0) + sign * coefficient
            return _clean(result)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return self._multiply(self.parse(node.left), self.parse(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            numerator = self.parse(node.left)
            denominator = self.parse(node.right)
            if set(denominator) != {zero} or abs(denominator[zero]) <= 1e-15:
                raise FormulaError("division is only allowed by a nonzero constant")
            return {
                powers: coefficient / denominator[zero]
                for powers, coefficient in numerator.items()
            }
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base = self.parse(node.left)
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                raise FormulaError("polynomial powers must be nonnegative integer literals")
            exponent = int(node.right.value)
            if exponent < 0 or exponent > self.maximum_degree:
                raise FormulaError(
                    f"polynomial power must lie between 0 and {self.maximum_degree}"
                )
            result = {zero: 1.0}
            for _ in range(exponent):
                result = self._multiply(result, base)
            return result
        raise FormulaError(
            f"unsupported formula syntax: {ast.dump(node, include_attributes=False)}"
        )

    def _multiply(
        self,
        left: dict[Exponent, float],
        right: dict[Exponent, float],
    ) -> dict[Exponent, float]:
        result: dict[Exponent, float] = {}
        for left_powers, left_coefficient in left.items():
            for right_powers, right_coefficient in right.items():
                powers = tuple(a + b for a, b in zip(left_powers, right_powers))
                if sum(powers) > self.maximum_degree:
                    raise FormulaError(
                        f"expanded polynomial degree exceeds {self.maximum_degree}"
                    )
                result[powers] = (
                    result.get(powers, 0.0) + left_coefficient * right_coefficient
                )
        return _clean(result)


def parse_polynomial_formula(
    expression: str,
    variable_names: list[str] | tuple[str, ...],
    parameters: Mapping[str, float] | None = None,
    threshold: float | None = None,
    assume_convex: bool = False,
) -> PolynomialFormula:
    variables = tuple(variable_names)
    if not variables or len(set(variables)) != len(variables):
        raise FormulaError("formula variables must be a non-empty unique list")
    if not all(name.isidentifier() for name in variables):
        raise FormulaError("formula variables must be valid identifiers")
    normalized = expression.replace("≤", "<=").replace("^", "**")
    try:
        root = ast.parse(normalized, mode="eval").body
    except SyntaxError as error:
        raise FormulaError(f"invalid formula: {error.msg}") from error
    parser = _PolynomialParser(variables, parameters or {})
    if isinstance(root, ast.Compare):
        if (
            len(root.ops) != 1
            or len(root.comparators) != 1
            or not isinstance(root.ops[0], ast.LtE)
        ):
            raise FormulaError("formula must contain exactly one '<=' comparison")
        left_terms = parser.parse(root.left)
        right_terms = parser.parse(root.comparators[0])
        zero = (0,) * len(variables)
        if set(right_terms) != {zero}:
            raise FormulaError("the right-hand side of '<=' must be constant")
        parsed_threshold = right_terms[zero]
        if threshold is not None and not math.isclose(
            parsed_threshold, float(threshold), rel_tol=0.0, abs_tol=1e-12
        ):
            raise FormulaError("formula threshold conflicts with the separate threshold")
        threshold_value = parsed_threshold
    else:
        if threshold is None:
            raise FormulaError(
                "an expression without '<=' requires a separate numeric threshold"
            )
        left_terms = parser.parse(root)
        threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise FormulaError("formula threshold must be finite")
    formula = PolynomialFormula(
        variables,
        left_terms,
        threshold_value,
        expression,
        convexity_assumed=assume_convex,
    )
    formula.validate_shape_assumptions()
    return formula
