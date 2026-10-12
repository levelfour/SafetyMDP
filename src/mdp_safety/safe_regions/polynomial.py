"""Polynomial sublevel geometry and checked numerical boundary recovery."""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult, linprog, minimize

from ..formula import PolynomialFormula
from .base import ProjectionPoint, RegionError, Vector
from .sublevel import SmoothSublevelRegion

class FormulaRegion(SmoothSublevelRegion):
    """Polynomial sublevel region entered as a mathematical formula."""

    def __init__(self, formula: PolynomialFormula, tolerance: float = 1e-9) -> None:
        super().__init__(formula.dimension, tolerance)
        self.formula = formula
        self.threshold = formula.threshold

    def cache_signature(self) -> object:
        return (self.dimension, self.tolerance, tuple(sorted(self.formula.terms.items())), self.threshold)

    def value(self, point: Vector) -> float:
        return self.formula.value(self._check_point(point))

    def gradient(self, point: Vector) -> Vector:
        return self.formula.gradient(self._check_point(point))


    def _accept_slsqp_point(
        self, candidate: Vector, message: str, operation: str
    ) -> Vector:
        """Recover feasibility, without claiming that the point is optimal."""
        candidate = np.asarray(candidate, dtype=float)
        recovery_tolerance = max(1e-6, 1000.0 * self.tolerance)
        if (
            candidate.shape != (self.dimension,)
            or not np.all(np.isfinite(candidate))
            or self.violation(candidate) > recovery_tolerance
        ):
            raise RegionError(f"{operation} failed: {message}")
        try:
            return self._strictly_feasible_copy(candidate)
        except RegionError as error:
            raise RegionError(f"{operation} failed: {message}") from error

    def project(self, point: Vector, norm: str) -> ProjectionPoint:
        p = self._check_point(point)
        if norm == "linf":
            return self._project_linf(p)
        initial = self._ray_start(p)
        if norm == "l2":
            result = minimize(
                lambda q: 0.5 * float(np.dot(q - p, q - p)),
                initial,
                jac=lambda q: q - p,
                constraints=self._constraints(),
                bounds=[(0.0, 1.0)] * self.dimension,
                method="SLSQP",
                options={"ftol": 1e-10, "maxiter": 3000},
            )
            q = self._accept_slsqp_point(result.x, result.message, "L2 projection")
            return ProjectionPoint(q, float(np.linalg.norm(p - q)))
        raise RegionError(f"unsupported projection norm: {norm}")

    def maximize_linear(self, weight: Vector) -> tuple[float, Vector]:
        original_weight = self._check_point(weight)
        scale = float(np.max(np.abs(original_weight)))
        if scale == 0.0:
            return 0.0, np.zeros(self.dimension)
        w = original_weight / scale
        result = minimize(
            lambda x: -float(np.dot(w, x)),
            self._ray_start(w),
            jac=lambda x: -w,
            constraints=self._constraints(),
            bounds=[(0.0, 1.0)] * self.dimension,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 3000},
        )
        # Audit support optimality in normalized weight units before the
        # separate direction-score check. Gauge's direction check also
        # divides by the safe-support denominator.
        allowed_gap = max(5e-8, 10.0 * self.tolerance)
        # Neither success=True nor feasibility alone establishes optimality.
        # A convex tangent relaxation provides an independent support-gap test.
        for attempt in range(2):
            try:
                point = self._accept_slsqp_point(
                    result.x, result.message, "safe-region support optimization"
                )
                gap = self._support_gap(w, point)
                if gap <= allowed_gap:
                    return float(np.dot(original_weight, point)), point
            except RegionError:
                gap = float("inf")
            if attempt == 0:
                result = self._scaled_support_solve(w)
        raise RegionError(
            "safe-region support optimality could not be validated "
            f"(gap={gap:.3g}, tolerance={allowed_gap:.3g}; {result.message})"
        )

    def _support_gap(self, weight: Vector, point: Vector) -> float:
        """Numerical primal/dual gap using the convex tangent relaxation.

        Convexity implies S lies in {x>=0, sum(x)<=1, g*x<=b}, with
        g=grad f(point), b=theta-f(point)+g*point. Any lambda,mu>=0
        satisfying lambda*g+mu>=weight yields the upper bound lambda*b+mu.
        Recompute mu from lambda to remove LP stationarity residuals. This is
        a floating-point audit, not an interval/exact-arithmetic certificate.
        """
        gradient = self.gradient(point)
        gradient_scale = float(np.max(np.abs(gradient)))
        if gradient_scale == 0.0:
            return max(0.0, float(np.max(weight))) - float(weight @ point)
        normal = gradient / gradient_scale
        bound = (self.threshold - self.value(point)) / gradient_scale + float(normal @ point)
        if not np.isfinite(bound) or bound < 0.0:
            return float("inf")
        dual = linprog(
            [bound, 1.0], A_ub=-np.column_stack((normal, np.ones(self.dimension))),
            b_ub=-weight, bounds=[(0.0, None), (0.0, None)], method="highs",
        )
        if not dual.success or not np.all(np.isfinite(dual.x)):
            return float("inf")
        multiplier = max(0.0, float(dual.x[0]))
        simplex_multiplier = max(0.0, float(np.max(weight - multiplier * normal)))
        upper = multiplier * bound + simplex_multiplier
        lower = float(weight @ point)
        roundoff = 64.0 * np.finfo(float).eps * max(1.0, abs(upper), abs(lower))
        return max(0.0, upper - lower) + roundoff

    def _scaled_support_solve(self, weight: Vector) -> OptimizeResult:
        """Retry in axis-radius coordinates; do not change the optimization.

        The safe region is downward closed, so these radii bound every feasible
        coordinate. Unit box bounds avoid heterogeneous coordinate scales that
        can make the unscaled SLSQP line search fail.
        """
        radii = np.array([
            self._strictly_feasible_copy(axis)[index]
            for index, axis in enumerate(np.eye(self.dimension))
        ])
        # Round upper coordinate bounds outward: feasible endpoints alone
        # would shave a floating-point sliver off the original feasible set.
        radii = np.minimum(1.0, radii + 8.0 * np.finfo(float).eps)
        zero = np.zeros(self.dimension)
        constraint_scale = max(abs(self.threshold - self.value(zero)), 1e-12)
        objective = weight * radii
        objective_scale = max(float(np.max(np.abs(objective))), 1e-300)
        result = minimize(
            lambda y: -float(objective @ y) / objective_scale,
            np.full(self.dimension, 0.25 / self.dimension),
            jac=lambda y: -objective / objective_scale,
            constraints=[
                {"type": "ineq", "fun": lambda y: 1.0 - float(radii @ y),
                 "jac": lambda y: -radii},
                {"type": "ineq", "fun": lambda y: (self.threshold - self.value(radii * y)) / constraint_scale,
                 "jac": lambda y: -self.gradient(radii * y) * radii / constraint_scale},
            ],
            bounds=[(0.0, 1.0)] * self.dimension, method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 3000},
        )
        result.x = radii * result.x
        return result

    def supporting_normal(self, point: Vector, projected: Vector, norm: str) -> Vector:
        return self._boundary_normal(point, projected, norm)
