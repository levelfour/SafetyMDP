"""Polyhedral safe regions and their optimized projection kernels."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize, nnls

from .base import ProjectionPoint, RegionError, SafeRegion, Vector

class PolytopeRegion(SafeRegion):
    """Region ``{x in simplex | A x <= b}`` with nonnegative rows of A."""

    def __init__(self, matrix: Any, bounds: Any, tolerance: float = 1e-9) -> None:
        a = np.asarray(matrix, dtype=float)
        b = np.asarray(bounds, dtype=float)
        if a.ndim != 2 or a.shape[0] == 0:
            raise RegionError("polytope A must be a non-empty 2-D array")
        if b.shape != (a.shape[0],):
            raise RegionError("polytope b must have one value per row of A")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise RegionError("polytope contains a non-finite coefficient")
        super().__init__(a.shape[1], tolerance)
        if np.min(a) < -tolerance:
            raise RegionError(
                "downward closure requires nonnegative polytope coefficients"
            )
        if np.min(b) < -tolerance:
            raise RegionError("the origin must belong to the safe region (require b >= 0)")
        self.matrix = np.maximum(a, 0.0)
        self.bounds = b

    def cache_signature(self) -> object:
        return (self.dimension, self.tolerance, self.matrix.shape,
                self.matrix.tobytes(), self.bounds.tobytes())

    def projection_scores(self, points: Vector, norm: str) -> Vector | None:
        if norm != "linf":
            return None
        values = self._check_gauge_points(points)
        # For a downward-closed region, a point within L-infinity distance t
        # exists iff (p-t)_+ is feasible. Sort p once and solve each piecewise
        # linear inequality using weighted prefix sums. Include the simplex.
        matrix = np.vstack((self.matrix, np.ones((1, self.dimension))))
        bounds = np.r_[self.bounds, 1.0]
        if np.any(bounds < 0):
            return None
        result = np.zeros(len(values))
        chunk = max(1, 1_000_000 // (len(matrix) * self.dimension))
        for start in range(0, len(values), chunk):
            p = values[start:start + chunk]
            order = np.argsort(-p, axis=1, kind="stable")
            sorted_p = np.take_along_axis(p, order, axis=1)
            sorted_a = np.take(matrix, order, axis=1).transpose(1, 0, 2)
            denominator = np.cumsum(sorted_a, axis=2)
            numerator = np.cumsum(sorted_a * sorted_p[:, None, :], axis=2) - bounds[None, :, None]
            ratios = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
            result[start:start + len(p)] = np.maximum(0.0, np.max(ratios, axis=(1, 2)))
        return result

    def l2_distance_bounds(self, points: Vector) -> tuple[Vector, Vector]:
        values = self._check_gauge_points(points)
        matrix = np.vstack((self.matrix, np.ones((1, self.dimension))))
        bounds = np.r_[self.bounds, 1.0]
        norms = np.linalg.norm(matrix, axis=1)
        keep = norms > 0
        lower = np.maximum(0, np.max((values @ matrix[keep].T - bounds[keep]) / norms[keep], axis=1))
        scores = self.projection_scores(values, "linf")
        if scores is None:
            return np.zeros(len(values)), np.full(len(values), np.inf)
        # Increase t slightly: the resulting downward-shifted point is a
        # feasible primal upper bound, including when a facet has b=0.
        padding = 1e-10 * np.maximum(1.0, np.max(values, axis=1))
        feasible = np.maximum(values - (scores + padding)[:, None], 0.0)
        upper = np.linalg.norm(values - feasible, axis=1) + padding
        return np.maximum(0, lower - padding), upper

    def l2_dual_lower_bound(self, point: Vector, projected: Vector) -> float:
        # Any nonnegative multiplier is dual feasible; NNLS only finds a
        # useful one. Never use the approximate primal distance as a lower
        # bound when pruning another vertex.
        matrix, bounds = self._linear_inequalities()
        norms = np.linalg.norm(matrix, axis=1)
        keep = norms > 0
        matrix, bounds = matrix[keep] / norms[keep, None], bounds[keep] / norms[keep]
        active = bounds - matrix @ projected <= max(1e-7, 100 * self.tolerance)
        if not np.any(active):
            return 0.0
        matrix, bounds = matrix[active], bounds[active]
        try:
            multipliers, _ = nnls(matrix.T, point - projected, maxiter=1000)
        except RuntimeError:
            return 0.0
        normal = matrix.T @ multipliers
        linear = float(multipliers @ (matrix @ point - bounds))
        squared = float(normal @ normal)
        error = 1e-12 * (1.0 + abs(linear) + squared)
        return float(np.sqrt(max(0.0, 2 * linear - squared - error)))

    def absorption_radius(self) -> float:
        # rho*Delta satisfies a_i*x <= rho*max_j(a_ij).
        maxima = np.max(self.matrix, axis=1)
        nonzero = maxima > 0.0
        if not np.any(nonzero):
            return 1.0
        return float(max(0.0, min(1.0, np.min(self.bounds[nonzero] / maxima[nonzero]))))

    def gauge_scores(self, points: Vector) -> Vector:
        values = self._check_gauge_points(points)
        if self.absorption_radius() <= max(1e-12, 100.0 * self.tolerance):
            raise RegionError(
                "gauge selection requires S to contain rho times the simplex for some rho > 0"
            )
        # Include the simplex facet even when it is not present in A. Ignore
        # redundant zero rows, including 0*x <= 0, without dividing by zero.
        nonzero = np.any(self.matrix > 0.0, axis=1)
        matrix = self.matrix[nonzero]
        bounds = self.bounds[nonzero]
        scores = np.sum(values, axis=1)
        if len(matrix):
            # Bound temporary memory for high-dimensional outer polytopes.
            chunk_size = max(1, 1_000_000 // len(matrix))
            for start in range(0, len(values), chunk_size):
                stop = start + chunk_size
                ratios = (values[start:stop] @ matrix.T) / bounds
                scores[start:stop] = np.maximum(scores[start:stop], np.max(ratios, axis=1))
        return scores

    def gauge(self, point: Vector) -> tuple[float, Vector]:
        p = self._check_point(point)
        gamma = float(self.gauge_scores(p[None, :])[0])
        if gamma == 0.0:
            return 0.0, np.zeros(self.dimension)
        # Round outward until the radial point passes the original membership
        # check; do not use an approximate boundary as a safety certificate.
        for _ in range(16):
            boundary = p / gamma
            if self.contains(boundary, tolerance=0.0):
                return gamma, boundary
            gamma = float(np.nextafter(gamma, np.inf))
        # Unusual roundoff still has the validated generic implementation.
        return self._gauge_with_radius(p, self.absorption_radius())

    def contains(self, point: Vector, tolerance: float | None = None) -> bool:
        x = self._check_point(point)
        tol = self.tolerance if tolerance is None else tolerance
        return bool(
            np.min(x) >= -tol
            and np.max(x) <= 1.0 + tol
            and np.sum(x) <= 1.0 + tol
            and np.max(self.matrix @ x - self.bounds) <= tol
        )

    def violation(self, point: Vector) -> float:
        x = self._check_point(point)
        return float(
            max(
                -float(np.min(x)),
                float(np.max(x - 1.0)),
                float(np.sum(x) - 1.0),
                float(np.max(self.matrix @ x - self.bounds)),
            )
        )

    def _linear_inequalities(self) -> tuple[Vector, Vector]:
        identity = np.eye(self.dimension)
        a = np.vstack((self.matrix, identity, -identity, np.ones((1, self.dimension))))
        b = np.concatenate((self.bounds, np.ones(self.dimension), np.zeros(self.dimension), [1.0]))
        return a, b

    def project(self, point: Vector, norm: str) -> ProjectionPoint:
        p = self._check_point(point)
        a, b = self._linear_inequalities()
        if norm == "linf":
            # Variables are (q_1, ..., q_d, t).  Because the region is
            # downward closed, an optimum with q <= p always exists.
            objective = np.zeros(self.dimension + 1)
            objective[-1] = 1.0
            upper_dev = np.hstack((-np.eye(self.dimension), -np.ones((self.dimension, 1))))
            lower_dev = np.hstack((np.eye(self.dimension), -np.ones((self.dimension, 1))))
            lp_a = np.vstack((np.hstack((a, np.zeros((a.shape[0], 1)))), upper_dev, lower_dev))
            lp_b = np.concatenate((b, -p, p))
            result = linprog(
                objective,
                A_ub=lp_a,
                b_ub=lp_b,
                bounds=[(None, None)] * self.dimension + [(0.0, None)],
                method="highs",
            )
            if not result.success:
                raise RegionError(f"L-infinity projection failed: {result.message}")
            # A canonical nearest point has q_i = max(p_i-t, 0). Unlike an
            # arbitrary LP optimum this exposes all saturated coordinates.
            q = self._strictly_feasible_copy(np.maximum(p - max(0.0, result.x[-1]), 0.0))
            return ProjectionPoint(q, float(np.max(np.abs(p - q))))

        if norm != "l2":
            raise RegionError(f"unsupported projection norm: {norm}")
        constraints = LinearConstraint(a, -np.inf, b)
        initial = np.minimum(np.maximum(p, 0.0), 1.0)
        if np.sum(initial) > 1.0:
            initial /= np.sum(initial)
        result = minimize(
            lambda q: 0.5 * float(np.dot(q - p, q - p)),
            initial,
            jac=lambda q: q - p,
            constraints=[constraints],
            method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 2000},
        )
        if not result.success:
            raise RegionError(f"L2 projection failed: {result.message}")
        q = self._strictly_feasible_copy(result.x)
        return ProjectionPoint(q, float(np.linalg.norm(p - q)))

    def maximize_linear(self, weight: Vector) -> tuple[float, Vector]:
        w = self._check_point(weight)
        a, b = self._linear_inequalities()
        result = linprog(-w, A_ub=a, b_ub=b, bounds=[(None, None)] * self.dimension, method="highs")
        if not result.success:
            raise RegionError(f"safe-region support optimization failed: {result.message}")
        point = self._strictly_feasible_copy(result.x)
        # The LP optimum is useful for diagnostics.  The returned feasible point
        # is used as the numerically robust lower certificate.
        return float(-result.fun), point

    def supporting_normal(self, point: Vector, projected: Vector, norm: str) -> Vector:
        p = self._check_point(point)
        q = self._check_point(projected)
        if norm == "l2":
            return np.maximum(p - q, 0.0)
        if norm not in {"linf", "gauge"}:
            raise RegionError(f"unsupported projection norm: {norm}")
        matrix = np.vstack((self.matrix, np.ones((1, self.dimension))))
        bounds = np.r_[self.bounds, 1.0]
        nonzero = np.max(matrix, axis=1) > 0
        matrix, bounds = matrix[nonzero], bounds[nonzero]
        # Work with normalized rows: rescaling an input inequality must not
        # change which (possibly only approximately active) face is chosen.
        scales = np.max(matrix, axis=1)
        matrix, bounds = matrix / scales[:, None], bounds / scales
        if norm == "gauge":
            if np.any(bounds <= 0):
                raise RegionError("gauge requires positive bounds on nonzero rows")
            # Pick a genuinely maximal radial ratio, not largest raw violation
            # among loosely active faces. Redundant rows and the simplex are OK.
            return matrix[int(np.argmax((matrix @ p) / bounds))].copy()
        delta = float(np.max(np.abs(p - q)))
        tolerance = max(1e-9, 100 * self.tolerance)
        active = np.abs(matrix @ q - bounds) <= tolerance
        saturated = p - q >= delta - tolerance
        best, best_error = None, np.inf
        for row in matrix[active]:
            w = row.copy()
            # At q_i=0 we may subtract a lower-orthant normal and still
            # support S0. This is essential when p_i < delta.
            w[(q <= tolerance) & ~saturated] = 0.0
            if np.any(w[~saturated] > tolerance) or np.max(w) <= 0:
                continue
            w[~saturated] = 0.0
            error = abs(float(w @ (p - q)) / float(np.sum(w)) - delta)
            if error < best_error:
                best, best_error = w, error
        if best is None:
            raise RegionError("L-infinity projection has no normal satisfying Section 3 equation (6)")
        return best
