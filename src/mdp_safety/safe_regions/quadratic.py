"""Matrix-input quadratic geometry with SLSQP projections and supports."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from .base import ProjectionPoint, RegionError, Vector
from .sublevel import SmoothSublevelRegion

class QuadraticRegion(SmoothSublevelRegion):
    """Region ``x'Qx + c'x + constant <= threshold`` in the simplex."""

    def __init__(
        self,
        matrix: Any,
        linear: Any,
        constant: float,
        threshold: float,
        tolerance: float = 1e-9,
    ) -> None:
        q = np.asarray(matrix, dtype=float)
        c = np.asarray(linear, dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise RegionError("quadratic Q must be square")
        if c.shape != (q.shape[0],):
            raise RegionError("quadratic linear term must match Q")
        super().__init__(q.shape[0], tolerance)
        if not np.allclose(q, q.T, atol=tolerance):
            raise RegionError("quadratic Q must be symmetric")
        eigenvalues = np.linalg.eigvalsh(q)
        if float(np.min(eigenvalues)) < -tolerance:
            raise RegionError("quadratic Q must be positive semidefinite")
        if np.min(q) < -tolerance or np.min(c) < -tolerance:
            raise RegionError(
                "downward closure currently requires elementwise Q >= 0 and c >= 0"
            )
        if not np.isfinite(constant) or not np.isfinite(threshold):
            raise RegionError("quadratic constant and threshold must be finite")
        if constant > threshold + tolerance:
            raise RegionError("the origin must belong to the safe region")
        self.matrix = (q + q.T) / 2.0
        self.linear = np.maximum(c, 0.0)
        self.constant = float(constant)
        self.threshold = float(threshold)

    def cache_signature(self) -> object:
        return (self.dimension, self.tolerance, self.matrix.tobytes(),
                self.linear.tobytes(), self.constant, self.threshold)

    def value(self, point: Vector) -> float:
        x = self._check_point(point)
        return float(x @ self.matrix @ x + self.linear @ x + self.constant)

    def gradient(self, point: Vector) -> Vector:
        x = self._check_point(point)
        return 2.0 * self.matrix @ x + self.linear


    def project(self, point: Vector, norm: str) -> ProjectionPoint:
        p = self._check_point(point)
        if norm == "linf":
            return self._project_linf(p)
        initial = self._ray_start(p)
        if norm == "l2":
            objective = lambda q: 0.5 * float(np.dot(q - p, q - p))
            jacobian = lambda q: q - p
            result = minimize(
                objective,
                initial,
                jac=jacobian,
                constraints=self._constraints(),
                bounds=[(0.0, 1.0)] * self.dimension,
                method="SLSQP",
                options={"ftol": 1e-10, "maxiter": 3000},
            )
            if not result.success:
                raise RegionError(f"L2 projection failed: {result.message}")
            q = self._strictly_feasible_copy(result.x)
            return ProjectionPoint(q, float(np.linalg.norm(p - q)))
        raise RegionError(f"unsupported projection norm: {norm}")

    def maximize_linear(self, weight: Vector) -> tuple[float, Vector]:
        w = self._check_point(weight)
        result = minimize(
            lambda x: -float(np.dot(w, x)),
            self._ray_start(w),
            jac=lambda x: -w,
            constraints=self._constraints(),
            bounds=[(0.0, 1.0)] * self.dimension,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 3000},
        )
        if not result.success:
            raise RegionError(f"safe-region support optimization failed: {result.message}")
        point = self._strictly_feasible_copy(result.x)
        return float(-result.fun), point

    def supporting_normal(self, point: Vector, projected: Vector, norm: str) -> Vector:
        return self._boundary_normal(point, projected, norm)
