"""Shared simplex geometry for differentiable convex sublevel regions."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from .base import ProjectionPoint, RegionError, SafeRegion, Vector

class SmoothSublevelRegion(SafeRegion):
    """Common membership and feasible initialization, not a solver policy."""

    @abstractmethod
    def value(self, point: Vector) -> float:
        pass

    @abstractmethod
    def gradient(self, point: Vector) -> Vector:
        pass

    def _project_linf(self, point: Vector) -> ProjectionPoint:
        """Monotonicity reduces the LP/convex program to one scalar root.

        dist_inf(p,S0) <= t iff (p-t)_+ belongs to S0, for p >= 0.
        Bisection retains a feasible endpoint of this monotone membership test.
        """
        p = self._check_point(point)
        if np.any(p < 0):
            raise RegionError("monotone L-infinity projection requires a nonnegative point")
        if self.contains(p, tolerance=0.0):
            return ProjectionPoint(p.copy(), 0.0)
        low, high = 0.0, float(np.max(p))
        if not self.contains(np.zeros(self.dimension), tolerance=0.0):
            raise RegionError("the origin is not feasible for L-infinity projection")
        for _ in range(100):
            middle = (low + high) / 2.0
            if middle == low or middle == high:
                break
            if self.contains(np.maximum(p - middle, 0.0), tolerance=0.0):
                high = middle
            else:
                low = middle
        q = np.maximum(p - high, 0.0)
        return ProjectionPoint(q, float(np.max(p - q)))

    def _boundary_normal(self, point: Vector, projected: Vector, norm: str) -> Vector:
        p, q = self._check_point(point), self._check_point(projected)
        if norm == "l2":
            return np.maximum(p - q, 0.0)
        if norm not in {"linf", "gauge"}:
            raise RegionError(f"unsupported projection norm: {norm}")
        # A simplex boundary can be the active boundary when f is slack.
        if self.value(q) < self.threshold - max(1e-10, 100 * self.tolerance):
            normal = np.ones(self.dimension)
        else:
            normal = np.maximum(self.gradient(q), 0.0)
        if norm == "linf":
            delta = float(np.max(p - q))
            saturated = p - q >= delta - max(1e-10, 100 * self.tolerance)
            normal[(q <= max(1e-10, 100 * self.tolerance)) & ~saturated] = 0.0
        if float(np.max(normal)) <= self.tolerance:
            raise RegionError("sublevel boundary has no usable outward normal")
        return normal

    def contains(self, point: Vector, tolerance: float | None = None) -> bool:
        x = self._check_point(point)
        tol = self.tolerance if tolerance is None else tolerance
        return bool(
            np.min(x) >= -tol
            and np.max(x) <= 1.0 + tol
            and np.sum(x) <= 1.0 + tol
            and self.value(x) <= self.threshold + tol
        )

    def violation(self, point: Vector) -> float:
        x = self._check_point(point)
        return float(
            max(
                -float(np.min(x)),
                float(np.max(x - 1.0)),
                float(np.sum(x) - 1.0),
                self.value(x) - self.threshold,
            )
        )

    def _constraints(self) -> list[dict[str, Any]]:
        constraints = self._domain_constraints()
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: self.threshold - self.value(x),
                "jac": lambda x: -self.gradient(x),
            }
        )
        return constraints

    def _ray_start(self, direction: Vector) -> Vector:
        """Construct a nonzero feasible start along a nonnegative ray."""
        ray = np.maximum(np.asarray(direction, dtype=float), 0.0)
        if float(np.sum(ray)) <= self.tolerance:
            ray = np.ones(self.dimension)
        ray /= max(1.0, float(np.sum(ray)), float(np.max(ray)))
        if self.contains(ray, tolerance=0.0):
            return 0.5 * ray
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2.0
            if self.contains(middle * ray, tolerance=0.0):
                low = middle
            else:
                high = middle
        return max(0.25 * low, 1e-8) * ray
