"""Safe-region interface, validation and generic radial geometry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


class RegionError(ValueError):
    """Raised when a safe-region specification violates the assumptions."""


@dataclass(frozen=True)
class ProjectionPoint:
    point: Vector
    distance: float


class SafeRegion(ABC):
    """Closed, convex, downward-closed subset of the probability simplex."""

    def __init__(self, dimension: int, tolerance: float = 1e-9) -> None:
        if dimension < 1:
            raise RegionError("safe-region dimension must be positive")
        self.dimension = dimension
        self.tolerance = tolerance

    def cache_signature(self) -> object:
        """Geometry revision for optional caches; None disables cross-call reuse."""
        return None

    def projection_scores(self, points: Vector, norm: str) -> Vector | None:
        """Optional fast distances. Finalists still use the ordinary projection."""
        return None

    def l2_distance_bounds(self, points: Vector) -> tuple[Vector, Vector]:
        return np.zeros(len(points)), np.full(len(points), np.inf)

    def l2_dual_lower_bound(self, point: Vector, projected: Vector) -> float:
        return 0.0

    @abstractmethod
    def contains(self, point: Vector, tolerance: float | None = None) -> bool:
        pass

    @abstractmethod
    def violation(self, point: Vector) -> float:
        """Return a positive number only when ``point`` is outside the region."""

    @abstractmethod
    def project(self, point: Vector, norm: str) -> ProjectionPoint:
        pass

    @abstractmethod
    def maximize_linear(self, weight: Vector) -> tuple[float, Vector]:
        pass

    @abstractmethod
    def supporting_normal(self, point: Vector, projected: Vector, norm: str) -> Vector:
        """Return a nonnegative outward normal at a projection point."""

    def _check_point(self, point: Vector) -> Vector:
        value = np.asarray(point, dtype=float)
        if value.shape != (self.dimension,):
            raise RegionError(
                f"expected a {self.dimension}-vector, got shape {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise RegionError("point contains a non-finite value")
        return value

    def _domain_constraints(self) -> list[dict[str, Any]]:
        return [
            {"type": "ineq", "fun": lambda x: x, "jac": lambda x: np.eye(self.dimension)},
            {
                "type": "ineq",
                "fun": lambda x: 1.0 - x,
                "jac": lambda x: -np.eye(self.dimension),
            },
            {
                "type": "ineq",
                "fun": lambda x: 1.0 - float(np.sum(x)),
                "jac": lambda x: -np.ones(self.dimension),
            },
        ]

    def _strictly_feasible_copy(self, point: Vector) -> Vector:
        """Find the nearest feasible point along the uniform downward path.

        Keep the feasible endpoint of a monotone bracket. Decimal-size nudges
        can overshoot the boundary enough to invalidate a supporting direction.
        This restores feasibility only; callers must still check optimality.
        """
        q = np.maximum(self._check_point(point), 0.0)
        if self.contains(q, tolerance=0.0):
            return q
        feasible = np.zeros(self.dimension)
        if not self.contains(feasible, tolerance=0.0):
            raise RegionError("the origin is not feasible for boundary recovery")
        low, high = 0.0, float(np.max(q))
        for _ in range(100):
            middle = low + (high - low) / 2.0
            if middle == low or middle == high:
                break
            candidate = np.maximum(q - middle, 0.0)
            if self.contains(candidate, tolerance=0.0):
                high, feasible = middle, candidate
            else:
                low = middle
        return feasible

    def absorption_radius(self) -> float:
        """Largest axis radius certified to satisfy rho * simplex subset region."""
        radii: list[float] = []
        for index in range(self.dimension):
            direction = np.zeros(self.dimension)
            direction[index] = 1.0
            if self.contains(direction, tolerance=0.0):
                radii.append(1.0)
                continue
            low, high = 0.0, 1.0
            for _ in range(80):
                middle = (low + high) / 2.0
                if self.contains(middle * direction, tolerance=0.0):
                    low = middle
                else:
                    high = middle
            radii.append(low)
        return min(radii)

    def gauge(self, point: Vector) -> tuple[float, Vector]:
        """Return gamma_S(point) and a feasible radial boundary point."""
        p = self._check_point(point)
        return self._gauge_with_radius(p, self.absorption_radius())

    def gauge_scores(self, points: Vector) -> Vector:
        """Evaluate candidate gauges, sharing the region-only radius calculation.

        Subclasses can provide a batched closed form. The radius is reused only
        within this call, so changing a region between selections cannot leave
        a stale cache.
        """
        values = self._check_gauge_points(points)
        radius = self.absorption_radius()
        return np.asarray([self._gauge_with_radius(p, radius)[0] for p in values])

    def _check_gauge_points(self, points: Vector) -> Vector:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise RegionError("gauge candidates must have shape (count, dimension)")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RegionError("gauge candidates must be finite and nonnegative")
        return values

    def _gauge_with_radius(self, p: Vector, radius: float) -> tuple[float, Vector]:
        if np.any(p < 0.0):
            raise RegionError("gauge candidates must be finite and nonnegative")
        if not np.any(p):
            return 0.0, np.zeros(self.dimension)
        if radius <= max(1e-12, 100.0 * self.tolerance):
            raise RegionError(
                "gauge selection requires S to contain rho times the simplex for some rho > 0"
            )
        high = max(1.0, float(np.sum(p)), float(np.max(p)) / radius)
        while not self.contains(p / high, tolerance=0.0):
            high *= 2.0
            if high > 1e15:
                raise RegionError("failed to bracket the safe-region gauge")
        low = 0.0
        for _ in range(100):
            middle = (low + high) / 2.0
            if middle == low or middle == high:
                break
            if middle > 0.0 and self.contains(p / middle, tolerance=0.0):
                high = middle
            else:
                low = middle
        boundary = self._strictly_feasible_copy(p / high)
        return high, boundary
