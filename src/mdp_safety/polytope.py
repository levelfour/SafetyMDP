"""Outer polytope used by the CEGAR loop."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, QhullError


Vector = NDArray[np.float64]


class PolytopeError(RuntimeError):
    pass


class VertexEnumerationLimit(PolytopeError):
    """Raised when combinatorial floating-point enumeration exceeds its size guard."""


@dataclass(frozen=True)
class Halfspace:
    normal: Vector
    bound: float
    source: str


class OuterPolytope:
    """Intersection of the probability simplex and Storm-certified cuts."""

    def __init__(self, dimension: int, tolerance: float = 1e-9) -> None:
        self.dimension = dimension
        self.tolerance = tolerance
        self.cuts: list[Halfspace] = []
        self._vertices_cache_key = None
        self._vertices_cache = None

    def inequalities(self) -> tuple[Vector, Vector]:
        identity = np.eye(self.dimension)
        a = np.vstack(
            (
                -identity,
                identity,
                np.ones((1, self.dimension)),
                *([cut.normal for cut in self.cuts] if self.cuts else []),
            )
        )
        b = np.concatenate(
            (
                np.zeros(self.dimension),
                np.ones(self.dimension),
                [1.0],
                [cut.bound for cut in self.cuts],
            )
        )
        return a, b

    def add_cut(self, normal: Vector, bound: float, source: str) -> bool:
        w = np.asarray(normal, dtype=float)
        if w.shape != (self.dimension,) or not np.all(np.isfinite(w)):
            raise PolytopeError("cut has an invalid normal")
        if np.min(w) < -self.tolerance or not np.isfinite(bound):
            raise PolytopeError("cut must have a nonnegative normal and finite bound")
        scale = float(np.max(w))
        if scale <= self.tolerance:
            raise PolytopeError("cut normal is zero")
        w = np.maximum(w / scale, 0.0)
        u = float(bound / scale)
        for old in self.cuts:
            if np.allclose(old.normal, w, atol=self.tolerance, rtol=0.0):
                if old.bound <= u + self.tolerance:
                    return False
        self.cuts.append(Halfspace(w, u, source))
        return True

    def support(self, normal: Vector) -> tuple[float, Vector]:
        a, b = self.inequalities()
        result = linprog(-normal, A_ub=a, b_ub=b, bounds=[(None, None)] * self.dimension, method="highs")
        if not result.success:
            raise PolytopeError(f"outer-polytope LP failed: {result.message}")
        return float(-result.fun), np.asarray(result.x, dtype=float)

    def maximum(self, normal: Vector) -> float:
        return self.support(normal)[0]

    def vertices(self) -> Vector:
        # Include public geometry in the key, so direct changes to cuts or
        # tolerance cannot silently reuse a stale enumeration.
        key = (self.dimension, self.tolerance,
               tuple((cut.normal.tobytes(), cut.bound) for cut in self.cuts))
        if key == self._vertices_cache_key:
            return self._vertices_cache.copy()
        a, b = self.inequalities()
        vertices = self._qhull_vertices(a, b)
        if vertices is None:
            vertices = self._combinatorial_vertices(a, b)
        if len(vertices) == 0:
            raise PolytopeError("outer polytope is empty or numerically degenerate")
        rounded = np.round(vertices, decimals=13)
        unique = np.unique(rounded, axis=0)
        unique[np.abs(unique) < self.tolerance] = 0.0
        unique[np.abs(unique - 1.0) < self.tolerance] = 1.0
        order = np.lexsort(tuple(unique[:, index] for index in reversed(range(self.dimension))))
        result = unique[order]
        self._vertices_cache_key = key
        self._vertices_cache = result.copy()
        return result

    def _qhull_vertices(self, a: Vector, b: Vector) -> Vector | None:
        norms = np.linalg.norm(a, axis=1)
        objective = np.zeros(self.dimension + 1)
        objective[-1] = -1.0
        augmented = np.hstack((a, norms[:, None]))
        result = linprog(
            objective,
            A_ub=augmented,
            b_ub=b,
            bounds=[(None, None)] * self.dimension + [(0.0, None)],
            method="highs",
        )
        if not result.success or result.x[-1] <= max(self.tolerance, 1e-10):
            return None
        halfspaces = np.hstack((a, -b[:, None]))
        try:
            return np.asarray(
                HalfspaceIntersection(halfspaces, result.x[:-1]).intersections,
                dtype=float,
            )
        except (QhullError, ValueError):
            return None

    def _combinatorial_vertices(self, a: Vector, b: Vector) -> Vector:
        if len(a) > 40 or self.dimension > 8:
            raise VertexEnumerationLimit(
                "degenerate vertex enumeration is too large; loosen tolerances or reduce dimension"
            )
        found: list[Vector] = []
        feasibility_tolerance = max(1e-8, 100 * self.tolerance)
        for active in combinations(range(len(a)), self.dimension):
            active_a = a[list(active)]
            if np.linalg.matrix_rank(active_a, tol=self.tolerance) < self.dimension:
                continue
            try:
                point = np.linalg.solve(active_a, b[list(active)])
            except np.linalg.LinAlgError:
                continue
            if np.max(a @ point - b) <= feasibility_tolerance:
                found.append(point)
        if not found:
            return np.empty((0, self.dimension), dtype=float)
        return np.asarray(found, dtype=float)
