"""Shared inner-hull state for the two LK sandwich strategies.

No facet-selection rule lives here: concrete strategies must choose their
own maximum indicator. Storm is used only by the scalar oracle.
"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
from scipy.optimize import linprog

from ..inner_facets import downward_hull_facets
from ..polytope import OuterPolytope
from ..safe_regions.base import RegionError, SafeRegion, Vector
from .base import ProjectionStrategy, Separation


class SandwichApproximationComplete(RegionError):
    """Reaching an approximation tolerance does not prove safety."""

    def __init__(self, precision: float, maximum_gap: float, *, metric: str) -> None:
        self.maximum_gap = maximum_gap
        self.metric = metric
        super().__init__(
            f"all enumerated inner facets are within sandwich precision {precision:g}; "
            "outer containment has not established safety"
        )


def inner_approximation_contains(
    point: Vector, inner_points: list[Vector], tolerance: float
) -> bool:
    """Test membership in the downward closure of conv({0} union points)."""
    value = np.asarray(point, dtype=float)
    if np.max(value) <= tolerance:
        return True
    if not inner_points:
        return False
    samples = np.asarray(inner_points, dtype=float)
    count = len(samples)
    result = linprog(
        np.zeros(count),
        A_ub=np.vstack((-samples.T, np.ones((1, count)))),
        b_ub=np.concatenate((-value + tolerance, [1.0 + tolerance])),
        bounds=[(0.0, None)] * count,
        method="highs",
    )
    return bool(result.success)


def inner_samples(points: list[Vector], dimension: int, tolerance: float) -> Vector:
    """Validate and deduplicate lower-bound points without rounding them up."""
    if not points:
        return np.empty((0, dimension), dtype=float)
    values = np.asarray(points, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != dimension
        or not np.all(np.isfinite(values))
    ):
        raise RegionError("sandwich inner points have an invalid shape or value")
    if np.min(values) < -tolerance:
        raise RegionError("sandwich inner points must be nonnegative")
    # Preserve lower bounds: rounding to nearest can enlarge the inner set.
    return np.unique(np.maximum(values, 0.0), axis=0)


class InnerFacetStrategy(ProjectionStrategy):
    """Coordinate initialization and cached facets, without a selection policy."""

    metric_prefix: str
    supports_inner_refinement = True
    retain_all_support_cuts = True

    def __init__(self, precision: float = 1e-6) -> None:
        self.precision = precision
        self._facets_key = None
        self._facets = ()

    def select(
        self,
        outer: OuterPolytope,
        unsafe_vertices: list[Vector],
        region: SafeRegion,
        inner_points: list[Vector] | None = None,
    ) -> Separation:
        del unsafe_vertices
        if not np.isfinite(self.precision) or self.precision <= 0:
            raise RegionError("sandwich approximation precision must be positive and finite")
        raw_points = [] if inner_points is None else inner_points
        samples = inner_samples(raw_points, outer.dimension, region.tolerance)
        # Count oracle queries, not unique points: duplicate schedulers must
        # not prevent initialization of the remaining coordinates.
        if len(raw_points) < outer.dimension:
            weight = np.eye(outer.dimension)[len(raw_points)]
            value, point = outer.support(weight)
            index = int(np.argmax(samples @ weight)) if len(samples) else None
            support_point = samples[index] if index is not None else np.zeros(outer.dimension)
            inner_support = float(weight @ support_point)
            return Separation(
                counterexample=point, projected=support_point.copy(),
                support_point=support_point.copy(), weight=weight,
                distance=max(0.0, value - inner_support), safe_support=inner_support,
                certified_safe_value=inner_support, margin=value - inner_support,
                metric=f"{self.metric_prefix}-coordinate-initialization",
                selection_guarantee="coordinate-initialization", initializing=True,
            )

        key = (samples.shape, samples.tobytes())
        if key != self._facets_key:
            facets = downward_hull_facets(samples)
            self._facets, self._facets_key = facets, key
        return self._select_facet(outer)

    @abstractmethod
    def _select_facet(self, outer: OuterPolytope) -> Separation:
        """Select a direction or signal that the strategy's tolerance is met."""
