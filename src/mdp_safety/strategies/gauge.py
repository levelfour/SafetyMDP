"""Gauge-based relative refinement."""

from __future__ import annotations

import numpy as np

from ..direction_scores import GaugeUnit, check_supporting_direction
from ..polytope import OuterPolytope
from ..safe_regions.base import RegionError, SafeRegion, Vector
from .base import ProjectionStrategy, Separation

class GaugeStrategy(ProjectionStrategy):
    """Relative refinement maximizing the safe-region gauge over outer vertices."""

    name = "gauge"
    error_unit = GaugeUnit()
    maximizes_direction_score = True

    def select(
        self,
        outer: OuterPolytope,
        unsafe_vertices: list[Vector],
        region: SafeRegion,
        inner_points: list[Vector] | None = None,
    ) -> Separation:
        del outer, inner_points
        self._prepare_cache(region)
        if region.absorption_radius() <= max(1e-12, 100.0 * region.tolerance):
            raise RegionError(
                "gauge selection requires S to contain rho times the simplex for some rho > 0"
            )
        scores = region.gauge_scores(np.asarray(unsafe_vertices, dtype=float))
        # np.argmax selects the first candidate on an exact tie.
        point = unsafe_vertices[int(np.argmax(scores))]
        gamma, boundary = region.gauge(point)
        raw_weight = region.supporting_normal(point, boundary, "gauge")
        scale = float(np.max(raw_weight))
        if not np.isfinite(scale) or scale <= region.tolerance:
            raise RegionError("gauge selection produced a zero supporting weight")
        weight = np.maximum(raw_weight / scale, 0.0)
        safe_support, support_point = self._support(region, weight)
        check_supporting_direction(point, boundary, weight, safe_support,
                                   gamma - 1.0, self.error_unit, region.tolerance)
        margin = float(np.dot(weight, point) - safe_support)
        if margin <= max(1e-8, 100.0 * region.tolerance):
            raise RegionError(
                f"gauge selection failed to produce a strict separator (margin={margin:.3g})"
            )
        return Separation(
            counterexample=point,
            projected=boundary,
            support_point=support_point,
            weight=weight,
            distance=gamma - 1.0,
            safe_support=safe_support,
            certified_safe_value=float(np.dot(weight, support_point)),
            margin=margin,
            metric="safe-region-gauge-minus-one",
        )
