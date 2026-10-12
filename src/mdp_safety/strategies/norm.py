"""L2 and L-infinity projection-based weight selection."""

from __future__ import annotations

import numpy as np

from ..direction_scores import L2Unit, LInfUnit, check_supporting_direction
from ..polytope import OuterPolytope
from ..safe_regions.base import ProjectionPoint, RegionError, SafeRegion, Vector
from .base import ProjectionStrategy, Separation

class NormProjectionStrategy(ProjectionStrategy):
    maximizes_direction_score = True

    def __init__(self, norm: str) -> None:
        if norm not in {"l2", "linf"}:
            raise RegionError(f"unsupported projection norm: {norm}")
        self.norm = norm
        self.name = norm
        self.error_unit = L2Unit() if norm == "l2" else LInfUnit()

    def separate(self, point: Vector, region: SafeRegion) -> Separation:
        self._prepare_cache(region)
        p = np.asarray(point, dtype=float)
        projection = region.project(p, self.norm)
        return self._separate_projected(p, projection, region)

    def _separate_projected(self, p: Vector, projection: ProjectionPoint, region: SafeRegion) -> Separation:
        raw_weight = region.supporting_normal(p, projection.point, self.norm)
        scale = float(np.max(raw_weight))
        if not np.isfinite(scale) or scale <= region.tolerance:
            raise RegionError(f"{self.name} projection produced a zero separating weight")
        weight = np.maximum(raw_weight / scale, 0.0)
        safe_support, support_point = self._support(region, weight)
        check_supporting_direction(p, projection.point, weight, safe_support,
                                   projection.distance, self.error_unit, region.tolerance)
        certified_safe_value = float(np.dot(weight, support_point))
        margin = float(np.dot(weight, p) - safe_support)
        if margin <= max(1e-8, 100.0 * region.tolerance):
            raise RegionError(
                f"{self.name} projection failed to produce a strict separator "
                f"(margin={margin:.3g})"
            )
        return Separation(
            counterexample=p,
            projected=projection.point.copy(),
            support_point=support_point,
            weight=weight,
            distance=projection.distance,
            safe_support=safe_support,
            certified_safe_value=certified_safe_value,
            margin=margin,
            metric=f"{self.norm}-projection-distance",
        )

    def select(
        self,
        outer: OuterPolytope,
        unsafe_vertices: list[Vector],
        region: SafeRegion,
        inner_points: list[Vector] | None = None,
    ) -> Separation:
        del outer, inner_points
        self._prepare_cache(region)
        points = np.asarray(unsafe_vertices, dtype=float)
        scores = region.projection_scores(points, self.norm)
        if scores is not None:
            index = int(np.argmax(scores))
            # Compute the selected candidate's projection and supporting
            # direction, then validate strict separation once.
            projection = region.project(points[index], self.norm)
            return self._separate_projected(points[index], projection, region)
        lower, upper = region.l2_distance_bounds(points) if self.norm == "l2" else (
            np.zeros(len(points)), np.full(len(points), np.inf)
        )
        best_lower = float(np.max(lower))
        best = None
        best_index = len(points)
        # Distance upper bounds determine evaluation order and pruning;
        # every potentially farthest vertex remains eligible.
        for index in np.argsort(-upper, kind="stable"):
            if upper[index] + 1e-10 < best_lower:
                continue
            p = points[index]
            key = p.tobytes()
            if key not in self._projection_cache:
                projection = region.project(p, self.norm)
                dual_lower = region.l2_dual_lower_bound(p, projection.point) if self.norm == "l2" else 0.0
                self._projection_cache[key] = (projection, dual_lower)
                if len(self._projection_cache) > 32768:
                    self._projection_cache.popitem(last=False)
            self._projection_cache.move_to_end(key)
            projection, dual_lower = self._projection_cache[key]
            best_lower = max(best_lower, dual_lower)
            if best is None or projection.distance > best.distance or (
                projection.distance == best.distance and index < best_index
            ):
                best, best_index = projection, int(index)
        if best is None:
            raise RegionError("projection screening left no eligible candidate")
        return self._separate_projected(points[best_index], best, region)
