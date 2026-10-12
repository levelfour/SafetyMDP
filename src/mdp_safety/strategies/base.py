"""Weight-selection interface and shared geometry-aware caches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np

from ..direction_scores import DirectionScore, ErrorUnit, evaluate_direction
from ..polytope import OuterPolytope
from ..safe_regions.base import RegionError, SafeRegion, Vector

@dataclass(frozen=True)
class Separation:
    counterexample: Vector
    projected: Vector
    support_point: Vector
    weight: Vector
    distance: float
    safe_support: float
    certified_safe_value: float
    margin: float
    metric: str
    direction_score: DirectionScore | None = None
    selection_guarantee: str = "external-or-baseline"
    initializing: bool = False


class ProjectionStrategy(ABC):
    name: str
    error_unit: ErrorUnit | None = None
    maximizes_direction_score: bool = False
    retain_all_support_cuts: bool = False
    supports_inner_refinement: bool = False

    def audit_selection(self, outer: OuterPolytope, region: SafeRegion, result: Separation) -> Separation:
        """Audit the normalized support-gap score within query-selection time.

        Reuses the outer vertices already enumerated by the verifier. Only
        built-in exact-reduction rules may claim the geometric excess equals
        the score; custom heuristic strategies need not maximize it.
        """
        if self.error_unit is None:
            return result
        outer_support = float(np.max(outer.vertices() @ result.weight))
        score = evaluate_direction(result.weight, outer_support, result.safe_support, self.error_unit)
        guarantee = "finite-candidate-heuristic"
        if self.maximizes_direction_score:
            allowed = max(5e-7, 1000 * region.tolerance) * max(1.0, abs(result.distance))
            if abs(score.signed_score - result.distance) > allowed:
                raise RegionError("Section 3 score disagrees with the maximal geometric excess")
            guarantee = "global-geometric-reduction-numerically-checked"
        return replace(result, direction_score=score, selection_guarantee=guarantee)

    def _prepare_cache(self, region: SafeRegion) -> None:
        signature = region.cache_signature()
        key = (id(region), signature)
        if signature is None or getattr(self, "_cache_key", None) != key:
            self._cache_key = key
            self._cache_region = region
            self._support_cache = OrderedDict()
            self._projection_cache = OrderedDict()

    def _support(self, region: SafeRegion, weight: Vector) -> tuple[float, Vector]:
        key = weight.tobytes()
        if key not in self._support_cache:
            value, point = region.maximize_linear(weight)
            self._support_cache[key] = (value, point.copy())
            if len(self._support_cache) > 4096:
                self._support_cache.popitem(last=False)
        self._support_cache.move_to_end(key)
        value, point = self._support_cache[key]
        return value, point.copy()

    @abstractmethod
    def select(
        self,
        outer: OuterPolytope,
        unsafe_vertices: list[Vector],
        region: SafeRegion,
        inner_points: list[Vector] | None = None,
    ) -> Separation:
        pass
