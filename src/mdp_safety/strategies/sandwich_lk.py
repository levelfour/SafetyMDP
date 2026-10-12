"""Lammel-Kuefer-style maximum-indicator inner-facet selection.

Evaluate every upper facet of the inner hull for nonnegative terminal
probabilities and maximize its absolute or relative indicator. See
``docs/sandwich_lk.md`` for definitions and the algorithm reference.
The shared Storm oracle and floating-point hull calculations do not establish
an exact-oracle convergence rate for this implementation.
"""

from __future__ import annotations

import numpy as np

from ..polytope import OuterPolytope
from ..safe_regions.base import RegionError
from .base import Separation
from .sandwich import SandwichApproximationComplete, InnerFacetStrategy


class LammelKueferStrategy(InnerFacetStrategy):
    """Maximum absolute epsilon indicator; precision is in probability units.

For I = {x: a_j x <= b_j}, a_j >= 0, the epsilon indicator is
max_j [h_Q(a_j)-b_j]_+ / ||a_j||_1. Indeed, x-epsilon*1 lies
in I iff all facet inequalities with those shifted bounds hold.
"""

    name = "sandwich-lk"
    metric_prefix = "sandwich-lk-epsilon"
    relative = False

    def _select_facet(self, outer: OuterPolytope) -> Separation:
        best = None
        maximum_score = -np.inf
        for facet in self._facets:
            denominator = facet.bound if self.relative else float(np.sum(facet.normal))
            if not np.isfinite(denominator) or denominator <= 0:
                # Do not silently change the reference point or floor a bound:
                # that would implement a different relative-error indicator.
                raise RegionError(
                    "LK gauge requires positive inner support in every nonzero "
                    "nonnegative direction after coordinate initialization; "
                    "a zero-support facet makes the origin-based gauge unavailable "
                    "(use sandwich-lk for the absolute epsilon indicator)"
                )
            outer_support, point = outer.support(facet.normal)
            gap = outer_support - facet.bound
            score = gap / denominator
            if not np.isfinite(score):
                raise RegionError("LK sandwich indicator is not finite")
            # Facets have deterministic lexical order. Retain its first exact
            # tie, but never stop at the first positive/eligible score.
            if score > maximum_score:
                maximum_score = score
                best = (facet, point, gap)
        if best is None:
            raise RegionError("LK sandwich requires a complete nonempty inner facet list")
        if maximum_score <= self.precision:
            raise SandwichApproximationComplete(
                self.precision, max(0.0, maximum_score),
                metric=f"{self.metric_prefix}-tolerance-check",
            )
        facet, point, gap = best
        return Separation(
            counterexample=point, projected=facet.support_point.copy(),
            support_point=facet.support_point.copy(), weight=facet.normal.copy(),
            distance=maximum_score, safe_support=facet.bound,
            certified_safe_value=facet.bound, margin=gap,
            metric=f"{self.metric_prefix}-max-inner-facet-gap",
            selection_guarantee="maximum-inner-facet-indicator-numerically-checked",
        )


class LammelKueferGaugeStrategy(LammelKueferStrategy):
    """Maximum inner-hull gauge minus one, about the fixed reference c=0.

For positive b_j, gamma_I(x) = max_j a_j*x/b_j on x >= 0.
This is the gauge of the evolving INNER hull, not the safe-region gauge.
Zero is a conservative nadir reference for maximized probabilities. No
empirical nadir estimation or artificial positive-support regularizer is used.
Precision is a dimensionless relative excess, NOT an absolute distance.
"""

    name = "sandwich-lk-gauge"
    metric_prefix = "sandwich-lk-gauge"
    relative = True
