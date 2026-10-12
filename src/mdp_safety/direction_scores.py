"""Normalized support-gap scores and numerical supporting-direction checks.

The MDP frontend uses W = R_+^d and S = S0 - R_+^d. Thus h_S = h_S0
on every admitted direction. These are numerical diagnostics, not interval
certificates of a global nonlinear optimum.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .safe_regions.base import RegionError, Vector


class ErrorUnit(ABC):
    """Extension point for h_T; adding a unit also requires a direction solver."""

    name: str

    @abstractmethod
    def support(self, weight: Vector, safe_support: float) -> float:
        """Return h_T(weight), strictly positive for every nonzero w >= 0."""


class L2Unit(ErrorUnit):
    name = "l2-ball"

    def support(self, weight: Vector, safe_support: float) -> float:
        return float(np.linalg.norm(weight))


class LInfUnit(ErrorUnit):
    name = "linf-ball"

    def support(self, weight: Vector, safe_support: float) -> float:
        # The dual norm is L1, NOT Linf.
        return float(np.linalg.norm(weight, ord=1))


class GaugeUnit(ErrorUnit):
    name = "safe-region-about-reference"

    def __init__(self, reference: Vector | None = None) -> None:
        self.reference = None if reference is None else np.asarray(reference, dtype=float).copy()

    def support(self, weight: Vector, safe_support: float) -> float:
        if self.reference is None:
            return float(safe_support)  # The MDP rule fixes c = 0.
        if self.reference.shape != weight.shape or not np.all(np.isfinite(self.reference)):
            raise RegionError("gauge reference has an invalid dimension or value")
        return float(safe_support - weight @ self.reference)


@dataclass(frozen=True)
class DirectionScore:
    unit: str
    outer_support: float
    safe_support: float
    support_gap: float
    normalizer: float
    signed_score: float
    positive_part: float


def evaluate_direction(
    weight: Vector, outer_support: float, safe_support: float, unit: ErrorUnit,
) -> DirectionScore:
    """Evaluate v_T(w), preserving its sign even when Q is already safe.

    positive_part is for this direction only. It equals the global excess V_T
    only for a maximizing direction (or when global containment is known).
    A positive denominator at one w does not certify admissibility of a unit
    at every w; GaugeStrategy separately checks the absorption assumption.
    """
    w = np.asarray(weight, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise RegionError("a direction must be finite, nonzero, and in R_+^d")
    denominator = unit.support(w, safe_support)
    if not np.all(np.isfinite([outer_support, safe_support, denominator])) or denominator <= 0:
        raise RegionError("the unit of error must have a finite positive normalizer")
    gap = float(outer_support - safe_support)
    score = gap / denominator
    return DirectionScore(unit.name, float(outer_support), float(safe_support), gap,
                          denominator, score, max(0.0, score))


def check_supporting_direction(
    point: Vector, boundary: Vector, weight: Vector, safe_support: float,
    expected_excess: float, unit: ErrorUnit, tolerance: float,
) -> None:
    """Check boundary support and normalized gap against the geometric excess."""
    local = evaluate_direction(weight, float(weight @ point), safe_support, unit)
    scale = max(1.0, abs(expected_excess), abs(local.signed_score))
    allowed = max(5e-7, 1000.0 * tolerance) * scale
    # Dividing by h_T also detects errors amplified by a thin gauge direction.
    support_error = abs(safe_support - float(weight @ boundary)) / local.normalizer
    score_error = abs(local.signed_score - expected_excess)
    if support_error > allowed or score_error > allowed:
        raise RegionError(
            "Section 3 direction certificate failed "
            f"(unit={unit.name}, support_error={support_error:.3g}, "
            f"score_error={score_error:.3g}, tolerance={allowed:.3g})"
        )
