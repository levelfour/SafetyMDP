"""Verification settings and the precision-refinement schedule."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

from ..defaults import DEFAULT_TIMEOUT_SECONDS
from ..timeouts import validate_timeout


@dataclass(frozen=True)
class VerificationOptions:
    projection: str = "l2"
    storm_precision: float = 1e-7
    minimum_storm_precision: float = 1e-12
    precision_reduction: float = 0.1
    geometry_tolerance: float = 1e-8
    max_iterations: int = 100
    storm_timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS
    keep_working_files: bool = False
    record_scheduler_actions: bool = True
    sandwich_precision: float = 1e-6

    def validate(self) -> None:
        if not math.isfinite(self.sandwich_precision) or self.sandwich_precision <= 0:
            raise ValueError("sandwich precision must be positive and finite")
        if not 0.0 < self.minimum_storm_precision <= self.storm_precision < 1.0:
            raise ValueError("require 0 < minimum precision <= Storm precision < 1")
        if not 0.0 < self.precision_reduction < 1.0:
            raise ValueError("precision reduction must lie in (0, 1)")
        if not math.isfinite(self.geometry_tolerance) or self.geometry_tolerance <= 0:
            raise ValueError("geometry tolerance must be positive and finite")
        validate_timeout(self.storm_timeout_seconds, "Storm timeout")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max iterations must be a positive integer")


def refinement_precisions(options: VerificationOptions) -> Iterator[float]:
    """Reduce absolute OVI precision geometrically, including the configured minimum."""
    precision = options.storm_precision
    while precision >= options.minimum_storm_precision * (1.0 - 1e-12):
        yield precision
        next_precision = precision * options.precision_reduction
        if next_precision < options.minimum_storm_precision:
            if precision > options.minimum_storm_precision:
                precision = options.minimum_storm_precision
            else:
                return
        else:
            precision = next_precision
