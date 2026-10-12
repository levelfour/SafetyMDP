"""Storm result records, file decoding and outward-rounded probability bounds."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .model import DrnModel


class StormError(RuntimeError):
    pass


@dataclass(frozen=True)
class IntervalResult:
    estimate: float
    lower: float
    upper: float
    precision: float
    elapsed_seconds: float
    command: list[str]


@dataclass(frozen=True)
class SupportResult:
    interval: IntervalResult
    scheduler: dict[int, int]
    scheduler_actions: dict[int, str]
    weighted_model: DrnModel


@dataclass(frozen=True)
class OracleQueryResult:
    support: SupportResult
    coordinates: list[IntervalResult]


def parse_initial_value(result_path: Path) -> float:
    try:
        result_data = json.loads(result_path.read_text(encoding="utf-8"))
        initial_values = [entry["v"] for entry in result_data if "init" in entry.get("l", [])]
        if len(initial_values) != 1:
            raise ValueError(f"expected one initial result, found {len(initial_values)}")
        estimate = float(initial_values[0])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise StormError(f"cannot parse Storm result file {result_path}: {error}") from error
    return estimate


def probability_interval(estimate: float, precision: float, upper_cap: float = 1.0) -> tuple[float, float]:
    """Enclose a sound absolute-precision estimate with roundoff and outward rounding."""
    guard = max(8.0 * math.ulp(estimate), 1e-15)
    lower = max(0.0, math.nextafter(estimate - precision - guard, -math.inf))
    upper = min(upper_cap, math.nextafter(estimate + precision + guard, math.inf))
    return lower, upper
