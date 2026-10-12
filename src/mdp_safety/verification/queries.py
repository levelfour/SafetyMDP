"""One weighted optimization and coordinate bounds for its fixed scheduler."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..model import DrnModel
from ..oracle import ReachabilityOracle
from ..storm_results import SupportResult


@dataclass(frozen=True)
class EvaluatedQuery:
    support: SupportResult
    estimate: list[float]
    lower: list[float]
    upper: list[float]
    scheduler_seconds: float


def evaluate_query(
    oracle: ReachabilityOracle,
    model: DrnModel,
    terminal_map: dict[int, int],
    terminal_labels: list[str],
    weight: np.ndarray,
    precision: float,
    work: Path,
    query_id: str,
) -> EvaluatedQuery:
    """Keep the optimization and same-scheduler evaluation inseparable.

    This helper does not select directions or modify the outer/inner geometry.
    The engine decides how to use the returned bounds and accounts for queries.
    """
    result = oracle.query(
        model, terminal_map, terminal_labels, weight, precision, work, query_id,
    )
    support, coordinates = result.support, result.coordinates
    if len(coordinates) != len(terminal_labels):
        raise ValueError("oracle returned the wrong number of scheduler coordinates")
    return EvaluatedQuery(
        support=support,
        estimate=[coordinate.estimate for coordinate in coordinates],
        lower=[coordinate.lower for coordinate in coordinates],
        upper=[coordinate.upper for coordinate in coordinates],
        scheduler_seconds=sum(coordinate.elapsed_seconds for coordinate in coordinates),
    )
