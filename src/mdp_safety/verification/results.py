"""Stable verification records, Pareto filtering and result construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

import numpy as np

from ..model import DrnModel
from ..direction_scores import DirectionScore
from ..storm_results import IntervalResult
from ..strategies.base import Separation
from .options import VerificationOptions

if TYPE_CHECKING:
    from .state import VerificationState


@dataclass
class QueryRecord:
    iteration: int
    attempt: int
    projection: str
    counterexample: list[float]
    projected: list[float]
    weight: list[float]
    projection_distance: float
    selection_metric: str
    selection_score: float
    separator_margin: float
    safe_support: float
    storm_precision: float
    support_estimate: float
    support_lower: float
    support_upper: float
    support_seconds: float
    witness_seconds: float
    witness_estimate: list[float]
    witness_lower: list[float]
    witness_upper: list[float]
    witness_violation_lower: float
    scheduler_actions: dict[str, str]
    outcome: str
    direction_score: DirectionScore | None = None
    selection_guarantee: str = "external-or-baseline"


@dataclass
class SelectionRecord:
    iteration: int
    projection: str
    elapsed_seconds: float
    cpu_seconds: float
    metric: str
    score: float
    counterexample: list[float]
    projected: list[float]
    weight: list[float]
    direction_score: DirectionScore | None = None
    selection_guarantee: str = "external-or-baseline"


@dataclass
class SafetyResult:
    status: str
    reason: str
    model_path: str
    terminal_labels: list[str]
    terminalize: bool
    projection: str
    storm_version: str
    storm_configuration: dict[str, Any]
    options: dict[str, Any]
    elapsed_seconds: float
    iterations: int
    oracle_call_count: int
    weight_selection_call_count: int
    weight_selection_seconds: float
    weight_selection_cpu_seconds: float
    outer_vertex_enumeration_call_count: int
    outer_vertex_enumeration_seconds: float
    maximum_outer_vertex_count: int
    storm_call_count: int
    state_count: int
    original_choice_count: int
    outer_halfspaces: list[dict[str, Any]]
    storm_cuts: list[dict[str, Any]]
    outer_vertices: list[list[float]]
    pareto_samples: list[list[float]]
    inner_approximation_points: list[list[float]]
    selections: list[SelectionRecord]
    queries: list[QueryRecord]
    witness: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = 6
        return data


def _float_list(values: np.ndarray | list[float]) -> list[float]:
    return [float(value) for value in values]


def _pareto_filter(points: list[list[float]], tolerance: float) -> list[list[float]]:
    if not points:
        return []
    unique: list[np.ndarray] = []
    for point in points:
        value = np.asarray(point, dtype=float)
        if not any(np.allclose(value, old, atol=tolerance, rtol=0.0) for old in unique):
            unique.append(value)
    nondominated: list[np.ndarray] = []
    for index, point in enumerate(unique):
        dominated = any(
            index != other_index
            and np.all(other >= point - tolerance)
            and np.any(other > point + tolerance)
            for other_index, other in enumerate(unique)
        )
        if not dominated:
            nondominated.append(point)
    nondominated.sort(key=lambda p: tuple(float(x) for x in p))
    return [_float_list(point) for point in nondominated]


def make_query_record(
    projection: str,
    iteration: int,
    attempt: int,
    separation: Separation,
    interval: IntervalResult,
    estimate: list[float],
    lower: list[float],
    upper: list[float],
    witness_seconds: float,
    lower_violation: float,
    scheduler_actions: dict[int, str],
    outcome: str,
) -> QueryRecord:
    return QueryRecord(
        iteration=iteration,
        attempt=attempt,
        projection=projection,
        counterexample=_float_list(separation.counterexample),
        projected=_float_list(separation.projected),
        weight=_float_list(separation.weight),
        projection_distance=separation.distance,
        selection_metric=separation.metric,
        selection_score=(separation.direction_score.signed_score if separation.direction_score else separation.distance),
        separator_margin=separation.margin,
        safe_support=separation.safe_support,
        storm_precision=interval.precision,
        support_estimate=interval.estimate,
        support_lower=interval.lower,
        support_upper=interval.upper,
        support_seconds=interval.elapsed_seconds,
        witness_seconds=witness_seconds,
        witness_estimate=estimate,
        witness_lower=lower,
        witness_upper=upper,
        witness_violation_lower=lower_violation,
        scheduler_actions={
            str(key): value for key, value in sorted(scheduler_actions.items())
        },
        outcome=outcome,
        direction_score=separation.direction_score,
        selection_guarantee=separation.selection_guarantee,
    )


def build_result(
    state: VerificationState, options: VerificationOptions, model_path: Path,
    terminal_labels: list[str], terminalize: bool, projection_name: str,
    storm_version: str, base_model: DrnModel, started: float,
) -> SafetyResult:
    """Serialize one completed state, including normalized support-gap diagnostics."""
    outer = state.outer
    queries = state.queries
    selections = state.selections
    pareto_samples = state.pareto_samples
    inner_points = state.inner_points
    witness = state.witness
    status = state.status
    reason = state.reason
    completed_iterations = state.completed_iterations
    outer_vertex_enumeration_call_count = state.outer_vertex_enumeration_call_count
    outer_vertex_enumeration_seconds = state.outer_vertex_enumeration_seconds
    maximum_outer_vertex_count = state.maximum_outer_vertex_count
    final_vertices = state.final_vertices
    outer_a, outer_b = outer.inequalities()
    base_sources = (
        [f"probability-lower-{index + 1}" for index in range(outer.dimension)]
        + [f"probability-upper-{index + 1}" for index in range(outer.dimension)]
        + ["subdistribution-simplex"]
    )
    cut_sources = [cut.source for cut in outer.cuts]
    return SafetyResult(
        status=status,
        reason=reason,
        model_path=str(model_path),
        terminal_labels=list(terminal_labels),
        terminalize=terminalize,
        projection=projection_name,
        storm_version=storm_version,
        storm_configuration={
            "sound": True,
            "method": "optimistic-value-iteration",
            "absolute_precision": True,
            "upper_bound_rule": "nextafter(estimate + precision + roundoff)",
        },
        options=asdict(options),
        elapsed_seconds=time.perf_counter() - started,
        iterations=completed_iterations,
        oracle_call_count=len(queries),
        weight_selection_call_count=len(selections),
        weight_selection_seconds=sum(
            selection.elapsed_seconds for selection in selections
        ),
        weight_selection_cpu_seconds=sum(
            selection.cpu_seconds for selection in selections
        ),
        outer_vertex_enumeration_call_count=(
            outer_vertex_enumeration_call_count
        ),
        outer_vertex_enumeration_seconds=outer_vertex_enumeration_seconds,
        maximum_outer_vertex_count=maximum_outer_vertex_count,
        storm_call_count=len(queries) * (1 + outer.dimension),
        state_count=len(base_model.states),
        original_choice_count=base_model.number_of_choices,
        outer_halfspaces=[
            {
                "normal": _float_list(normal),
                "bound": float(bound),
                "source": source,
            }
            for normal, bound, source in zip(
                outer_a, outer_b, base_sources + cut_sources
            )
        ],
        storm_cuts=[
            {
                "normal": _float_list(cut.normal),
                "bound": cut.bound,
                "source": cut.source,
            }
            for cut in outer.cuts
        ],
        outer_vertices=[_float_list(vertex) for vertex in final_vertices],
        pareto_samples=_pareto_filter(
            pareto_samples, options.geometry_tolerance
        ),
        inner_approximation_points=_pareto_filter(
            inner_points, options.geometry_tolerance
        ),
        selections=selections,
        queries=queries,
        witness=witness,
    )
