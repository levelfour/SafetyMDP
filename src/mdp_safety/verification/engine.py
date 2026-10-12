"""The CEGAR state machine, separate from file lifecycle and result serialization."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from ..model import DrnModel
from ..oracle import ReachabilityOracle
from ..safe_regions.base import RegionError, SafeRegion
from ..polytope import OuterPolytope, VertexEnumerationLimit
from ..strategies import ProjectionStrategy, Separation, inner_approximation_contains
from ..strategies.sandwich import SandwichApproximationComplete
from .decisions import query_outcome
from .options import VerificationOptions, refinement_precisions
from .queries import EvaluatedQuery, evaluate_query
from .results import SelectionRecord, _float_list, make_query_record
from .state import VerificationState


class VerificationRun:
    def __init__(
        self, options: VerificationOptions, projection: ProjectionStrategy,
        storm: ReachabilityOracle, model: DrnModel, terminal_labels: list[str],
        terminal_map: dict[int, int], region: SafeRegion, work: Path,
    ) -> None:
        self.options = options
        self.projection = projection
        self.storm = storm
        self.model = model
        self.terminal_labels = terminal_labels
        self.terminal_map = terminal_map
        self.region = region
        self.work = work
        self.state = VerificationState(OuterPolytope(region.dimension, options.geometry_tolerance / 100.0))

    def execute(self) -> VerificationState:
        state = self.state
        for iteration in range(1, self.options.max_iterations + 1):
            state.completed_iterations = iteration
            try:
                state.final_vertices = state.enumerate_vertices()
            except VertexEnumerationLimit as error:
                state.reason = f"outer-polytope vertex enumeration limit: {error}"
                state.vertex_enumeration_limited = True
                break
            unsafe_vertices = [vertex for vertex in state.final_vertices
                if not self.region.contains(vertex, tolerance=self.options.geometry_tolerance)]
            if not unsafe_vertices:
                state.status = "SAFE"
                state.reason = (
                    "every vertex of the Storm-certified outer achievable "
                    "region lies in the convex safe region"
                )
                break
            try:
                separation = self._select(iteration, unsafe_vertices)
            except SandwichApproximationComplete as error:
                state.reason = f"sandwich approximation tolerance reached; safety undecided: {error}"
                break
            except RegionError as error:
                state.reason = f"direction selection could not be numerically validated: {error}"
                break
            progress = self._refine(iteration, separation)
            if state.status == "UNSAFE":
                break
            if not progress:
                state.reason = (
                    "Storm's OVI interval could neither exclude the selected "
                    "outer vertex nor certify an unsafe scheduler at minimum precision"
                )
                break
        if state.status == "UNKNOWN" and not state.vertex_enumeration_limited:
            try:
                state.final_vertices = state.enumerate_vertices()
            except VertexEnumerationLimit as error:
                state.reason = f"outer-polytope vertex enumeration limit: {error}"
                state.final_vertices = np.empty((0, self.region.dimension), dtype=float)
        return state

    def _select(self, iteration: int, unsafe_vertices: list[np.ndarray]) -> Separation:
        state = self.state
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            separation = self.projection.select(
                state.outer, unsafe_vertices, self.region,
                [np.asarray(point, dtype=float) for point in state.inner_points],
            )
        except SandwichApproximationComplete as error:
            # The all-facet tolerance check is a measured selection attempt,
            # even though it produces neither a direction nor an oracle query.
            state.selections.append(SelectionRecord(
                iteration=iteration, projection=self.projection.name,
                elapsed_seconds=time.perf_counter() - wall_started,
                cpu_seconds=time.process_time() - cpu_started,
                metric=error.metric, score=error.maximum_gap,
                counterexample=[], projected=[], weight=[],
                selection_guarantee="no-eligible-facet-no-oracle",
            ))
            raise
        separation = self.projection.audit_selection(state.outer, self.region, separation)
        cpu_seconds = time.process_time() - cpu_started
        wall_seconds = time.perf_counter() - wall_started
        state.selections.append(SelectionRecord(
            iteration=iteration, projection=self.projection.name,
            elapsed_seconds=wall_seconds, cpu_seconds=cpu_seconds,
            metric=separation.metric,
            score=(separation.direction_score.signed_score if separation.direction_score else separation.distance),
            counterexample=_float_list(separation.counterexample),
            projected=_float_list(separation.projected), weight=_float_list(separation.weight),
            direction_score=separation.direction_score,
            selection_guarantee=separation.selection_guarantee,
        ))
        return separation

    def _refine(self, iteration: int, separation: Separation) -> bool:
        state = self.state
        tolerance = self.options.geometry_tolerance
        for attempt, precision in enumerate(refinement_precisions(self.options), 1):
            query_id = f"{iteration:04d}-{attempt:02d}"
            query = evaluate_query(
                self.storm, self.model, self.terminal_map, self.terminal_labels,
                separation.weight, precision, self.work, query_id,
            )
            support = query.support
            estimate, lower = query.estimate, query.lower
            state.pareto_samples.append(estimate)
            inner_progress = self.projection.supports_inner_refinement and not inner_approximation_contains(
                np.asarray(lower, dtype=float),
                [np.asarray(old, dtype=float) for old in state.inner_points], tolerance,
            )
            state.inner_points.append(lower)
            lower_violation = self.region.violation(np.asarray(lower))
            outcome = query_outcome(
                lower_violation, support.interval.upper,
                float(np.dot(separation.weight, separation.counterexample)), tolerance,
                self.projection.supports_inner_refinement, inner_progress,
            )
            retained_cut = False
            if self.projection.retain_all_support_cuts and outcome != "unsafe-witness":
                # LK strategies retain every nonredundant oracle upper halfspace,
                # including queries that make progress only in the inner hull.
                retained_cut = state.outer.add_cut(
                    separation.weight, support.interval.upper, f"storm-ovi-{query_id}",
                )
                if separation.initializing:
                    outcome = "initialized"
                elif retained_cut:
                    outcome = "refined"
            self._record_query(iteration, attempt, separation, query, lower_violation, outcome)
            if outcome == "unsafe-witness":
                self._record_witness(separation, query, lower_violation)
                return True
            if outcome == "initialized":
                return True
            if outcome == "refined":
                if self.projection.retain_all_support_cuts:
                    return retained_cut or inner_progress
                added = state.outer.add_cut(separation.weight, support.interval.upper, f"storm-ovi-{query_id}")
                if not added:
                    state.reason = "Storm produced no numerically stronger outer cut"
                return added
            if outcome == "inner-refined":
                return True
        return False

    def _scheduler_actions(self, query: EvaluatedQuery) -> dict[int, str]:
        return query.support.scheduler_actions if self.options.record_scheduler_actions else {}

    def _record_query(
        self, iteration: int, attempt: int, separation: Separation,
        query: EvaluatedQuery, lower_violation: float, outcome: str,
    ) -> None:
        self.state.queries.append(make_query_record(
            self.projection.name, iteration, attempt, separation, query.support.interval,
            query.estimate, query.lower, query.upper, query.scheduler_seconds,
            lower_violation, self._scheduler_actions(query), outcome,
        ))

    def _record_witness(
        self, separation: Separation, query: EvaluatedQuery, lower_violation: float,
    ) -> None:
        self.state.status = "UNSAFE"
        self.state.reason = (
            "a scheduler synthesized by Storm has a componentwise "
            "lower reachability bound outside the downward-closed safe region"
        )
        self.state.witness = {
            "point_estimate": query.estimate,
            "point_lower": query.lower,
            "point_upper": query.upper,
            "safe_region_violation_at_lower": lower_violation,
            "weight": _float_list(separation.weight),
            "scheduler_actions": {
                str(key): value for key, value in sorted(self._scheduler_actions(query).items())
            },
        }
