from pathlib import Path

import numpy as np
import pytest

from mdp_safety.model import Action, DrnModel, State
from mdp_safety.polytope import OuterPolytope, VertexEnumerationLimit
from mdp_safety.regions import PolytopeRegion
from mdp_safety.storm_results import IntervalResult, OracleQueryResult, SupportResult
from mdp_safety.verification.decisions import query_outcome
from mdp_safety.verification.options import refinement_precisions
from mdp_safety.verifier import SafetyVerifier, VerificationOptions


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["terminal_left", "terminal_right"]


class FakeOracle:
    """Only for state-machine tests; real-Storm tests remain separate."""

    version = "test-oracle"

    def __init__(self, upper_mode="tight"):
        self.upper_mode = upper_mode
        self.precisions = []
        self.work = None

    def build_prism(self, prism_path, output_path, constants=None):
        self.work = output_path.parent
        return DrnModel("MDP", [
            State(0, {"init"}, [Action("go", [(1, .2), (2, .1), (3, .7)])]),
            State(1, {LABELS[0]}, [Action("absorb", [(1, 1.)])]),
            State(2, {LABELS[1]}, [Action("absorb", [(2, 1.)])]),
            State(3, {"safe"}, [Action("absorb", [(3, 1.)])]),
        ])

    def maximize_weighted_reachability(self, model, terminals, weight, precision, work, query_id):
        self.precisions.append(precision)
        value = float(np.dot(weight, [.2, .1]))
        loose = self.upper_mode == "loose" or (self.upper_mode == "first-loose" and precision >= .1)
        interval = IntervalResult(value, value, 1.0 if loose else value, precision, 0., [])
        return SupportResult(interval, {0: 0}, {0: "go"}, model)

    def evaluate_scheduler(self, model, labels, terminals, weighted, scheduler, precision, work, query_id):
        return [IntervalResult(value, value, value, precision, 0., []) for value in [.2, .1]]

    def query(self, model, terminal_map, labels, weight, precision, work, query_id):
        support = self.maximize_weighted_reachability(model, terminal_map, weight, precision, work, query_id)
        return OracleQueryResult(support, self.evaluate_scheduler(
            model, labels, set(terminal_map), support.weighted_model, support.scheduler, precision, work, query_id))


def run(oracle, bounds=(.4, .4), **changes):
    options = dict(storm_precision=.1, minimum_storm_precision=.01, max_iterations=1)
    options.update(changes)
    verifier = SafetyVerifier(VerificationOptions(**options), "not-an-installed-storm", storm_runner=oracle)
    result = verifier.verify(ROOT / "tests/fixtures/smoke/two_terminal.pm", LABELS, PolytopeRegion(np.eye(2), bounds))
    return result


@pytest.mark.parametrize("violation,upper,allow,progress,expected", [
    (.1, .1, True, True, "unsafe-witness"),
    (0., .1, True, True, "refined"),
    (0., 1., True, True, "inner-refined"),
    (0., 1., False, True, "retry-with-higher-precision"),
    (0., 1., True, False, "retry-with-higher-precision"),
])
def test_decision_precedence(violation, upper, allow, progress, expected):
    assert query_outcome(violation, upper, 1., 1e-8, allow, progress) == expected


def test_precision_schedule_reaches_minimum_once():
    options = VerificationOptions(storm_precision=.1, minimum_storm_precision=.015, precision_reduction=.5)
    assert list(refinement_precisions(options)) == [.1, .05, .025, .015]
    assert list(refinement_precisions(VerificationOptions(storm_precision=.1, minimum_storm_precision=.1))) == [.1]


def test_retry_preserves_oracle_count_and_precision():
    oracle = FakeOracle("first-loose")
    result = run(oracle)
    assert oracle.precisions == pytest.approx([.1, .01])
    assert [query.outcome for query in result.queries] == ["retry-with-higher-precision", "refined"]
    assert result.status == "UNKNOWN"  # The one-iteration budget leaves containment unchecked.
    assert result.oracle_call_count == 2 and result.weight_selection_call_count == 1
    assert result.storm_call_count == 6
    assert len(result.storm_cuts) == 1
    assert oracle.work is not None and not oracle.work.exists()


def test_no_progress_terminates_at_minimum_precision():
    oracle = FakeOracle("loose")
    result = run(oracle)
    assert result.status == "UNKNOWN" and "minimum precision" in result.reason
    # In floating point, .1*.1 is just above .01,
    # so the exact configured minimum is queried once more before stopping.
    assert oracle.precisions == [.1, .1 * .1, .01]
    assert result.oracle_call_count == 3 and result.iterations == 1


def test_no_timeout_still_stops_after_50_refinement_iterations(monkeypatch):
    from mdp_safety.verification.engine import VerificationRun
    iterations = []

    def keep_refining(self, iteration, separation):
        iterations.append(iteration)
        return True

    monkeypatch.setattr(VerificationRun, "_refine", keep_refining)
    result = run(FakeOracle(), max_iterations=50, storm_timeout_seconds=None)
    assert iterations == list(range(1, 51))
    assert result.status == "UNKNOWN" and result.iterations == 50
    assert result.reason == "maximum number of refinements reached"
    assert result.options["storm_timeout_seconds"] is None


def test_unsafe_uses_lower_bound_and_preserves_action_omission():
    result = run(FakeOracle(), bounds=(.1, .05), record_scheduler_actions=False)
    assert result.status == "UNSAFE" and result.oracle_call_count == 1
    assert result.witness["safe_region_violation_at_lower"] > 0
    assert result.witness["scheduler_actions"] == result.queries[0].scheduler_actions == {}


def test_initially_safe_requires_no_oracle_query():
    result = run(FakeOracle(), bounds=(1., 1.))
    assert result.status == "SAFE" and result.oracle_call_count == 0
    assert result.outer_vertex_enumeration_call_count == 1


def test_vertex_limit_is_unknown_not_unsafe(monkeypatch):
    def limited(self):
        raise VertexEnumerationLimit("test limit")
    monkeypatch.setattr(OuterPolytope, "vertices", limited)
    result = run(FakeOracle())
    assert result.status == "UNKNOWN" and "test limit" in result.reason
    assert result.outer_vertex_enumeration_call_count == 1
    assert result.oracle_call_count == 0 and result.outer_vertices == []


def test_repeated_verification_has_fresh_run_state():
    verifier = SafetyVerifier(storm_runner=FakeOracle())
    region = PolytopeRegion(np.eye(2), [.1, .05])
    results = [verifier.verify(ROOT / "tests/fixtures/smoke/two_terminal.pm", LABELS, region) for _ in range(2)]
    assert [result.oracle_call_count for result in results] == [1, 1]
    assert results[0].queries is not results[1].queries


@pytest.mark.parametrize("projection", ["l2", "linf", "gauge"])
def test_section3_diagnostics_are_serialized(projection):
    result = run(FakeOracle(), projection=projection)
    data = result.to_dict()
    assert data["schema_version"] == 6
    score = data["selections"][0]["direction_score"]
    assert score["signed_score"] == pytest.approx(score["support_gap"] / score["normalizer"])
    assert data["queries"][0]["direction_score"] == score
    assert data["queries"][0]["selection_score"] == score["signed_score"]


def test_direction_certificate_failure_is_unknown_without_oracle_call(monkeypatch):
    from mdp_safety.regions import RegionError
    from mdp_safety.strategies.norm import NormProjectionStrategy

    def fail(self, *args):
        raise RegionError("test direction certificate failure")

    monkeypatch.setattr(NormProjectionStrategy, "audit_selection", fail)
    result = run(FakeOracle())
    assert result.status == "UNKNOWN" and "test direction certificate failure" in result.reason
    assert result.oracle_call_count == 0 and result.witness is None


@pytest.mark.parametrize("projection", ["sandwich-lk", "sandwich-lk-gauge"])
def test_sandwich_initializes_all_coordinates_even_with_duplicate_samples(projection):
    oracle = FakeOracle("loose")
    result = run(oracle, projection=projection, max_iterations=2)
    assert result.oracle_call_count == 2
    assert oracle.precisions == [.1, .1]
    assert [q.weight for q in result.queries] == [[1., 0.], [0., 1.]]
    assert [q.outcome for q in result.queries] == ["initialized", "initialized"]
    assert len(result.storm_cuts) == 2


@pytest.mark.parametrize("projection,prefix", [
    ("sandwich-lk", "sandwich-lk-epsilon"),
    ("sandwich-lk-gauge", "sandwich-lk-gauge"),
])
def test_sandwich_tolerance_completion_is_unknown_and_serialized(projection, prefix):
    from dataclasses import replace
    class InexactOracle(FakeOracle):
        def maximize_weighted_reachability(self, *args):
            support = super().maximize_weighted_reachability(*args)
            return replace(support, interval=replace(support.interval, upper=support.interval.upper + 1e-6))
    result = run(InexactOracle(), bounds=(.2 + .5e-6, .1 + .5e-6),
                 projection=projection, max_iterations=5, sandwich_precision=1e-4)
    assert result.status == "UNKNOWN"
    assert "sandwich approximation tolerance reached" in result.reason
    assert result.oracle_call_count == 2
    assert result.weight_selection_call_count == 3
    assert result.selections[-1].weight == []
    assert result.selections[-1].metric == f"{prefix}-tolerance-check"
    assert result.selections[-1].elapsed_seconds > 0
    assert result.weight_selection_seconds == pytest.approx(sum(s.elapsed_seconds for s in result.selections))
    assert result.options["sandwich_precision"] == 1e-4


def test_sandwich_repeated_verification_restarts_coordinate_initialization():
    verifier = SafetyVerifier(VerificationOptions(projection="sandwich-lk"), storm_runner=FakeOracle())
    region = PolytopeRegion(np.eye(2), [.1, .05])
    results = [verifier.verify(ROOT / "tests/fixtures/smoke/two_terminal.pm", LABELS, region) for _ in range(2)]
    assert [result.queries[0].weight for result in results] == [[1., 0.], [1., 0.]]


@pytest.mark.parametrize("precision", [0, -1, float("nan"), float("inf")])
def test_invalid_sandwich_precision_is_rejected(precision):
    with pytest.raises(ValueError, match="sandwich precision"):
        VerificationOptions(sandwich_precision=precision).validate()


def test_query_evaluates_exactly_the_scheduler_returned_by_weighted_optimization():
    class TrackingOracle(FakeOracle):
        def maximize_weighted_reachability(self, *args):
            self.support = super().maximize_weighted_reachability(*args)
            self.query_id = args[-1]
            return self.support

        def evaluate_scheduler(self, model, labels, terminals, weighted, scheduler, precision, work, query_id):
            assert weighted is self.support.weighted_model
            assert scheduler is self.support.scheduler
            assert query_id == self.query_id
            return super().evaluate_scheduler(model, labels, terminals, weighted, scheduler, precision, work, query_id)

    result = run(TrackingOracle(), bounds=(.1, .05))
    assert result.status == "UNSAFE"
    assert result.witness["point_lower"] == [.2, .1]
    assert result.storm_call_count == 3


def test_wrong_coordinate_count_is_rejected():
    class IncompleteOracle(FakeOracle):
        def evaluate_scheduler(self, *args):
            return super().evaluate_scheduler(*args)[:1]

    with pytest.raises(ValueError, match="number of scheduler coordinates"):
        run(IncompleteOracle())
