import json

import pytest

from mdp_safety import benchmarking, drn, projections, regions, storm, verifier
from mdp_safety.experiments.models import BenchmarkSettings
from mdp_safety.model import DrnModel
from mdp_safety.safe_regions import FormulaRegion, PolytopeRegion, QuadraticRegion
from mdp_safety.specification import specification_from_dict
from mdp_safety.storm_results import IntervalResult, StormError, parse_initial_value, probability_interval
from mdp_safety.strategies import LammelKueferStrategy
from mdp_safety.verification.results import SafetyResult


def test_legacy_imports_reexport_same_objects():
    assert benchmarking.BenchmarkSettings is BenchmarkSettings
    assert drn.DrnModel is DrnModel
    assert projections.LammelKueferStrategy is LammelKueferStrategy
    assert regions.PolytopeRegion is PolytopeRegion
    assert regions.QuadraticRegion is QuadraticRegion
    assert regions.FormulaRegion is FormulaRegion
    assert storm.IntervalResult is IntervalResult
    assert storm.StormError is StormError
    assert verifier.SafetyResult is SafetyResult


def test_registry_is_shared_and_inner_capability_is_not_tied_to_display_name():
    from mdp_safety.strategies import make_projection, register_projection
    class RenamedSandwich(LammelKueferStrategy):
        name = "test-sandwich-alias"
    register_projection("test-sandwich-alias", RenamedSandwich)
    try:
        assert isinstance(projections.make_projection("test-sandwich-alias"), RenamedSandwich)
        assert make_projection("test-sandwich-alias").supports_inner_refinement
        assert not make_projection("l2").supports_inner_refinement
    finally:
        projections._STRATEGIES.pop("test-sandwich-alias")


def test_specification_with_derived_terminal_sets_and_formula():
    spec = specification_from_dict({
        "terminal_sets": [{"name": "bad", "all_of": ["finished"]}],
        "terminalize": True,
        "safe_region": {"type": "formula", "formula": "x1^2 <= .4"},
    })
    assert spec.terminal_labels == ["bad"] and spec.terminalize
    assert spec.safe_region.dimension == 1


@pytest.mark.parametrize("values", [
    {"terminal_sets": []}, {"terminal_sets": [None]},
    {"terminal_labels": "bad"}, {"terminal_labels": [None]},
    {"terminal_labels": ["bad"], "terminalize": "yes"},
])
def test_invalid_specification_is_rejected_before_execution(values):
    with pytest.raises(ValueError):
        specification_from_dict(values)


@pytest.mark.parametrize("value", [0., .2, 1.])
def test_outward_interval_preserves_original_bound_rule(value):
    low, high = probability_interval(value, 1e-7)
    assert 0 <= low <= value <= high <= 1
    assert low <= max(0, value - 1e-7)
    assert high >= min(1, value + 1e-7)


@pytest.mark.parametrize("data", [[], [{"l": ["init"]}], [{"l": ["init"], "v": .1}] * 2])
def test_storm_initial_value_decode_error(data, tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    with pytest.raises(StormError, match="cannot parse"):
        parse_initial_value(path)


def test_storm_result_decode_and_cap(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps([{"l": ["init"], "v": .2}]))
    assert parse_initial_value(path) == .2
    assert probability_interval(.2, 1e-7, upper_cap=.2)[1] == .2


def test_provenance_includes_nested_implementation_modules(tmp_path):
    from mdp_safety.experiments.provenance import python_source_hashes
    nested = tmp_path / "strategies"
    nested.mkdir()
    source = nested / "gauge.py"
    source.write_text("revision = 1\n")
    before = python_source_hashes(tmp_path)
    assert "strategies/gauge.py" in before
    source.write_text("revision = 2\n")
    assert before != python_source_hashes(tmp_path)
