import numpy as np
import pytest
from scipy.optimize import linprog

from mdp_safety.inner_facets import downward_hull_facets
from mdp_safety.polytope import OuterPolytope
from mdp_safety.regions import PolytopeRegion, RegionError
from mdp_safety.strategies.sandwich import SandwichApproximationComplete, InnerFacetStrategy, inner_samples
from mdp_safety.strategies.sandwich_lk import LammelKueferStrategy


def test_two_point_hull_has_three_upper_facets():
    facets = downward_hull_facets(np.array([[.8, .2], [.2, .8]]))
    normals = np.array([f.normal / max(f.normal) for f in facets])
    assert normals == pytest.approx(np.array([[0., 1.], [1., 1.], [1., 0.]]), abs=1e-12)
    assert [f.bound / max(f.normal) for f in facets] == pytest.approx([.8, 1., .8])


@pytest.mark.parametrize("dimension", [1, 2, 6, 10, 14])
def test_singleton_and_duplicate_hulls_include_zero_coordinates(dimension):
    point = np.linspace(0., .7, dimension)
    facets = downward_hull_facets(np.array([point, point]))
    assert len(facets) == dimension
    for facet in facets:
        assert np.linalg.norm(facet.normal) == pytest.approx(1.)
        assert facet.bound == pytest.approx(facet.normal @ point)
        assert np.count_nonzero(facet.normal > 1e-10) == 1


def test_anisotropic_probabilities_keep_small_coordinate_facets():
    points = np.array([[.8, 2e-13], [.2, 8e-13]])
    facets = downward_hull_facets(points)
    assert len(facets) == 3
    for facet in facets:
        assert facet.bound == pytest.approx(np.max(points @ facet.normal), rel=1e-12, abs=1e-25)
    assert any(0 < f.normal[0] < 1e-10 and f.normal[1] > .99 for f in facets)


@pytest.mark.parametrize("dimension", [2, 3, 5, 8])
def test_facet_hull_support_matches_independent_sample_support(dimension):
    rng = np.random.default_rng(419 + dimension)
    points = rng.dirichlet(np.ones(dimension + 1), size=12)[:, :dimension]
    facets = downward_hull_facets(points)
    a = np.array([f.normal for f in facets])
    b = np.array([f.bound for f in facets])
    for normal in rng.random((15, dimension)):
        result = linprog(-normal, A_ub=a, b_ub=b, bounds=[(None, None)] * dimension, method="highs")
        assert result.success
        assert -result.fun == pytest.approx(np.max(points @ normal), abs=1e-8)


@pytest.mark.parametrize("points", [[], [[-1., .2]], [[np.nan, .2]], [[np.inf, .2]]])
def test_invalid_samples_fail_explicitly(points):
    with pytest.raises(RegionError):
        downward_hull_facets(np.array(points))


def test_selects_maximum_facet_without_enumerating_outer_vertices(monkeypatch):
    outer = OuterPolytope(2)
    region = PolytopeRegion(np.eye(2), [.95, .95])
    points = [np.array([.2, .8]), np.array([.2, .8])]
    # Lexical facet order is e2 then e1. Gaps are .2 and .8 respectively.
    monkeypatch.setattr(outer, "vertices", lambda: pytest.fail("must not enumerate outer vertices"))
    result = LammelKueferStrategy().select(outer, [], region, points)
    assert result.weight == pytest.approx([1., 0.], abs=1e-12)
    assert result.distance == pytest.approx(.8)
    assert "max-inner-facet" in result.metric


def test_largest_facet_gap_uses_l2_unit_normal():
    outer = OuterPolytope(2)
    outer.add_cut(np.array([0., 1.]), .8 + 1e-8, "test")
    outer.add_cut(np.array([1., 0.]), .8 + 1e-8, "test")
    points = [np.array([.6, .2]), np.array([.2, .6])]
    # Clip the coordinate directions to their inner bounds; the diagonal
    # face then remains eligible while e2 and e1 do not.
    outer.add_cut(np.array([0., 1.]), .6 + 1e-8, "test")
    outer.add_cut(np.array([1., 0.]), .6 + 1e-8, "test")
    result = LammelKueferStrategy(1e-6).select(outer, [], PolytopeRegion([[1., 1.]], [.9]), points)
    assert result.weight == pytest.approx(np.ones(2) / np.sqrt(2), abs=1e-12)
    assert result.distance == pytest.approx(.1)
    assert result.support_point @ result.weight == pytest.approx(result.safe_support)


def test_gap_tolerance_is_not_a_safety_certificate():
    outer = OuterPolytope(2)
    outer.add_cut(np.array([1., 0.]), .2, "test")
    outer.add_cut(np.array([0., 1.]), .3, "test")
    region = PolytopeRegion([[1., 1.]], [.49])
    with pytest.raises(SandwichApproximationComplete):
        LammelKueferStrategy().select(outer, [], region, [np.array([.2, .3])] * 2)


def test_initialization_counts_duplicate_queries_and_has_no_strategy_run_state():
    strategy = LammelKueferStrategy()
    outer = OuterPolytope(3)
    region = PolytopeRegion(np.eye(3), [.9] * 3)
    for count in [0, 1, 2, 0]:
        result = strategy.select(outer, [], region, [np.zeros(3)] * count)
        assert result.weight == pytest.approx(np.eye(3)[count])
        assert result.initializing


def test_facet_cache_invalidates_when_samples_change(monkeypatch):
    import mdp_safety.strategies.sandwich as module
    real = module.downward_hull_facets
    calls = []
    def counted(points):
        calls.append(points.copy())
        return real(points)
    monkeypatch.setattr(module, "downward_hull_facets", counted)
    strategy = LammelKueferStrategy()
    outer = OuterPolytope(2)
    region = PolytopeRegion(np.eye(2), [.9, .9])
    points = [np.array([.2, .3])] * 2
    strategy.select(outer, [], region, points)
    strategy.select(outer, [], region, points)
    points[0][1] = .4
    strategy.select(outer, [], region, points)
    assert len(calls) == 2


@pytest.mark.parametrize("dimension", [6, 10])
def test_former_robot_projection_failure_inputs_have_valid_facets(dimension):
    import json
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "sandwich" / f"robot_d{dimension}_legacy_failure.json"
    points = np.array(json.loads(fixture.read_text())["samples"])
    facets = downward_hull_facets(points)
    a = np.array([f.normal for f in facets])
    b = np.array([f.bound for f in facets])
    for normal in np.random.default_rng(dimension).random((10, dimension)):
        result = linprog(-normal, A_ub=a, b_ub=b, bounds=[(None, None)] * dimension, method="highs")
        assert result.success
        assert -result.fun == pytest.approx(np.max(points @ normal), abs=1e-8)


def test_qhull_failure_is_explicit_not_a_partial_hull(monkeypatch):
    import mdp_safety.inner_facets as module
    def fail(*args):
        raise module.QhullError("test geometry failure")
    monkeypatch.setattr(module, "HalfspaceIntersection", fail)
    with pytest.raises(RegionError, match="inner facet enumeration failed"):
        downward_hull_facets(np.array([[.2, .3]]))

def test_common_facet_strategy_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        InnerFacetStrategy()


def test_sample_cleanup_does_not_round_lower_bounds_up():
    sample = np.array([.12345678901236, .23456789012346])
    assert np.array_equal(inner_samples([sample], 2, 1e-10)[0], sample)


@pytest.mark.parametrize("points", [
    [[.1]], [[np.nan, .2]], [[np.inf, .2]], [[-.1, .2]],
])
def test_shared_sample_validation_rejects_invalid_points(points):
    with pytest.raises(RegionError, match="inner points"):
        inner_samples([np.asarray(p) for p in points], 2, 1e-10)
