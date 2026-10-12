import numpy as np
import pytest

from mdp_safety.polytope import OuterPolytope
from mdp_safety.projections import make_projection
from mdp_safety.regions import PolytopeRegion, region_from_dict


@pytest.mark.parametrize("dimension", [2, 4, 10])
def test_linf_prefix_distances_match_lp(dimension):
    rng = np.random.default_rng(20260907 + dimension)
    matrix = np.vstack((np.eye(dimension), rng.uniform(0, 1, (12, dimension))))
    bounds = rng.uniform(0.01, 0.6, len(matrix))
    bounds[0] = 0  # The optimized norm path also supports nonabsorbing regions.
    region = PolytopeRegion(matrix, bounds)
    points = rng.dirichlet(np.ones(dimension), 40)
    points[::3, 0] = 0
    points[1] *= 1.8  # Exercise the simplex facet as well.
    scores = region.projection_scores(points, "linf")
    reference = [region.project(p, "linf").distance for p in points]
    np.testing.assert_allclose(scores, reference, atol=2e-9, rtol=1e-8)


@pytest.mark.parametrize("norm", ["l2", "linf"])
def test_selected_distance_matches_exhaustive_projection(norm):
    rng = np.random.default_rng(71)
    region = PolytopeRegion(np.vstack((np.eye(4), [[1, .6, .3, 0], [0, 0, .5, 1]])), [.14, .28, .3, .19, .2, .21])
    points = list(rng.dirichlet(np.ones(4), 70))
    reference = [region.project(p, norm) for p in points]
    index = int(np.argmax([q.distance for q in reference]))
    selected = make_projection(norm).select(OuterPolytope(4), points, region)
    assert selected.distance == pytest.approx(reference[index].distance, abs=1e-8)
    np.testing.assert_allclose(selected.counterexample, points[index], atol=1e-8)
    assert selected.margin > 0


def test_l2_primal_dual_bounds_enclose_reference_distance():
    rng = np.random.default_rng(33)
    region = PolytopeRegion(np.vstack((np.eye(5), rng.random((7, 5)))), [.1, .2, .05, .3, .15, *([.35]*7)])
    points = rng.dirichlet(np.ones(5), 60)
    lower, upper = region.l2_distance_bounds(points)
    for p, lo, hi in zip(points, lower, upper):
        projected = region.project(p, "l2")
        dual = region.l2_dual_lower_bound(p, projected.point)
        assert lo <= projected.distance + 1e-9 <= hi + 1e-8
        assert dual <= projected.distance + 1e-9
        assert projected.distance - dual < 2e-6


@pytest.mark.parametrize("norm", ["l2", "linf"])
def test_only_finalist_support_is_solved_and_mutation_invalidates_cache(norm, monkeypatch):
    region = PolytopeRegion(np.eye(2), [.2, .3])
    points = [np.array([1., 0.]), *[np.array([.21 + .0001*j, .2]) for j in range(50)]]
    calls = {"project": 0, "support": 0}
    original_project = region.project
    original_support = region.maximize_linear
    def project(*args):
        calls["project"] += 1
        return original_project(*args)
    def support(*args):
        calls["support"] += 1
        return original_support(*args)
    monkeypatch.setattr(region, "project", project)
    monkeypatch.setattr(region, "maximize_linear", support)
    strategy = make_projection(norm)
    strategy.select(OuterPolytope(2), points, region)
    assert calls == {"project": 1, "support": 1}
    strategy.select(OuterPolytope(2), points, region)
    assert calls["support"] == 1
    assert calls["project"] == (1 if norm == "l2" else 2)
    region.bounds[0] = .1
    selected = strategy.select(OuterPolytope(2), points, region)
    assert selected.distance == pytest.approx(.9)
    assert calls["support"] == 2



def test_vertex_cache_tracks_geometry_and_returns_independent_arrays(monkeypatch):
    outer = OuterPolytope(2)
    calls = 0
    original = outer._qhull_vertices
    def enumerate_vertices(*args):
        nonlocal calls
        calls += 1
        return original(*args)
    monkeypatch.setattr(outer, "_qhull_vertices", enumerate_vertices)
    points = outer.vertices()
    points[:] = -100
    assert np.min(outer.vertices()) >= 0
    assert calls == 1
    outer.add_cut(np.array([1., 0.]), .4, "test")
    assert np.max(outer.vertices()[:, 0]) == pytest.approx(.4)
    assert calls == 2
    outer.cuts[0].normal[0] = .5
    assert np.max(outer.vertices()[:, 0]) == pytest.approx(.8)
    assert calls == 3


def test_formula_cache_tracks_parameter_changes():
    region = region_from_dict({"type": "formula", "formula": "x1^2 + x2^2 <= 0.25"}, dimension_hint=2)
    strategy = make_projection("l2")
    points = [np.array([1., 0.]), np.array([0., 1.])]
    first = strategy.select(OuterPolytope(2), points, region)
    region.threshold = .16
    second = strategy.select(OuterPolytope(2), points, region)
    assert first.distance == pytest.approx(.5, abs=1e-6)
    assert second.distance == pytest.approx(.6, abs=1e-6)
