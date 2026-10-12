import numpy as np
import pytest

from mdp_safety.polytope import OuterPolytope
from mdp_safety.projections import make_projection
from mdp_safety.regions import PolytopeRegion, RegionError, SafeRegion, region_from_dict


def test_closed_form_matches_membership_bisection_and_homogeneity():
    rng = np.random.default_rng(270905)
    matrix = rng.uniform(0.0, 2.0, (17, 6))
    matrix[0] = 0.0
    bounds = rng.uniform(0.02, 1.0, 17)
    bounds[0] = 0.0  # A redundant row must not cause a division by zero.
    region = PolytopeRegion(matrix, bounds)
    points = rng.dirichlet(np.ones(6), size=30) * rng.uniform(0.01, 2.0, (30, 1))
    direct = region.gauge_scores(points)
    reference = [SafeRegion.gauge(region, p)[0] for p in points]
    np.testing.assert_allclose(direct, reference, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(region.gauge_scores(0.037 * points), 0.037 * direct)
    for p in points:
        gamma, q = region.gauge(p)
        assert region.contains(q, tolerance=0.0)
        np.testing.assert_allclose(q * gamma, p, atol=1e-15)
        assert not region.contains(q * (1 + 1e-8), tolerance=0.0)


def test_gauge_includes_simplex_and_zero_point():
    region = PolytopeRegion([[0.0, 0.0]], [0.0])
    np.testing.assert_allclose(region.gauge_scores([[0, 0], [1, 1], [0.1, 0.2]]), [0, 2, 0.3])
    assert region.gauge(np.zeros(2))[0] == 0.0
    assert region.absorption_radius() == 1.0
    assert region.gauge_scores(np.empty((0, 2))).shape == (0,)


def test_radius_is_not_stale_when_region_changes():
    region = PolytopeRegion(np.eye(2), [0.2, 0.4])
    assert region.absorption_radius() == pytest.approx(0.2)
    region.bounds[0] = 0.1
    assert region.absorption_radius() == pytest.approx(0.1)
    assert region.gauge([0.2, 0])[0] == pytest.approx(2.0)


def test_batched_generic_gauge_reuses_radius(monkeypatch):
    region = region_from_dict({"type": "formula", "formula": "x1^4 + 2*x2^4 <= 0.1"}, dimension_hint=2)
    calls = 0
    original = region.absorption_radius

    def counted():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(region, "absorption_radius", counted)
    points = np.asarray([[1, 0], [0, 1], [0.4, 0.6]])
    scores = region.gauge_scores(points)
    assert calls == 1
    expected = ((points[:, 0] ** 4 + 2 * points[:, 1] ** 4) / 0.1) ** 0.25
    np.testing.assert_allclose(scores, expected, rtol=1e-12)


@pytest.mark.parametrize("points", [[[np.nan, 0]], [[-0.1, 0.2]], [[1, 2, 3]]])
def test_gauge_rejects_invalid_points(points):
    with pytest.raises(RegionError):
        PolytopeRegion(np.eye(2), [0.2, 0.4]).gauge_scores(points)


def test_nonabsorbing_region_is_rejected():
    region = PolytopeRegion([[1, 0]], [0])
    with pytest.raises(RegionError, match="rho times"):
        region.gauge_scores([[0, 1]])


def test_batch_selection_only_constructs_one_boundary(monkeypatch):
    region = PolytopeRegion([[1, 0], [0, 1], [1, 1]], [0.2, 0.4, 0.5])
    outer = OuterPolytope(2)
    vertices = [p for p in outer.vertices() if not region.contains(p)]
    calls = 0
    original = region.gauge

    def counted(p):
        nonlocal calls
        calls += 1
        return original(p)

    monkeypatch.setattr(region, "gauge", counted)
    selected = make_projection("gauge").select(outer, vertices, region)
    assert calls == 1
    assert selected.margin > 0
    np.testing.assert_allclose(selected.weight, [1, 0])
