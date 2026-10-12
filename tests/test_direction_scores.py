"""Normalized support-gap and supporting-direction tests, independent of the oracle."""

import numpy as np
import pytest

from mdp_safety.direction_scores import (
    GaugeUnit, L2Unit, LInfUnit, check_supporting_direction, evaluate_direction,
)
from mdp_safety.polytope import OuterPolytope
from mdp_safety.projections import make_projection
from mdp_safety.regions import PolytopeRegion, QuadraticRegion, RegionError, region_from_dict


def box_outer(caps):
    outer = OuterPolytope(len(caps))
    for row, bound in zip(np.eye(len(caps)), caps):
        outer.add_cut(row, bound, "test-box")
    return outer


@pytest.mark.parametrize("name,weight,excess", [
    ("l2", [1., .5], np.sqrt(5) / 20),
    ("linf", [1., 0.], .1),
    ("gauge", [0., 1.], 1.),
])
def test_example_3_9(name, weight, excess):
    region = PolytopeRegion(np.eye(2), [.4, .05])
    outer = box_outer([.5, .1])
    strategy = make_projection(name)
    result = strategy.select(outer, [v for v in outer.vertices() if not region.contains(v)], region)
    result = strategy.audit_selection(outer, region, result)
    np.testing.assert_allclose(result.weight, weight, atol=2e-7)
    assert result.distance == pytest.approx(excess, abs=2e-7)
    assert result.direction_score.signed_score == pytest.approx(excess, abs=2e-7)
    assert result.selection_guarantee == "global-geometric-reduction-numerically-checked"
    # Independently bound all directions on a dense grid, without using the
    # strategy's projections or its choice of normal.
    angles = np.linspace(0, np.pi / 2, 10001)
    weights = np.c_[np.cos(angles), np.sin(angles)]
    gap = weights @ np.array([.1, .05])
    denominator = (np.linalg.norm(weights, axis=1) if name == "l2" else
                   np.sum(weights, axis=1) if name == "linf" else weights @ [.4, .05])
    assert np.max(gap / denominator) <= result.direction_score.signed_score + 2e-7


@pytest.mark.parametrize("unit", [L2Unit(), LInfUnit(), GaugeUnit()])
def test_scale_invariant_signed_score(unit):
    w = np.array([2., 1.])
    a = evaluate_direction(w, .2, .4, unit)
    b = evaluate_direction(17 * w, 17 * .2, 17 * .4, unit)
    assert a.signed_score == pytest.approx(b.signed_score)
    assert a.signed_score < 0 and a.positive_part == 0


def test_gauge_reference_and_linf_dual_norm():
    assert LInfUnit().support(np.array([2., 1.]), 1.) == 3.
    assert GaugeUnit(np.array([.1, .05])).support(np.array([2., 1.]), .5) == pytest.approx(.25)
    with pytest.raises(RegionError, match="positive normalizer"):
        evaluate_direction(np.array([1., 0.]), .5, .1, GaugeUnit([.1, 0.]))


@pytest.mark.parametrize("weight", [[0., 0.], [-1., 1.], [np.nan, 1.]])
def test_invalid_direction_rejected(weight):
    with pytest.raises(RegionError):
        evaluate_direction(np.array(weight), 1., .4, L2Unit())


def test_equation_6_rejects_supporting_but_nonmaximizing_normal():
    # A supporting normal of the box at q need not be an L-infinity
    # subgradient of the residual: e2 fails the latter condition.
    with pytest.raises(RegionError, match="direction certificate failed"):
        check_supporting_direction(np.array([.5, .1]), np.array([.4, .05]),
                                   np.array([0., 1.]), .05, .1, LInfUnit(), 1e-10)


@pytest.mark.parametrize("factor", [1e-4, 1., 100., 1e5])
def test_linf_direction_independent_of_constraint_row_scaling(factor):
    region = PolytopeRegion([[1, 0], [0, factor]], [.4, .05 * factor])
    result = make_projection("linf").separate(np.array([.5, .1]), region)
    np.testing.assert_allclose(result.weight, [1., 0.], atol=1e-8)
    assert result.margin / np.sum(result.weight) == pytest.approx(.1, abs=1e-8)


@pytest.mark.parametrize("kind", ["polytope", "quadratic", "formula"])
def test_linf_normal_accounts_for_zero_coordinates(kind):
    region = (PolytopeRegion([[1, 1]], [.1]) if kind == "polytope" else
              QuadraticRegion(np.zeros((2, 2)), [1, 1], 0, .1) if kind == "quadratic" else
              region_from_dict({"type": "formula", "formula": "x1+x2 <= .1"}, dimension_hint=2))
    result = make_projection("linf").separate(np.array([.6, .05]), region)
    np.testing.assert_allclose(result.projected, [.1, 0.], atol=1e-8)
    np.testing.assert_allclose(result.weight, [1., 0.], atol=1e-8)
    assert result.distance == pytest.approx(.5, abs=1e-8)


@pytest.mark.parametrize("name", ["l2", "linf", "gauge"])
def test_score_matches_geometry_on_curved_region(name):
    region = QuadraticRegion(np.diag([1., 4.]), [0., 0.], 0., .1)
    outer = box_outer([.5, .3])
    strategy = make_projection(name)
    result = strategy.select(outer, [p for p in outer.vertices() if not region.contains(p)], region)
    result = strategy.audit_selection(outer, region, result)
    w = result.weight
    # Ellipsoid support, unaffected by the simplex for these radii.
    expected_support = np.sqrt(.1 * (w[0] ** 2 + w[1] ** 2 / 4))
    assert result.direction_score.safe_support == pytest.approx(expected_support, abs=5e-7)
    assert result.direction_score.signed_score == pytest.approx(result.distance, abs=5e-7)



def test_gauge_uses_maximal_ratio_not_almost_active_raw_violation():
    region = PolytopeRegion([[1, 0], [0, 1e6]], [.2, 1e6 * .20000005])
    outer = box_outer([.4, .4])
    strategy = make_projection("gauge")
    result = strategy.select(outer, [p for p in outer.vertices() if not region.contains(p)], region)
    np.testing.assert_allclose(result.weight, [1, 0])
    assert strategy.audit_selection(outer, region, result).direction_score.signed_score == pytest.approx(1.)


def test_gauge_facet_query_bound_under_the_stated_oracle_hypothesis():
    region = PolytopeRegion([[1, 0], [0, 1], [1, 1]], [.4, .2, .5])
    outer, strategy, queried = OuterPolytope(2), make_projection("gauge"), []
    for _ in range(4):
        unsafe = [v for v in outer.vertices() if not region.contains(v)]
        if not unsafe:
            break
        result = strategy.select(outer, unsafe, region)
        strategy.audit_selection(outer, region, result)
        assert not any(np.allclose(result.weight, w) for w in queried)
        queried.append(result.weight)
        # Ideal upper-bound oracle for K=S0. This is NOT an assumption made
        # about finite-precision Storm in integration tests.
        outer.add_cut(result.weight, result.safe_support, "ideal-support")
    assert len(queried) <= 3
    assert all(region.contains(v) for v in outer.vertices())
