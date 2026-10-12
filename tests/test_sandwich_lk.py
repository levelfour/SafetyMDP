import numpy as np
import pytest
from scipy.optimize import linprog

from mdp_safety.polytope import OuterPolytope
from mdp_safety.regions import PolytopeRegion, RegionError
from mdp_safety.strategies import make_projection
from mdp_safety.strategies.sandwich import SandwichApproximationComplete
from mdp_safety.strategies.sandwich_lk import LammelKueferStrategy, LammelKueferGaugeStrategy
from mdp_safety.verifier import SafetyVerifier, VerificationOptions


METHODS = ("sandwich-lk", "sandwich-lk-gauge")


@pytest.mark.parametrize("name", METHODS)
def test_lk_selects_maximum_not_first_eligible_facet(name, monkeypatch):
    outer = OuterPolytope(2)
    region = PolytopeRegion(np.eye(2), [.95, .95])
    points = [np.array([.2, .8])] * 2
    monkeypatch.setattr(outer, "vertices", lambda: pytest.fail("no outer vertex enumeration in selection"))
    from mdp_safety.inner_facets import downward_hull_facets
    first_normal = downward_hull_facets(np.asarray(points))[0].normal
    maximum = make_projection(name).select(outer, [], region, points)
    assert first_normal == pytest.approx([0., 1.], abs=1e-12)
    assert maximum.weight == pytest.approx([1., 0.], abs=1e-12)
    assert maximum.distance == pytest.approx(.8 if name == "sandwich-lk" else 4.)
    assert maximum.direction_score is None  # Not a safe-region score.
    assert maximum.selection_guarantee == "maximum-inner-facet-indicator-numerically-checked"


def test_epsilon_and_gauge_normalizers_produce_different_faces():
    outer = OuterPolytope(2)
    outer.add_cut(np.array([1., 0.]), .78, "x")
    outer.add_cut(np.array([0., 1.]), .78, "y")
    outer.add_cut(np.ones(2), .9, "sum")
    points = [np.array([.6, 0.]), np.array([0., .6])]
    region = PolytopeRegion(np.eye(2), [.95, .95])
    epsilon = LammelKueferStrategy().select(outer, [], region, points)
    gauge = LammelKueferGaugeStrategy().select(outer, [], region, points)
    assert epsilon.weight == pytest.approx([0., 1.], abs=1e-12)  # lexical exact tie
    assert epsilon.distance == pytest.approx(.18)
    assert gauge.weight == pytest.approx(np.ones(2) / np.sqrt(2), abs=1e-12)
    assert gauge.distance == pytest.approx(.5)
    # The raw diagonal support gap is larger: epsilon MUST divide by L1.
    assert gauge.margin > epsilon.margin


def primal_indicator(point, samples, relative):
    count, dimension = samples.shape
    if relative:
        result = linprog(np.ones(count), A_ub=-samples.T, b_ub=-point,
                         bounds=(0, None), method="highs")
        assert result.success
        return max(0., float(result.fun) - 1.)
    # min epsilon s.t. sum(lambda*s) >= point-epsilon*1,
    # sum(lambda) <= 1, lambda >= 0. Includes 0 in the inner convex hull.
    result = linprog(
        np.r_[np.zeros(count), 1.],
        A_ub=np.vstack((np.column_stack((-samples.T, -np.ones(dimension))),
                        np.r_[np.ones(count), 0.])),
        b_ub=np.r_[-point, 1.], bounds=(0, None), method="highs",
    )
    assert result.success
    return float(result.fun)


@pytest.mark.parametrize("dimension", [1, 2, 3, 5])
@pytest.mark.parametrize("relative", [False, True])
def test_maximum_indicator_matches_independent_vertex_primal_lps(dimension, relative):
    rng = np.random.default_rng(492 + dimension)
    samples = rng.dirichlet(np.ones(dimension + 1), size=max(dimension, 5))[:, :dimension] * .65
    outer = OuterPolytope(dimension)
    for normal in rng.random((3, dimension)):
        outer.add_cut(normal, float(np.max(samples @ normal)) + .1, "test")
    expected = max(primal_indicator(p, samples, relative) for p in outer.vertices())
    strategy = LammelKueferGaugeStrategy() if relative else LammelKueferStrategy()
    selection = strategy.select(outer, [], PolytopeRegion(np.eye(dimension), np.ones(dimension)), list(samples))
    assert selection.distance == pytest.approx(expected, abs=2e-8)
    assert np.linalg.norm(selection.weight) == pytest.approx(1.)
    assert selection.safe_support == pytest.approx(np.max(samples @ selection.weight))
    assert selection.margin == pytest.approx(selection.weight @ selection.counterexample - selection.safe_support)


@pytest.mark.parametrize("name", METHODS)
def test_lk_weights_do_not_depend_on_the_safe_region(name, monkeypatch):
    outer = OuterPolytope(2)
    samples = [np.array([.6, .1]), np.array([.1, .5])]
    results = []
    for bounds in ([.9, .9], [.01, .3]):
        region = PolytopeRegion(np.eye(2), bounds)
        monkeypatch.setattr(region, "maximize_linear", lambda *a: pytest.fail("not a specification score"))
        results.append(make_projection(name).select(outer, [], region, samples))
    assert results[0].weight == pytest.approx(results[1].weight)


@pytest.mark.parametrize("name", METHODS)
def test_lk_initialization_precision_and_reuse(name):
    strategy = SafetyVerifier(VerificationOptions(projection=name, sandwich_precision=.0123)).projection
    assert strategy.precision == .0123
    assert strategy.supports_inner_refinement and strategy.retain_all_support_cuts
    for count in (0, 1, 0):
        selection = strategy.select(OuterPolytope(2), [], PolytopeRegion(np.eye(2), [.9, .9]),
                                    [np.zeros(2)] * count)
        assert selection.weight == pytest.approx(np.eye(2)[count])
        assert selection.initializing and "lk" in selection.metric


@pytest.mark.parametrize("relative", [False, True])
def test_lk_threshold_is_in_its_own_indicator_units(relative):
    # Box outer approximation and singleton inner yield epsilon=.125
    # and gauge=1/3; test the <= stopping convention without a sqrt(d) factor.
    outer = OuterPolytope(2)
    outer.add_cut(np.array([1., 0.]), .5, "x")
    outer.add_cut(np.array([0., 1.]), .5, "y")
    samples = [np.array([.375, .375])] * 2
    region = PolytopeRegion(np.eye(2), [.3, .3])
    exact = 1. / 3. if relative else .125
    strategy = LammelKueferGaugeStrategy(exact) if relative else LammelKueferStrategy(exact)
    with pytest.raises(SandwichApproximationComplete) as error:
        strategy.select(outer, [], region, samples)
    assert error.value.maximum_gap == pytest.approx(exact)
    assert error.value.metric == f"{strategy.metric_prefix}-tolerance-check"
    strategy.precision = exact / 2
    assert strategy.select(outer, [], region, samples).distance == pytest.approx(exact)


def test_zero_support_is_not_regularized_or_silently_changed_to_absolute():
    outer = OuterPolytope(2)
    region = PolytopeRegion(np.eye(2), [.9, .9])
    points = [np.array([.4, 0.])] * 2
    with pytest.raises(RegionError, match="zero-support facet"):
        LammelKueferGaugeStrategy().select(outer, [], region, points)
    assert LammelKueferStrategy().select(outer, [], region, points).distance == pytest.approx(1.)


def test_tiny_positive_support_is_not_floored():
    points = [np.array([.4, 2e-13])] * 2
    selection = LammelKueferGaugeStrategy().select(
        OuterPolytope(2), [], PolytopeRegion(np.eye(2), [.9, .9]), points)
    assert selection.distance == pytest.approx(1 / 2e-13 - 1)
