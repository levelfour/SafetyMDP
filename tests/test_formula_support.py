"""Standalone numerical support regressions; no benchmark inputs are required."""

from types import SimpleNamespace

import numpy as np
import pytest

import mdp_safety.safe_regions.polynomial as polynomial
from mdp_safety.projections import make_projection
from mdp_safety.regions import RegionError, region_from_dict


def anisotropic_region():
    # Six-dimensional polynomial with strongly unequal coordinate scales.
    # This standalone solver fixture contains no MDP or evaluation results.
    formula = (
        "63.11332877279619*x1 + 68.37338158479663*x2 + 70.39617498663578*x3 + "
        "0.6855925042682334*x4 + 0.5511227803432597*x5 + 0.6884059513104055*x6 + "
        "12393.454834862114*x1^2 + 14614.239204201236*x2^2 + 14982.939419262482*x3^2 + "
        "1.5960221517901965*x4^2 + 0.8705310170704699*x5^2 + 1.6251188733939579*x6^2 + "
        "15315.910063319683*x1*x2 + 12767.938967133794*x1*x3 + 20.877936328892034*x1*x4 + "
        "20.877936328892034*x1*x5 + 50.489093129165056*x1*x6 + 16287.383099965795*x2*x3 + "
        "20.877936328892034*x2*x4 + 20.877936328892034*x2*x5 + 20.877936328892034*x2*x6 + "
        "69.35168840998055*x3*x4 + 20.877936328892034*x3*x5 + 20.877936328892034*x3*x6 + "
        "0.7816940429396252*x4*x5 + 0.5115854655388298*x4*x6 + 0.936193243975564*x5*x6 <= 1"
    )
    return region_from_dict({"type": "formula", "formula": formula},
                            dimension_hint=6, tolerance=1e-10)


@pytest.mark.parametrize("norm", ["l2", "linf"])
def test_anisotropic_axis_direction_passes_existing_certificate(norm):
    region = anisotropic_region()
    point = np.eye(6)[5]
    selection = make_projection(norm).separate(point, region)
    # Positive coefficients make the axis intercept the exact coordinate support.
    linear = region.gradient(np.zeros(6))[5]
    quadratic = region.formula.hessian(np.zeros(6))[5, 5] / 2.0
    expected = 2.0 / (linear + np.sqrt(linear * linear + 4.0 * quadratic))
    assert selection.safe_support == pytest.approx(expected, abs=2e-13)
    assert region.contains(selection.support_point, tolerance=0.0)
    assert selection.distance == pytest.approx(1.0 - expected, abs=1e-10)


def test_status_eight_recovery_does_not_overshoot_support(monkeypatch):
    region = anisotropic_region()
    raw = np.array([0., 0., 0., 0., 0., 0.6007252913522672])
    monkeypatch.setattr(polynomial, "minimize", lambda *a, **kw: SimpleNamespace(
        x=raw.copy(), success=False, status=8, message="Positive directional derivative for linesearch"))
    support, point = region.maximize_linear(np.eye(6)[5])
    assert support == pytest.approx(0.6007249367089185, abs=2e-13)
    assert region.contains(point, tolerance=0.0)
    assert raw[5] - point[5] < 4e-7  # Feasibility recovery must not overshoot the boundary.


@pytest.mark.parametrize("success", [True, False])
def test_feasible_but_nonoptimal_solver_point_is_not_accepted(monkeypatch, success):
    region = region_from_dict({"type": "formula", "formula": "x1^2+x2^2<=.25"},
                              dimension_hint=2, tolerance=1e-10)
    monkeypatch.setattr(polynomial, "minimize", lambda *a, **kw: SimpleNamespace(
        x=np.zeros(2), success=success, message="injected nonoptimal point"))
    with pytest.raises(RegionError, match="support optimality could not be validated"):
        region.maximize_linear(np.array([1., 1.]))


def test_scaled_retry_after_nonoptimal_first_solve(monkeypatch):
    region = region_from_dict({"type": "formula", "formula": "10000*x1^2+x2^2<=.25"},
                              dimension_hint=2, tolerance=1e-10)
    original = polynomial.minimize
    calls = []
    def injected(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(x=np.zeros(2), success=False, message="injected failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(polynomial, "minimize", injected)
    support, point = region.maximize_linear(np.array([1., 1.]))
    assert len(calls) == 2
    assert support == pytest.approx(0.5 * np.sqrt(1.0001), abs=1e-9)
    assert region.contains(point, tolerance=0.0)


@pytest.mark.parametrize("factor", [1e-8, 1., 1e8])
@pytest.mark.parametrize("weight", [[1., 1.], [0., 1.], [-1., 1.]])
def test_anisotropic_support_matches_closed_form_and_weight_scaling(factor, weight):
    region = region_from_dict({"type": "formula", "formula": "10000*x1^2+x2^2<=.25"},
                              dimension_hint=2, tolerance=1e-10)
    w = np.array(weight)
    expected = 0.5 * np.sqrt(max(w[0], 0.)**2 / 10000. + max(w[1], 0.)**2)
    value, point = region.maximize_linear(factor * w)
    assert value / factor == pytest.approx(expected, abs=1e-9)
    assert value == float((factor * w) @ point)
    assert region.contains(point, tolerance=0.0)


@pytest.mark.parametrize("formula", ["x1+x2<=.4", "x1^4+x2^4<=.0625",
                                     "x1^2+x2^2+2<=2.25"])
def test_support_audit_covers_linear_higher_degree_and_constant_terms(formula):
    region = region_from_dict({"type": "formula", "formula": formula},
                              dimension_hint=2, tolerance=1e-10)
    value, point = region.maximize_linear(np.array([1., 0.]))
    expected = .4 if formula.startswith("x1+") else .5
    assert value == pytest.approx(expected, abs=1e-9)
    assert region.contains(point, tolerance=0.0)


def test_zero_weight_and_interior_recovery_preserve_feasibility():
    region = anisotropic_region()
    value, point = region.maximize_linear(np.zeros(6))
    assert value == 0.0
    np.testing.assert_array_equal(point, np.zeros(6))
    interior = np.full(6, 1e-4)
    np.testing.assert_array_equal(region._strictly_feasible_copy(interior), interior)
