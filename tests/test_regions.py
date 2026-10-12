import numpy as np
import pytest

from mdp_safety.formula import FormulaError, parse_polynomial_formula
from mdp_safety.projections import (
    inner_approximation_contains,
    make_projection,
    projection_names,
)
from mdp_safety.polytope import OuterPolytope
from mdp_safety.regions import PolytopeRegion, QuadraticRegion, RegionError, region_from_dict


@pytest.mark.parametrize("name", ["l2", "linf"])
def test_polytope_projection_gives_strict_nonnegative_separator(name: str) -> None:
    region = PolytopeRegion([[1.0, 1.0]], [0.6])
    separation = make_projection(name).separate(np.array([0.8, 0.2]), region)
    assert separation.distance > 0.0
    assert np.all(separation.weight >= 0.0)
    assert np.max(separation.weight) == pytest.approx(1.0)
    assert separation.margin > 0.0
    assert region.contains(separation.projected, tolerance=1e-8)


@pytest.mark.parametrize("name", ["l2", "linf"])
def test_quadratic_projection_gives_strict_separator(name: str) -> None:
    region = QuadraticRegion(np.eye(2), [0.0, 0.0], 0.0, 0.38)
    separation = make_projection(name).separate(np.array([0.6, 0.3]), region)
    assert separation.distance > 0.0
    assert separation.margin > 0.0
    assert region.contains(separation.projected, tolerance=1e-8)


def test_region_validation_rejects_non_downward_polytope() -> None:
    with pytest.raises(RegionError, match="downward closure"):
        PolytopeRegion([[-1.0, 0.0]], [-0.1])


def test_projection_registry_contains_all_implementations() -> None:
    assert projection_names() == ("gauge", "l2", "linf",
                                  "sandwich-lk", "sandwich-lk-gauge")


@pytest.mark.parametrize("method", ["sandwich-lk", "sandwich-lk-gauge"])
def test_sandwich_uses_outer_inner_gap(method) -> None:
    region = PolytopeRegion([[1.0, 1.0]], [0.9])
    outer = OuterPolytope(2)
    unsafe = [point for point in outer.vertices() if not region.contains(point)]
    separation = make_projection(method).select(
        outer, unsafe, region, [np.array([0.6, 0.2])] * 2
    )
    assert separation.distance > 0.0
    assert separation.margin > 0.0
    assert np.all(separation.weight >= 0.0)
    assert np.max(separation.weight) == pytest.approx(1.0)


def test_sandwich_inner_membership_uses_convex_downward_closure() -> None:
    samples = [np.array([0.8, 0.2]), np.array([0.2, 0.8])]
    assert inner_approximation_contains(np.array([0.5, 0.5]), samples, 1e-9)
    assert inner_approximation_contains(np.array([0.4, 0.4]), samples, 1e-9)
    assert not inner_approximation_contains(np.array([0.7, 0.7]), samples, 1e-9)


def test_one_dimensional_outer_polytope_vertices() -> None:
    outer = OuterPolytope(1)
    outer.add_cut(np.array([1.0]), 0.4, "test")
    assert outer.vertices().ravel().tolist() == pytest.approx([0.0, 0.4])


def test_formula_region_parses_mathematical_notation_and_parameter() -> None:
    region = region_from_dict(
        {
            "type": "formula",
            "formula": "x1^2 + 2*x2^2 ≤ theta",
            "parameters": {"theta": 0.5},
        },
        dimension_hint=2,
    )
    assert region.contains(np.array([0.5, 0.3]))
    assert not region.contains(np.array([0.8, 0.1]))
    assert region.value(np.array([0.5, 0.3])) == pytest.approx(0.43)
    assert region.gradient(np.array([0.5, 0.3])) == pytest.approx([1.0, 1.2])


def test_formula_parser_rejects_code_and_nonconvex_quadratic() -> None:
    with pytest.raises(FormulaError, match="unsupported formula syntax"):
        parse_polynomial_formula("__import__('os').system('id') <= 1", ["x1"])
    with pytest.raises(FormulaError, match="not convex"):
        parse_polynomial_formula("x1*x2 <= 1", ["x1", "x2"])
    with pytest.raises(FormulaError, match="must be finite"):
        parse_polynomial_formula("x1 <= theta", ["x1"], {"theta": float("inf")})


def test_formula_support_recovers_accurate_status_eight_boundary_point() -> None:
    region = region_from_dict(
        {
            "type": "formula",
            "formula": "(x1+x2+x3+x4)^2 + 3*x4^2 <= 0.65",
        },
        dimension_hint=4,
    )
    support, point = region.maximize_linear(np.array([1.0, 0.0, 0.0, 0.0]))
    assert region.contains(point)
    assert support == pytest.approx(np.sqrt(0.65), rel=1e-6)


@pytest.mark.parametrize("name", ["gauge"])
def test_support_based_strategies_give_strict_separator(name: str) -> None:
    region = region_from_dict(
        {"type": "formula", "formula": "x1^2 + x2^2 <= 0.38"},
        dimension_hint=2,
    )
    outer = OuterPolytope(2)
    unsafe = [point for point in outer.vertices() if not region.contains(point)]
    separation = make_projection(name).select(outer, unsafe, region)
    assert separation.margin > 0.0
    assert separation.distance > 0.0
    assert np.all(separation.weight >= 0.0)


def test_gauge_requires_absorbing_safe_region() -> None:
    region = PolytopeRegion([[1.0, 0.0]], [0.0])
    outer = OuterPolytope(2)
    unsafe = [point for point in outer.vertices() if not region.contains(point)]
    with pytest.raises(RegionError, match="rho times the simplex"):
        make_projection("gauge").select(outer, unsafe, region)
