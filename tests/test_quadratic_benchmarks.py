"""Numerical labels and curvature of every paper specification."""
import json
from pathlib import Path

import numpy as np
import pytest

from mdp_safety.benchmarking import load_manifest
from mdp_safety.regions import FormulaRegion, region_from_dict
from mdp_safety.polytope import OuterPolytope
from mdp_safety.projections import make_projection

ROOT = Path(__file__).resolve().parents[1]


def test_all_fixed_default_specs_are_genuinely_curved():
    folder = ROOT / "experiments"
    cases = load_manifest(folder / "manifests/domains.json")
    metadata = {item["name"]: item for item in json.loads((folder / "metadata/specification_metadata.json").read_text())}
    assert len(cases) == 24
    for case in cases:
        assert case.specification.parent.name == "specs"
        region = region_from_dict(json.loads(case.specification.read_text())["safe_region"])
        assert isinstance(region, FormulaRegion) and region.formula.degree == 2
        # Full-rank curvature, positive gradient coefficients, and a curved
        # boundary cutting the simplex rather than a redundant quadratic.
        matrix = region.formula.hessian(np.zeros(region.dimension)) / 2
        assert np.linalg.eigvalsh(matrix)[0] > 0
        assert np.min(matrix) >= 0 and np.min(region.gradient(np.zeros(region.dimension))) > 0
        assert region.contains(np.zeros(region.dimension), tolerance=0)
        assert max(region.value(e) for e in np.eye(region.dimension)) > 1.01
        item = metadata[case.name]
        assert case.expected_status == item["expected_status"]
        c = item["calibration"]
        rows, budgets = np.asarray(item["rows"]), np.asarray(c["safe_budgets"])
        normalized = rows / np.asarray(item["budgets"])[:, None]
        expected_matrix = normalized.T @ normalized / (2 * len(rows) * c["raw_threshold"])
        expected_linear = np.mean(normalized, axis=0) / (2 * c["raw_threshold"])
        np.testing.assert_allclose(matrix, expected_matrix, rtol=1e-14)
        np.testing.assert_allclose(region.gradient(np.zeros(region.dimension)), expected_linear, rtol=1e-14)
        u = np.asarray(c["row_support_uppers"])
        weight = rows.T @ (u / budgets**2 + 1 / budgets) / (2 * len(rows))
        np.testing.assert_allclose(weight, c["linear_majorant"]["weight"], rtol=1e-14)
        assert c["raw_threshold"] > c["linear_majorant"]["upper"]
        if case.expected_status == "UNSAFE":
            assert region.violation(np.asarray(item["unsafe_witness"]["lower"])) > .0099
        else:
            point = np.asarray(c["linear_majorant"]["point"])
            assert region.contains(point, tolerance=0)


@pytest.mark.parametrize("method", ["l2", "linf", "gauge"])
def test_initial_weight_selection_on_every_fixed_specification(method):
    # Geometry compatibility only: no Storm oracle or performance experiment.
    for case in load_manifest(ROOT / "experiments/manifests/domains.json"):
        region = region_from_dict(json.loads(case.specification.read_text())["safe_region"], tolerance=1e-10)
        outer = OuterPolytope(region.dimension, 1e-10)
        candidates = [p for p in outer.vertices() if not region.contains(p, tolerance=1e-8)]
        strategy = make_projection(method)
        selection = strategy.select(outer, candidates, region)
        audited = strategy.audit_selection(outer, region, selection)
        assert audited.margin > 0, case.name
