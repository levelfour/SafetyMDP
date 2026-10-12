from pathlib import Path
import shutil

import pytest

from mdp_safety.regions import region_from_dict
from mdp_safety.verifier import SafetyVerifier, VerificationOptions


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("storm") is None, reason="Storm is not installed")


@pytest.mark.parametrize("projection", ["l2", "linf", "gauge", "sandwich-lk", "sandwich-lk-gauge"])
@pytest.mark.parametrize("specification,expected", [("safe_formula.json", "SAFE"), ("unsafe_polytope.json", "UNSAFE")])
def test_cli_without_storm_timeout(projection, specification, expected, capsys):
    import json
    from mdp_safety.cli import main
    assert main([str(ROOT / "tests/fixtures/smoke/two_terminal.pm"), str(ROOT / "tests/fixtures/smoke" / specification),
                 "--projection", projection, "--timeout", "none", "--max-iterations", "50"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == expected
    assert result["iterations"] <= 50
    assert result["options"]["storm_timeout_seconds"] is None


def _run(
    specification: str,
    projection: str,
    tmp_path: Path,
    record_scheduler_actions: bool = True,
):
    import json

    spec = json.loads((ROOT / "tests/fixtures/smoke" / specification).read_text(encoding="utf-8"))
    region = region_from_dict(spec["safe_region"])
    verifier = SafetyVerifier(
        VerificationOptions(
            projection=projection,
            storm_precision=1e-7,
            minimum_storm_precision=1e-10,
            max_iterations=20,
            record_scheduler_actions=record_scheduler_actions,
        )
    )
    return verifier.verify(
        ROOT / "tests/fixtures/smoke" / "two_terminal.pm",
        spec["terminal_labels"],
        region,
        working_directory=tmp_path,
    )


@pytest.mark.parametrize("projection", ["l2", "linf", "gauge", "sandwich-lk", "sandwich-lk-gauge"])
def test_safe_polytope_end_to_end(projection: str, tmp_path: Path) -> None:
    result = _run("safe_polytope.json", projection, tmp_path)
    assert result.status == "SAFE"
    assert result.outer_halfspaces
    assert len(result.pareto_samples) == 2
    assert result.storm_version
    assert result.oracle_call_count == len(result.queries)
    assert result.weight_selection_call_count == len(result.selections)
    assert result.weight_selection_call_count > 0
    assert result.weight_selection_seconds > 0.0
    assert result.weight_selection_cpu_seconds > 0.0
    assert all(query.support_upper >= query.support_estimate for query in result.queries)


@pytest.mark.parametrize("projection", ["l2", "linf", "gauge", "sandwich-lk", "sandwich-lk-gauge"])
def test_unsafe_polytope_has_scheduler_witness(projection: str, tmp_path: Path) -> None:
    result = _run("unsafe_polytope.json", projection, tmp_path)
    assert result.status == "UNSAFE"
    assert result.witness is not None
    assert result.witness["safe_region_violation_at_lower"] > 0.0


@pytest.mark.parametrize("projection", ["l2", "linf", "gauge", "sandwich-lk", "sandwich-lk-gauge"])
def test_formula_region_end_to_end(projection: str, tmp_path: Path) -> None:
    spec = {
        "type": "formula",
        "formula": "x1^2 + x2^2 <= theta",
        "parameters": {"theta": 0.38},
    }
    result = SafetyVerifier(
        VerificationOptions(projection=projection, max_iterations=20)
    ).verify(
        ROOT / "tests/fixtures/smoke" / "two_terminal.pm",
        ["terminal_left", "terminal_right"],
        region_from_dict(spec, dimension_hint=2),
        working_directory=tmp_path,
    )
    assert result.status == "SAFE"
    assert result.oracle_call_count > 0


def test_scheduler_actions_can_be_omitted_from_artifact(tmp_path: Path) -> None:
    result = _run(
        "unsafe_polytope.json",
        "l2",
        tmp_path,
        record_scheduler_actions=False,
    )
    assert result.status == "UNSAFE"
    assert result.witness is not None
    assert result.witness["scheduler_actions"] == {}
    assert all(query.scheduler_actions == {} for query in result.queries)


@pytest.mark.parametrize("projection", ["sandwich-lk", "sandwich-lk-gauge"])
def test_sandwich_cyclic_six_terminal_initialization_and_facet(tmp_path, projection):
    import numpy as np
    from mdp_safety.regions import PolytopeRegion
    dimension = 6
    lines = ["mdp", "module cyclic", "s : [0..7] init 0;"]
    # Six reachable terminal labels but only two competing actions: enough
    # to require a non-coordinate facet without an expensive performance run.
    for dominant in range(2):
        targets = [f"{7 if i == dominant else 1}/40:(s'={i+1})" for i in range(dimension)]
        lines.append(f"[a{dominant}] s=0 -> 1/2:(s'=0) + " + " + ".join(targets) + " + 1/5:(s'=7);")
    lines.extend(["[] s>0 -> (s'=s);", "endmodule"])
    labels = [f"bad{i}" for i in range(dimension)]
    lines.extend(f'label "{label}" = s={i+1};' for i, label in enumerate(labels))
    model = tmp_path / "cyclic.pm"
    model.write_text("\n".join(lines))
    result = SafetyVerifier(VerificationOptions(projection=projection, max_iterations=12)).verify(
        model, labels, PolytopeRegion([np.ones(dimension)], [.7]),
    )
    assert result.status == "SAFE"
    assert [q.weight for q in result.queries[:dimension]] == np.eye(dimension).tolist()
    metric = "max-inner-facet"
    assert any(metric in q.selection_metric for q in result.queries)
    assert all(np.linalg.norm(q.weight) == pytest.approx(1.) for q in result.queries)
