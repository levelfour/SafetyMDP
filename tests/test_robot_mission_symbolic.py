"""Fixed task-bit models, checked against pinned exact transition fingerprints."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import pytest

from scripts.fixed_inputs import fixed_metadata
from exact_prism import STORM, export_exact
from mdp_safety.benchmarking import load_manifest

ROOT = Path(__file__).resolve().parents[1]
# Pinned digests include every state, label, action and rational transition.
# Task-set/weather canonicalization removes dependence on Storm state numbering.
GRAPH_SHA256 = {
    4: "a835feaa4b3d786918adcdf607a39733ab9b5db6477f90bf10e0c3c6fbfbf9fd",
    6: "fedbba93f04ce5ed813a1f1fb36cf56dae0d4a5f36aeec546e25d3055bd85cfc",
    8: "bd207619333ab79ddb1b184e72216a6d5d7be15506b2314cae3714046de53516",
    10: "337efa2ed7dce79f5dae8be62873b12bba51b8deb7cd1e13ecc8afe9a7438bc5",
}


def canonical_graph(states, dimension):
    keys = {}
    for identifier, state in states.items():
        val = state["valuation"]
        outcome = val["outcome"]
        assert set(val) == {"outcome", "weather"} | {f"done{i}" for i in range(1, dimension + 1)}
        if outcome:
            assert 1 <= outcome <= dimension + 1
            assert val["weather"] == 0
            assert all(val[f"done{i}"] == 1 for i in range(1, dimension + 1))
            keys[identifier] = f"terminal:{outcome}"
        else:
            mask = sum(1 << i for i in range(dimension) if not val[f"done{i+1}"])
            assert mask > 0 and val["weather"] in (0, 1)
            keys[identifier] = f"running:{mask}:{val['weather']}"
    assert len(set(keys.values())) == len(states)
    graph = {}
    for identifier, state in states.items():
        actions = {}
        for name, transitions in state["actions"].items():
            assert sum(transitions.values()) == 1
            assert all(p > 0 for p in transitions.values())
            actions[name] = {keys[dest]: str(p) for dest, p in transitions.items()}
        graph[keys[identifier]] = {"labels": sorted(state["labels"]), "actions": actions}
    return graph


@pytest.mark.parametrize("dimension", [4, 6, 8, 10])
def test_robot_uses_completion_bits_and_two_commands_per_task(dimension):
    text = (ROOT / f"benchmarks/models/robot-mission-d{dimension}.prism").read_text()
    assert re.findall(r"^\s+(\w+) : bool init false;", text, re.M) == [f"done{i}" for i in range(1, dimension + 1)]
    assert re.findall(r"^\s+(\w+) : \[", text, re.M) == ["weather", "outcome"]
    assert len(re.findall(r"^\s+\[", text, re.M)) == 2 * dimension + 1
    assert len(text.splitlines()) <= 22 * dimension + 40


def test_all_robot_manifests_use_the_same_four_files():
    folder = ROOT / "benchmarks"
    manifests = list((ROOT / "experiments/manifests").glob("domains*.json"))
    found = set()
    for manifest in manifests:
        for case in load_manifest(manifest):
            match = re.fullmatch(r"robot-mission-d(4|6|8|10)-(safe|unsafe)", case.name)
            if match:
                dimension = int(match[1])
                assert case.model == (folder / f"models/robot-mission-d{dimension}.prism").resolve()
                found.add(dimension)
    assert found == {4, 6, 8, 10}


@pytest.mark.skipif(STORM is None, reason="Storm is not installed")
@pytest.mark.parametrize("dimension", [4, 6, 8, 10])
def test_storm_exact_fixed_robot_matches_original_graph(dimension, tmp_path):
    metadata = next(item for item in fixed_metadata("robot-mission") if item["dimension"] == dimension)
    prism = ROOT / f"benchmarks/models/robot-mission-d{dimension}.prism"
    graph = canonical_graph(export_exact(prism, tmp_path / "robot.drn"), dimension)
    assert hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest() == GRAPH_SHA256[dimension]
    assert len(graph) == metadata["state_count"] == 2 ** (dimension + 1) + dimension - 1
    assert sum(len(s["actions"]) for s in graph.values()) == metadata["choice_count"]
    assert sum(len(t) for s in graph.values() for t in s["actions"].values()) == metadata["transition_count"]
    assert graph[f"running:{2**dimension-1}:0"]["labels"] == ["init"]
    for i in range(1, dimension + 2):
        key = f"terminal:{i}"
        label = f"task_{i}_catastrophic_failure" if i <= dimension else "safe_completion"
        assert graph[key] == {"labels": [label], "actions": {"absorb": {key: str(Fraction(1))}}}


@pytest.mark.skipif(STORM is None, reason="Storm is not installed")
@pytest.mark.parametrize("status", ["safe", "unsafe"])
def test_task_bit_model_runs_through_real_verification_pipeline(status, tmp_path):
    from mdp_safety.regions import region_from_dict
    from mdp_safety.verifier import SafetyVerifier, VerificationOptions
    folder = ROOT / "benchmarks"
    spec = json.loads((folder / f"specs/robot-mission-d4-{status}.json").read_text())
    result = SafetyVerifier(VerificationOptions(projection="gauge", max_iterations=12)).verify(
        folder / "models/robot-mission-d4.prism", spec["terminal_labels"],
        region_from_dict(spec["safe_region"]), working_directory=tmp_path,
    )
    assert result.status == status.upper()
    assert result.oracle_call_count > 0
    if status == "unsafe":
        assert result.witness["safe_region_violation_at_lower"] > 0
