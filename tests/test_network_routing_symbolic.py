"""Fixed node/congestion models and exact transition fingerprints."""
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
GRAPH_SHA256 = {
    4: "14c969ead14a79f1bd6a34675ff990c586ffb080dfe2746c2b2c23168e7bd555",
    6: "3d78323c3a09adcce1d2191301161a849738519d1076fb26bcdfea476b6a9cfe",
    8: "e9494cade64ccabfe3d44411efb670cff106c7b884e9cdb7a12e33d2cdff098c",
    10: "85c502962e511e5415d9dbddd2d4742e81e8798ef41130e06291e6948b4800f4",
}


def canonical_graph(states, dimension):
    limit = 3 * dimension + 4
    keys = {}
    for identifier, state in states.items():
        val = state["valuation"]
        assert set(val) == {"node", "congestion"}
        assert 0 <= val["node"] <= limit + dimension
        assert val["congestion"] in (0, 1)
        if val["node"] >= limit:
            assert val["congestion"] == 0
        keys[identifier] = f"{val['node']}:{val['congestion']}"
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
def test_network_has_two_variables_and_four_commands(dimension):
    text = (ROOT / f"benchmarks/models/network-routing-d{dimension}.prism").read_text()
    assert re.findall(r"^\s+(\w+) : \[", text, re.M) == ["node", "congestion"]
    assert re.findall(r"^\s+\[(\w+)\]", text, re.M) == ["low_latency", "balanced", "secure_detour", "absorb"]
    assert len(text.splitlines()) <= 85


def test_all_network_manifests_use_the_same_four_files():
    folder = ROOT / "benchmarks"
    manifests = list((ROOT / "experiments/manifests").glob("domains*.json"))
    found = set()
    for manifest in manifests:
        for case in load_manifest(manifest):
            match = re.fullmatch(r"network-routing-d(4|6|8|10)-(safe|unsafe)", case.name)
            if match:
                dimension = int(match[1])
                assert case.model == (folder / f"models/network-routing-d{dimension}.prism").resolve()
                found.add(dimension)
    assert found == {4, 6, 8, 10}


@pytest.mark.skipif(STORM is None, reason="Storm is not installed")
@pytest.mark.parametrize("dimension", [4, 6, 8, 10])
def test_storm_exact_network_matches_verified_fingerprint(dimension, tmp_path):
    metadata = next(row for row in fixed_metadata("network-routing") if row["dimension"] == dimension)
    prism = ROOT / f"benchmarks/models/network-routing-d{dimension}.prism"
    graph = canonical_graph(export_exact(prism, tmp_path / "network.drn"), dimension)
    assert hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest() == GRAPH_SHA256[dimension]
    assert len(graph) == metadata["state_count"] == 7 * dimension + 9
    assert sum(len(s["actions"]) for s in graph.values()) == metadata["choice_count"]
    assert sum(len(t) for s in graph.values() for t in s["actions"].values()) == metadata["transition_count"]
    assert graph["0:0"]["labels"] == ["init"]
    limit = 3 * dimension + 4
    for node in range(limit):
        for congestion in (0, 1):
            assert set(graph[f"{node}:{congestion}"]["actions"]) == {"low_latency", "balanced", "secure_detour"}
    for i in range(dimension + 1):
        key = f"{limit+i}:0"
        label = f"failure_class_{i}" if i else "safe_completion"
        assert graph[key] == {"labels": [label], "actions": {"absorb": {key: str(Fraction(1))}}}


@pytest.mark.skipif(STORM is None, reason="Storm is not installed")
@pytest.mark.parametrize("status", ["safe", "unsafe"])
def test_node_model_runs_through_real_verification_pipeline(status, tmp_path):
    from mdp_safety.regions import region_from_dict
    from mdp_safety.verifier import SafetyVerifier, VerificationOptions
    folder = ROOT / "benchmarks"
    spec = json.loads((folder / f"specs/network-routing-d4-{status}.json").read_text())
    result = SafetyVerifier(VerificationOptions(projection="gauge", max_iterations=20)).verify(
        folder / "models/network-routing-d4.prism", spec["terminal_labels"],
        region_from_dict(spec["safe_region"]), working_directory=tmp_path,
    )
    assert result.status == status.upper()
    assert result.oracle_call_count > 0
    if status == "unsafe":
        assert result.witness["safe_region_violation_at_lower"] > 0
