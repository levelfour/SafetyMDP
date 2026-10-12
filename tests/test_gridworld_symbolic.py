"""Fixed Gridworld inputs: canonical paths and an exact transition fingerprint."""
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
# Pinned fingerprints cover states, labels, actions and rational transitions.
# Coordinate canonicalization makes them independent of Storm state numbering.
GRAPH_SHA256 = {
    4: "66522077ca12b18b181c20a2a42ed15321a8755d0df0308fe016e6e7ddec13f2",
    6: "632deabda7f3ef296cfb55ca80ba720c5454ccefc9dc539168fc0392e9e84bbe",
    8: "92705e6de8f3a86e066a92f38926530172fec9cfc0de879d39bb5f5e6f4aa5b6",
    10: "76bdc1f6c29b77a3804379e7016ca3db406a4e91adefb39718de7cecee6c4a74",
}


def _canonical_model(states):
    coordinates = {}
    for identifier, state in states.items():
        val = state["valuation"]
        coordinates[identifier] = (val["row"], val["col"])
    assert len(set(coordinates.values())) == len(states)
    model = {}
    for identifier, state in states.items():
        actions = {}
        for name, transitions in state["actions"].items():
            assert sum(transitions.values()) == 1
            assert all(probability > 0 for probability in transitions.values())
            actions[name] = {coordinates[d]: p for d, p in transitions.items()}
        model[coordinates[identifier]] = (state["labels"], actions)
    return model


@pytest.mark.parametrize("dimension", [4, 6, 8, 10])
def test_gridworld_has_two_coordinates_and_five_commands(dimension):
    path = ROOT / f"benchmarks/models/gridworld-d{dimension}.prism"
    text = path.read_text()
    assert re.findall(r"^\s+(\w+) : \[", text, re.M) == ["row", "col"]
    assert re.findall(r"^\s+\[(\w+)\]", text, re.M) == ["north", "east", "south", "west", "absorb"]
    assert len(text.splitlines()) <= 90


def test_all_gridworld_manifests_use_the_same_four_files():
    folder = ROOT / "benchmarks"
    manifests = list((ROOT / "experiments/manifests").glob("domains*.json"))
    found = set()
    for manifest in manifests:
        for case in load_manifest(manifest):
            match = re.fullmatch(r"gridworld-d(4|6|8|10)-(safe|unsafe)", case.name)
            if match:
                dimension = int(match[1])
                assert case.model == (folder / f"models/gridworld-d{dimension}.prism").resolve()
                found.add(dimension)
    assert found == {4, 6, 8, 10}


@pytest.mark.skipif(STORM is None, reason="Storm is not installed")
@pytest.mark.parametrize("dimension", [4, 6, 8, 10])
def test_storm_exact_fixed_model_matches_verified_graph(dimension, tmp_path):
    metadata = next(item for item in fixed_metadata("gridworld") if item["dimension"] == dimension)
    symbolic = ROOT / f"benchmarks/models/gridworld-d{dimension}.prism"
    new = _canonical_model(export_exact(symbolic, tmp_path / "symbolic.drn"))
    canonical = [{"cell": cell, "labels": sorted(labels),
                  "actions": {a: [[dest, str(p)] for dest, p in sorted(transitions.items())]
                              for a, transitions in sorted(actions.items())}}
                 for cell, (labels, actions) in sorted(new.items())]
    assert hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest() == GRAPH_SHA256[dimension]
    assert len(new) == metadata["state_count"] == 107
    assert sum(len(actions) for _, actions in new.values()) == metadata["choice_count"]
    assert sum(len(p) for _, actions in new.values() for p in actions.values()) == metadata["transition_count"]
    assert all(tuple(cell) not in new for cell in metadata["design"]["obstacle_cells"])
    assert "init" in new[(5, 5)][0]
    for i, cell in enumerate(metadata["design"]["hazard_cells"], 1):
        labels, actions = new[tuple(cell)]
        assert labels == {f"hazard_{i}"}
        assert actions == {"absorb": {tuple(cell): Fraction(1)}}
    assert new[(10, 10)] == ({"safe_completion"}, {"absorb": {(10, 10): Fraction(1)}})
