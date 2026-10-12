"""Guard the separation between benchmark inputs and experiment machinery."""
from pathlib import Path

from mdp_safety.benchmarking import load_manifest

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "benchmarks"
MANIFESTS = ROOT / "experiments/manifests"
METADATA = ROOT / "experiments/metadata"


def test_paper_inventory():
    actual = {str(p.relative_to(ROOT))
              for folder in ("models", "specs")
              for p in (INPUTS / folder).iterdir() if p.is_file()}
    assert len(actual) == 36
    assert len(list((INPUTS / "models").glob("*.prism"))) == 12
    assert len(list((INPUTS / "specs").glob("*.json"))) == 24
    cases = load_manifest(MANIFESTS / "domains.json")
    names = {case.name for case in cases}
    assert len(cases) == len(names) == 24
    assert {case.suite for case in cases} == {"domains"}
    assert {case.family for case in cases} == {"gridworld", "robot-mission", "network-routing"}
    referenced = {str(p.relative_to(ROOT)) for c in cases for p in (c.model, c.specification)}
    assert referenced == actual
    safe = load_manifest(MANIFESTS / "domains_safe.json")
    unsafe = load_manifest(MANIFESTS / "domains_unsafe.json")
    assert len(safe) == len(unsafe) == 12
    assert all(c.expected_status == "SAFE" for c in safe)
    assert all(c.expected_status == "UNSAFE" for c in unsafe)
    assert not ({c.name for c in safe} & {c.name for c in unsafe})
    assert {c.name for c in safe + unsafe} == names
    assert {c.suite for c in safe + unsafe} == {"domains"}


def test_benchmarks_contain_only_models_and_specifications():
    assert {p.name for p in INPUTS.iterdir()} == {"models", "specs"}
    for folder, extension, count in (("models", ".prism", 12), ("specs", ".json", 24)):
        files = list((INPUTS / folder).iterdir())
        assert len(files) == count
        assert all(p.is_file() and p.suffix == extension for p in files)
    assert {p.name for p in (ROOT / "experiments").iterdir()} == {
        "paper.json", "README.md", "manifests", "metadata"}
    assert {p.name for p in MANIFESTS.iterdir()} == {
        "domains.json", "domains_safe.json", "domains_unsafe.json"}
    assert {p.name for p in METADATA.iterdir()} == {
        "model_metadata.json", "specification_metadata.json"}
