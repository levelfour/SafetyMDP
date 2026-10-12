"""All benchmark families are fixed inputs, with no model/spec generators."""
from pathlib import Path

import pytest

from scripts import fixed_inputs
from mdp_safety.benchmarking import load_manifest


@pytest.mark.parametrize("domain", fixed_inputs.FIXED_DOMAINS)
def test_fixed_assets_copy_byte_for_byte(tmp_path, domain):
    fixed_inputs.copy_fixed_inputs(tmp_path, "models", "specs", domains=(domain,))
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 12
    for path in files:
        assert path.read_bytes() == (fixed_inputs.INPUT_ROOT / path.relative_to(tmp_path)).read_bytes()


def test_fixed_assets_in_place_are_not_written(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("attempt to overwrite a fixed input")
    monkeypatch.setattr(fixed_inputs.shutil, "copyfile", forbidden)
    fixed_inputs.copy_fixed_inputs(fixed_inputs.INPUT_ROOT, "models", "specs")


@pytest.mark.parametrize("domain", fixed_inputs.FIXED_DOMAINS)
def test_missing_fixed_model_fails_without_recreating_it(tmp_path, monkeypatch, domain):
    source, output = tmp_path / "missing", tmp_path / "out"
    monkeypatch.setattr(fixed_inputs, "INPUT_ROOT", source)
    with pytest.raises(FileNotFoundError, match="missing fixed benchmark input"):
        fixed_inputs.copy_fixed_inputs(output, "models", domains=(domain,))
    assert not output.exists()


def test_all_three_families_have_complete_fixed_inputs_and_metadata():
    cases = load_manifest(fixed_inputs.MANIFEST_ROOT / "domains.json")
    assert len(cases) == 24
    assert len({case.model for case in cases}) == 12
    assert all(case.model.is_file() and case.specification.is_file() for case in cases)
    rows = fixed_inputs.fixed_metadata()
    assert len(rows) == 12
    for domain in fixed_inputs.FIXED_DOMAINS:
        selected = fixed_inputs.fixed_metadata(domain)
        assert selected == [row for row in rows if row["domain"] == domain]
        assert {row["dimension"] for row in selected} == set(fixed_inputs.FIXED_DIMENSIONS)


def test_retired_generation_pipeline_is_absent():
    root = Path(__file__).resolve().parents[1]
    assert not list((root / "benchmarks").rglob("*.py"))


def test_invalid_fixed_domain_or_section_fails_before_copy(tmp_path):
    with pytest.raises(ValueError, match="not fixed-input domains"):
        fixed_inputs.copy_fixed_inputs(tmp_path, "models", domains=("unavailable-domain",))
    with pytest.raises(ValueError, match="unknown fixed input section"):
        fixed_inputs.copy_fixed_inputs(tmp_path, "unexpected")
    with pytest.raises(ValueError, match="not a fixed-input domain"):
        fixed_inputs.fixed_metadata("unavailable-domain")
    assert not list(tmp_path.iterdir())
