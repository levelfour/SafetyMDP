import csv
from dataclasses import replace
import json
import subprocess
from pathlib import Path

import pytest

from mdp_safety import benchmark
from mdp_safety.benchmark_report import _method
from mdp_safety.experiments.checkpoint import _configuration, _validate_resume
from mdp_safety.experiments.models import BenchmarkSettings, DEFAULT_PROJECTIONS


def test_method_override_runs_only_lk_and_records_identity(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    content = json.dumps([{"name": "tiny-safe", "model": "model.pm", "specification": "safe.json",
                           "projections": ["l2", "linf"]}])
    manifest.write_text(content)
    calls = []

    def execute(command, timeout):
        projection = command[command.index("--projection") + 1]
        calls.append(projection)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text(json.dumps({
            "status": "SAFE", "projection": projection, "queries": [], "elapsed_seconds": .01,
            "oracle_call_count": 0, "weight_selection_call_count": 1,
            "weight_selection_seconds": .002, "weight_selection_cpu_seconds": .001,
            "iterations": 1, "storm_call_count": 0, "state_count": 3,
            "original_choice_count": 3, "storm_version": "test",
        }))
        return subprocess.CompletedProcess(command, 0, "", ""), False

    monkeypatch.setattr(benchmark, "_execute", execute)
    output = tmp_path / "results"
    arguments = [str(manifest), "--output-dir", str(output), "--projections", "sandwich-lk",
                 "sandwich-lk-gauge", "--repetitions", "2"]
    assert benchmark.main(arguments) == 0
    assert calls == ["sandwich-lk"] * 2 + ["sandwich-lk-gauge"] * 2
    assert manifest.read_text() == content
    config = json.loads((output / "benchmark_config.json").read_text())
    assert config["projection_override"] == ["sandwich-lk", "sandwich-lk-gauge"]
    assert config["sandwich_lk_implementations"] == {
        "sandwich-lk": "lk-2023-max-inner-facet-epsilon-v1",
        "sandwich-lk-gauge": "lk-2023-max-inner-facet-gauge-origin-v1",
    }
    rows = list(csv.DictReader((output / "summary.csv").open()))
    assert {r["projection"] for r in rows} == set(calls)
    assert all(float(r["weight_selection_seconds"]) == .002 for r in rows)
    assert benchmark.main(arguments + ["--resume"]) == 0
    assert len(calls) == 4


@pytest.mark.parametrize("field", ["projection_override", "sandwich_lk_implementations"])
def test_lk_resume_rejects_changed_method_or_implementation(tmp_path, field):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    settings = BenchmarkSettings(manifest, tmp_path, projection_override=("sandwich-lk",))
    config = _configuration(settings, state="finished", started_at="test", benchmark_wall_seconds=0,
                            jobs=0, completed_jobs=0)
    config[field] = None
    (tmp_path / "benchmark_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        _validate_resume(settings)


def test_lk_version_is_recorded_for_manifest_selected_method(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"name": "tiny", "model": "x.pm", "specification": "s.json",
                                     "projections": ["sandwich-lk"]}]))
    settings = BenchmarkSettings(manifest, tmp_path)
    config = _configuration(settings, state="finished", started_at="test", benchmark_wall_seconds=0,
                            jobs=1, completed_jobs=0)
    assert config["projection_override"] is None
    assert list(config["sandwich_lk_implementations"]) == ["sandwich-lk"]


def test_lk_defaults_have_distinct_report_labels(tmp_path):
    assert DEFAULT_PROJECTIONS == ("l2", "linf", "gauge", "sandwich-lk", "sandwich-lk-gauge")
    assert _method("sandwich-lk") == "Sandwich (LK epsilon)"
    assert _method("sandwich-lk-gauge") == "Sandwich (LK gauge)"
    settings = BenchmarkSettings(tmp_path / "x.json", tmp_path)
    for override in ((), ("no-such-strategy",), ("sandwich-lk", "sandwich-lk")):
        with pytest.raises(ValueError, match="projection override"):
            replace(settings, projection_override=override).validate()


def test_existing_non_lk_checkpoint_does_not_require_new_optional_fields(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    settings = BenchmarkSettings(manifest, tmp_path)
    config = _configuration(settings, state="finished", started_at="test", benchmark_wall_seconds=0,
                            jobs=0, completed_jobs=0)
    config.pop("projection_override")
    config.pop("sandwich_lk_implementations")
    (tmp_path / "benchmark_config.json").write_text(json.dumps(config))
    _validate_resume(settings)
