"""Study orchestration tests; never run performance experiments."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts import run_study as study

ROOT = Path(__file__).resolve().parents[1]


def profile(all_methods=True):
    return json.loads(study.DEFAULT_PROFILE.read_text())


def test_paper_plan_covers_exactly_120_jobs(tmp_path):
    config = profile()
    suites = study.plan(ROOT, tmp_path / "absent", config, "missing-storm", False)
    assert [(s.key, s.instances, s.jobs) for s in suites] == [("domains", 24, 120)]
    settings = suites[0].settings
    assert settings.repetitions == 1
    assert settings.run_timeout_seconds is settings.storm_timeout_seconds is None
    assert settings.projection_override == ("l2", "linf", "gauge", "sandwich-lk", "sandwich-lk-gauge")
    assert settings.precision == 1e-7 and settings.geometry_tolerance == 1e-8
    assert settings.minimum_precision == 1e-11
    assert settings.sandwich_precision == 1e-6 and settings.max_iterations == 50
    from dataclasses import replace
    from mdp_safety.experiments.manifest import load_manifest, build_jobs
    cases = load_manifest(settings.manifest)
    jobs = build_jobs([replace(c, projections=settings.projection_override) for c in cases],
                      settings.repetitions)
    assert len(jobs) == 120
    assert len({job.key for job in jobs}) == 120
    expected_names = [f"{family}-d{dimension}-{status}"
                      for family in ("gridworld", "robot-mission", "network-routing")
                      for dimension in (4, 6, 8, 10)
                      for status in ("safe", "unsafe")]
    assert [case.name for case in cases] == expected_names
    assert [job.key for job in jobs] == [
        (name, method, 1) for name in expected_names for method in config["methods"]]
    for status in ("safe", "unsafe"):
        subset = load_manifest(ROOT / f"experiments/manifests/domains_{status}.json")
        assert [case.name for case in subset] == [name for name in expected_names if name.endswith("-" + status)]
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("mode", [[], ["--dry-run"]])
def test_plan_only_never_prepares_outputs_or_invokes_storm(tmp_path, monkeypatch, capsys, mode):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run must not invoke Storm or create output")
    monkeypatch.setattr(study, "prepare", forbidden)
    monkeypatch.setattr(study, "run_benchmark", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    output = tmp_path / "absent"
    assert study.main(["--output", str(output), "--storm", "missing-storm"] + mode) == 0
    text = capsys.readouterr().out
    assert "Total: 120 runs" in text
    assert "seed" not in text.lower()
    assert "max refinements: 50" in text
    assert "run timeout: disabled; Storm-call timeout: disabled" in text
    assert not output.exists()


def test_default_output_is_outside_input_tree(capsys):
    assert study.main([]) == 0
    assert f"output: {ROOT / 'results/paper'}\n" in capsys.readouterr().out


@pytest.mark.parametrize("field,value", [
    ("repetitions", 0), ("repetitions", 1.5), ("repetitions", True),
    ("run_timeout_seconds", 0), ("storm_timeout_seconds", -1),
    ("run_timeout_seconds", float("inf")), ("storm_timeout_seconds", float("nan")),
])
def test_invalid_limits_are_rejected(tmp_path, field, value):
    config = profile()
    config[field] = value
    with pytest.raises(ValueError, match=field):
        study.plan(ROOT, tmp_path, config, "storm", False)


def tiny_study(tmp_path, all_methods=False):
    config = profile(all_methods)
    # Keep multi-suite orchestration coverage independent of the shipped suite.
    template = config["suites"][0]
    config["suites"] = [dict(template, key=key, label=f"{key} fixture")
                       for key in ("first", "second")]
    for entry in config["suites"]:
        name = entry["key"]
        model = tmp_path / f"{name}.pm"
        model.write_text("mdp\n")
        specification = tmp_path / f"{name}-safe.json"
        specification.write_text("{}")
        manifest = tmp_path / f"{name}.json"
        manifest.write_text(json.dumps({"instances": [{
            "name": name, "model": str(model), "specification": str(specification),
            "projections": ["l2"], "expected_status": "SAFE", "family": "test"}]}))
        entry["manifest"] = str(manifest)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(config))
    return path, config


@pytest.mark.parametrize("all_methods", [False, True])
def test_execute_and_resume_use_one_repeat_50_refinements_and_no_timeouts(tmp_path, monkeypatch, all_methods):
    path, config = tiny_study(tmp_path, all_methods)
    output = tmp_path / "results"
    original_runner = study.run_benchmark
    calls = []
    # Use the Python executable as a harmless binary identity fixture, not real Storm.
    monkeypatch.setattr(study.shutil, "which", lambda name: sys.executable)

    def executor(command, timeout):
        calls.append((command, timeout))
        assert timeout is None
        assert command[command.index("--timeout") + 1] == "none"
        assert command[command.index("--max-iterations") + 1] == "50"
        assert command[command.index("--projection") + 1] in config["methods"]
        return subprocess.CompletedProcess(command, 3, "", "mock verifier error"), False

    monkeypatch.setattr(study, "run_benchmark", lambda settings: original_runner(settings, executor=executor))
    if all_methods:
        monkeypatch.setattr(study, "DEFAULT_PROFILE", path)
        args = ["--output", str(output), "--run"]
    else:
        args = ["--profile", str(path), "--output", str(output), "--run"]
    per_suite = len(config["methods"])
    total = per_suite * len(config["suites"])
    assert study.main(args) == 1  # ERROR outcomes do not prevent later suites.
    assert len(calls) == total
    for suite in config["suites"]:
        directory = output / suite["key"]
        checkpoint = json.loads((directory / "benchmark_config.json").read_text())
        assert checkpoint["repetitions"] == 1
        assert checkpoint["timeout_seconds"] is checkpoint["run_timeout_seconds"] is None
        assert checkpoint["max_iterations"] == 50
        assert checkpoint["state"] == "finished" and checkpoint["completed_jobs"] == per_suite
        assert checkpoint["projection_override"] == config["methods"]
        with (directory / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert {(r["projection"], r["repetition"]) for r in rows} == {
            (method, "1") for method in config["methods"]}
        assert {r["status"] for r in rows} == {"ERROR"}
    for artifact in output.rglob("*"):
        if artifact.is_file() and artifact.suffix in {".json", ".csv", ".md"}:
            assert "seed" not in artifact.read_text().lower(), artifact
    saved = json.loads((output / "study_provenance.json").read_text())
    assert saved["profile"] == config
    assert len(saved["profile_sha256"]) == 64
    assert (output / "full_report.md").is_file()
    assert config.get("title", "Paper study") in (output / "full_report.md").read_text()
    report = (output / "full_report.md").read_text()
    assert "whole-run timeout: disabled" in report
    assert "Storm-call timeout: disabled" in report
    assert "max iterations: 50" in report
    assert study.main(args + ["--resume"]) == 1
    assert len(calls) == total
    with pytest.raises(SystemExit) as exc:
        study.main(args)
    assert exc.value.code == 2  # Never overwrite an existing study.
    assert len(calls) == total


@pytest.mark.parametrize("change", ["settings", "storm_timeout", "refinements", "repetitions", "methods", "suites", "input", "code", "binary", "checkpoint", "method_order"])
def test_resume_rejects_incompatible_data_before_execution(tmp_path, monkeypatch, change):
    path, config = tiny_study(tmp_path)
    output = tmp_path / "results"
    monkeypatch.setattr(study.shutil, "which", lambda name: sys.executable)
    suites = study.plan(ROOT, output, config, "storm", False)
    study.prepare(ROOT, output, config, suites, False)
    if change == "settings":
        config["run_timeout_seconds"] = 300
    elif change == "storm_timeout":
        config["storm_timeout_seconds"] = 200
    elif change == "refinements":
        config["max_iterations"] = 100
    elif change == "repetitions":
        config["repetitions"] = 2
    elif change == "methods":
        config["methods"] = ["gauge"]
    elif change == "suites":
        config["suites"] = config["suites"][:1]
    elif change == "method_order":
        config["methods"] = list(reversed(config["methods"]))
    elif change == "input":
        (tmp_path / "second.pm").write_text("changed model")
    elif change in ("code", "binary"):
        provenance_path = output / "study_provenance.json"
        old = json.loads(provenance_path.read_text())
        old["code_sha256" if change == "code" else "storm_sha256"] = "old hash"
        provenance_path.write_text(json.dumps(old))
    else:
        directory = output / "second"
        directory.mkdir()
        (directory / "summary.csv").write_text("unverified old rows")
    suites = study.plan(ROOT, output, config, "storm", True)
    with pytest.raises(ValueError, match="changed study|missing checkpoint"):
        study.prepare(ROOT, output, config, suites, True)


def test_resume_rejects_legacy_directory_and_requires_run(tmp_path):
    config = profile()
    suites = study.plan(ROOT, tmp_path, config, "storm", True)
    with pytest.raises(ValueError, match="no matching study provenance"):
        study.prepare(ROOT, tmp_path, config, suites, True)
    with pytest.raises(SystemExit) as exc:
        study.main(["--resume"])
    assert exc.value.code == 2


def test_configuration_digest_is_stable_and_sensitive():
    from mdp_safety.experiments.provenance import configuration_digest
    assert configuration_digest({"b": 2, "a": 1}) == configuration_digest({"a": 1, "b": 2})
    assert configuration_digest({"a": 1}) != configuration_digest({"a": 2})


def test_documentation_does_not_describe_internal_ordering():
    paths = [ROOT / "README.md", ROOT / "experiments/README.md", *ROOT.glob("docs/*.md")]
    for path in paths:
        text = path.read_text().lower()
        assert "job-order seed" not in text, path
        assert "shuffle" not in text, path
        assert "ordering seed" not in text, path


def test_old_study_profile_requires_explicit_upgrade(tmp_path):
    config = profile()
    config["schema_version"] = 1
    with pytest.raises(ValueError, match="unsupported study profile"):
        study.plan(ROOT, tmp_path / "absent", config, "storm", False)
    assert not (tmp_path / "absent").exists()


def test_repetitions_follow_methods_without_reordering_instances():
    from dataclasses import replace
    from mdp_safety.experiments.manifest import build_jobs, load_manifest
    cases = load_manifest(ROOT / "experiments/manifests/domains.json")[:2]
    cases = [replace(case, projections=("gauge", "l2")) for case in cases]
    jobs = build_jobs(iter(cases), 2)
    assert [job.key for job in jobs] == [
        (case.name, method, repetition) for case in cases
        for method in ("gauge", "l2") for repetition in (1, 2)]
