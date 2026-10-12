import csv
import json
from pathlib import Path
import subprocess
import pytest

from mdp_safety import benchmark
from mdp_safety import benchmarking


def test_resume_rejects_previous_strategy_catalog(tmp_path):
    from mdp_safety.experiments.checkpoint import _configuration, _validate_resume
    from mdp_safety.experiments.models import BenchmarkSettings
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    settings = BenchmarkSettings(manifest, tmp_path)
    config = _configuration(settings, state="finished", started_at="test", benchmark_wall_seconds=0,
                            jobs=0, completed_jobs=0)
    config.pop("strategy_catalog_revision")
    (tmp_path / "benchmark_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="strategy_catalog_revision"):
        _validate_resume(settings)


def test_repeated_benchmark_is_ordered_and_aggregated(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "tiny",
                    "model": "model.pm",
                    "specification": "spec.json",
                    "projections": ["l2", "gauge"],
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_execute(command, _run_timeout):
        assert command[command.index("--sandwich-precision") + 1] == "2e-05"
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        projection = command[command.index("--projection") + 1]
        result = {
            "status": "SAFE",
            "elapsed_seconds": 0.01,
            "iterations": 2,
            "queries": [{}, {}],
            "oracle_call_count": 2,
            "weight_selection_call_count": 2,
            "weight_selection_seconds": 0.002,
            "weight_selection_cpu_seconds": 0.001,
            "outer_vertex_enumeration_call_count": 3,
            "outer_vertex_enumeration_seconds": 0.003,
            "maximum_outer_vertex_count": 8,
            "storm_call_count": 6,
            "state_count": 3,
            "original_choice_count": 4,
            "storm_version": "test",
            "projection": projection,
        }
        (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", ""), False

    monkeypatch.setattr(benchmark, "_execute", fake_execute)
    monkeypatch.setattr(benchmarking.platform, "platform", lambda: "test-platform")
    output = tmp_path / "results"
    assert (
        benchmark.main(
            [
                str(manifest),
                "--output-dir",
                str(output),
                "--repetitions",
                "2",
                "--sandwich-precision",
                "2e-5",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader((output / "summary.csv").open(encoding="utf-8")))
    assert len(rows) == 4
    assert {row["repetition"] for row in rows} == {"1", "2"}
    assert {row["projection"] for row in rows} == {"l2", "gauge"}
    assert [int(row["execution_order"]) for row in rows] == [1, 2, 3, 4]
    assert [(row["projection"], row["repetition"]) for row in rows] == [
        ("l2", "1"), ("l2", "2"), ("gauge", "1"), ("gauge", "2")]
    assert {row["suite"] for row in rows} == {tmp_path.name}
    assert {row["family"] for row in rows} == {"tiny"}

    aggregate = {
        row["projection"]: row
        for row in csv.DictReader((output / "aggregate.csv").open(encoding="utf-8"))
    }
    assert aggregate["l2"]["total_oracle_calls"] == "4"
    assert aggregate["l2"]["mean_suite_oracle_calls"] == "2.0"
    assert aggregate["l2"]["total_outer_vertex_enumeration_seconds"] == "0.006"
    assert aggregate["l2"]["maximum_outer_vertices"] == "8"
    config = json.loads((output / "benchmark_config.json").read_text(encoding="utf-8"))
    assert config["jobs"] == 4
    assert config["completed_jobs"] == 4
    assert config["schema_version"] == 5
    assert "sandwich_implementation" not in config
    assert config["state"] == "finished"
    assert "shuffle_seed" not in config
    assert config["job_order"] == "manifest-method-repetition-v1"
    assert "seed" not in json.dumps(config).lower()
    assert config["scheduler_actions_recorded"] is False
    assert config["sandwich_precision"] == 2e-5
    assert config["strategy_catalog_revision"] == "lk-only-v1"


def test_benchmark_records_whole_run_timeout(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "slow",
                    "model": "model.pm",
                    "specification": "spec.json",
                    "projections": ["l2"],
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_timeout(command, _run_timeout):
        return subprocess.CompletedProcess(command, 124, "", ""), True

    monkeypatch.setattr(benchmark, "_execute", fake_timeout)
    monkeypatch.setattr(benchmarking.platform, "platform", lambda: "test-platform")
    output = tmp_path / "results"
    assert (
        benchmark.main(
            [
                str(manifest),
                "--output-dir",
                str(output),
                "--run-timeout",
                "1",
            ]
        )
        == 1
    )
    row = next(csv.DictReader((output / "summary.csv").open(encoding="utf-8")))
    assert row["status"] == "TIMEOUT"
    aggregate = next(
        csv.DictReader((output / "aggregate.csv").open(encoding="utf-8"))
    )
    assert aggregate["timeouts"] == "1"
    assert aggregate["errors"] == "0"
    assert float(aggregate["total_wall_seconds"]) > 0.0
    assert aggregate["completed_total_wall_seconds"] == "0"


def test_resume_skips_checkpointed_jobs(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "tiny-safe",
                    "model": "model.pm",
                    "specification": "spec-safe.json",
                    "family": "example",
                    "suite": "test-suite",
                    "projections": ["l2", "linf"],
                }
            ]
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_execute(command, _run_timeout):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        projection = command[command.index("--projection") + 1]
        calls.append(projection)
        result = {
            "status": "SAFE",
            "elapsed_seconds": 0.01,
            "iterations": 1,
            "queries": [{}],
            "oracle_call_count": 1,
            "weight_selection_call_count": 1,
            "weight_selection_seconds": 0.001,
            "weight_selection_cpu_seconds": 0.001,
            "storm_call_count": 3,
            "state_count": 2,
            "original_choice_count": 2,
            "storm_version": "test",
        }
        (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", ""), False

    monkeypatch.setattr(benchmark, "_execute", fake_execute)
    output = tmp_path / "results"
    arguments = [
        str(manifest),
        "--output-dir",
        str(output),
        "--timeout",
        "600",
        "--run-timeout",
        "600",
    ]
    assert benchmark.main(arguments) == 0
    assert calls == ["l2", "linf"]
    calls.clear()
    assert benchmark.main([*arguments, "--resume"]) == 0
    assert calls == []
    rows = list(csv.DictReader((output / "summary.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert {row["expected_status"] for row in rows} == {"SAFE"}
    assert {row["label_matches"] for row in rows} == {"True"}
    config = json.loads((output / "benchmark_config.json").read_text(encoding="utf-8"))
    assert config["timeout_seconds"] == 600.0
    assert config["run_timeout_seconds"] == 600.0


def test_resume_rejects_changed_timeout(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "slow-unsafe",
                    "model": "model.pm",
                    "specification": "spec-unsafe.json",
                    "projections": ["l2"],
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_timeout(command, _run_timeout):
        return subprocess.CompletedProcess(command, 124, "", ""), True

    monkeypatch.setattr(benchmark, "_execute", fake_timeout)
    output = tmp_path / "results"
    assert benchmark.main(
        [str(manifest), "--output-dir", str(output), "--run-timeout", "1"]
    ) == 1
    try:
        benchmark.main(
            [
                str(manifest),
                "--output-dir",
                str(output),
                "--run-timeout",
                "2",
                "--resume",
            ]
        )
    except SystemExit as error:
        assert "run_timeout_seconds" in str(error)
    else:
        raise AssertionError("changed timeout should reject resume")


def test_suite_manifest_includes_leaf_manifests(tmp_path: Path) -> None:
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    (leaf / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "name": "gridworld-d4-safe",
                    "model": "model.prism",
                    "specification": "spec-safe.json",
                }
            ]
        ),
        encoding="utf-8",
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"suite": "combined", "include": ["leaf/manifest.json"]}),
        encoding="utf-8",
    )
    instances = benchmarking.load_manifest(suite)
    assert len(instances) == 1
    assert instances[0].suite == "combined"
    assert instances[0].family == "gridworld"
    assert instances[0].model == (leaf / "model.prism").resolve()
    first_digest = benchmarking.manifest_digest(suite)
    (leaf / "manifest.json").write_text("[]", encoding="utf-8")
    assert benchmarking.manifest_digest(suite) != first_digest


@pytest.mark.parametrize("field,value", [
    ("schema_version", 3), ("schema_version", 4), ("job_order", "incompatible"),
])
def test_resume_rejects_incompatible_execution_order(tmp_path, field, value):
    from mdp_safety.experiments.checkpoint import _configuration, _validate_resume
    from mdp_safety.experiments.models import BenchmarkSettings
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    settings = BenchmarkSettings(manifest, tmp_path)
    config = _configuration(settings, state="finished", started_at="test",
                            benchmark_wall_seconds=0, jobs=0, completed_jobs=0)
    path = tmp_path / "benchmark_config.json"
    path.write_text(json.dumps(config))
    _validate_resume(settings)
    config[field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        _validate_resume(settings)


def test_removed_order_option_is_rejected_before_execution(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid option must not run any job")
    monkeypatch.setattr(benchmark, "_execute", forbidden)
    assert "seed" not in benchmark._parser().format_help().lower()
    with pytest.raises(SystemExit) as error:
        benchmark.main(["missing.json", "--output-dir", str(tmp_path / "absent"),
                        "--shuffle-seed", "7"])
    assert error.value.code == 2
    assert not (tmp_path / "absent").exists()


def test_start_is_flushed_before_execution_and_partial_resume_keeps_positions(tmp_path, monkeypatch):
    from dataclasses import replace
    from mdp_safety.experiments import runner
    from mdp_safety.experiments.models import BenchmarkSettings
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"name": f"gridworld-d4-{status}", "model": "model.pm",
         "specification": f"{status}.json", "projections": ["l2", "gauge"]}
        for status in ("safe", "unsafe")
    ]))
    settings = BenchmarkSettings(manifest, tmp_path / "results")
    messages, starts = [], []

    def capture(message, **kwargs):
        assert kwargs.get("flush") is True
        messages.append(message)

    def execute(command, timeout):
        message = messages[-1]
        assert "] START gridworld-d4-" in message
        method = command[command.index("--projection") + 1]
        assert f"method={method}" in message
        assert "repetition=1/1" in message
        starts.append(message)
        if len(starts) == 2:
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(command, 3, "", "mock error"), False

    monkeypatch.setattr(runner, "print", capture, raising=False)
    with pytest.raises(KeyboardInterrupt):
        runner.run_benchmark(settings, executor=execute)
    checkpoint = json.loads((settings.output_directory / "benchmark_config.json").read_text())
    assert checkpoint["state"] == "interrupted" and checkpoint["completed_jobs"] == 1
    assert len(messages) == 3
    assert messages[0].startswith("[1/4] START")
    assert messages[1].startswith("[1/4] DONE")
    assert messages[2].startswith("[2/4] START")

    assert runner.run_benchmark(replace(settings, resume=True), executor=execute) == 1
    assert [message.split("]")[0] for message in starts] == ["[1/4", "[2/4", "[2/4", "[3/4", "[4/4"]
    assert [message.split(" | method=")[0].split(" START ")[1] for message in starts] == [
        "gridworld-d4-safe", "gridworld-d4-safe", "gridworld-d4-safe",
        "gridworld-d4-unsafe", "gridworld-d4-unsafe"]
    assert messages[-1].startswith("[4/4] DONE")
    checkpoint = json.loads((settings.output_directory / "benchmark_config.json").read_text())
    assert checkpoint["state"] == "finished" and checkpoint["completed_jobs"] == 4
    count = len(messages)
    assert runner.run_benchmark(replace(settings, resume=True), executor=execute) == 1
    assert len(messages) == count
