import csv
import json
from pathlib import Path

from mdp_safety.benchmark_report import SuiteArtifacts, render_report


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_report_keeps_suites_in_separate_tables(tmp_path: Path) -> None:
    row = {
        "projection": "linf",
        "family": "gridworld",
        "expected_status": "SAFE",
        "runs": 2,
        "completed": 2,
        "safe": 2,
        "unsafe": 0,
        "unknown": 0,
        "timeouts": 0,
        "total_oracle_calls": 4,
        "mean_oracle_calls": 2,
        "total_weight_selection_seconds": 0.01,
        "mean_weight_selection_ms_per_call": 2.5,
        "total_wall_seconds": 1.5,
        "mean_wall_seconds": 0.75,
    }
    suites = []
    for directory_name, label in (("first", "First suite"), ("second", "Second suite")):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "benchmark_config.json").write_text(
            json.dumps(
                {
                    "manifest": "/manifest.json",
                    "timeout_seconds": 600,
                    "run_timeout_seconds": 600,
                    "precision": 1e-7,
                    "minimum_precision": 1e-11,
                    "max_iterations": 100,
                    "repetitions": 1,
                }
            ),
            encoding="utf-8",
        )
        _write_csv(directory / "aggregate.csv", [row])
        _write_csv(directory / "aggregate_by_expected_status.csv", [row])
        _write_csv(directory / "aggregate_by_family.csv", [row])
        suites.append(SuiteArtifacts.load(label, directory))
    report = render_report(suites, "Results")
    assert report.count("| Method | Completed | SAFE |") == 4
    assert "## First suite" in report
    assert "## Second suite" in report
    assert report.count("whole-run timeout: 600 s") == 2
