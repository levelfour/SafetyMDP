"""Atomic benchmark artifacts and resume-configuration compatibility."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

from .aggregation import aggregate_rows
from .manifest import JOB_ORDER, manifest_digest, load_manifest
from .models import BenchmarkSettings, SUMMARY_FIELDS
from ..storm_api import SCHEDULER_BACKEND, stormpy_version

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _configuration(
    settings: BenchmarkSettings,
    *,
    state: str,
    started_at: str,
    benchmark_wall_seconds: float,
    jobs: int,
    completed_jobs: int,
) -> dict[str, Any]:
    manifest = settings.manifest.resolve()
    methods = (set(settings.projection_override) if settings.projection_override is not None else
               {method for instance in load_manifest(manifest) for method in instance.projections})
    lk_versions = {
        "sandwich-lk": "lk-2023-max-inner-facet-epsilon-v1",
        "sandwich-lk-gauge": "lk-2023-max-inner-facet-gauge-origin-v1",
    }
    return {
        "schema_version": 5,
        "state": state,
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat() if state == "finished" else None,
        "benchmark_wall_seconds": benchmark_wall_seconds,
        "manifest": str(manifest),
        "manifest_sha256": manifest_digest(manifest),
        "output_directory": str(settings.output_directory.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "storm": settings.storm,
        "oracle_backend": SCHEDULER_BACKEND,
        "stormpy_version": stormpy_version(),
        "precision": settings.precision,
        "minimum_precision": settings.minimum_precision,
        "sandwich_precision": settings.sandwich_precision,
        "strategy_catalog_revision": "lk-only-v1",
        "sandwich_lk_implementations": {name: version for name, version in lk_versions.items()
                                       if name in methods} or None,
        "projection_override": list(settings.projection_override) if settings.projection_override is not None else None,
        "geometry_tolerance": settings.geometry_tolerance,
        "max_iterations": settings.max_iterations,
        "timeout_seconds": settings.storm_timeout_seconds,
        "run_timeout_seconds": settings.run_timeout_seconds,
        "repetitions": settings.repetitions,
        "job_order": JOB_ORDER,
        "scheduler_actions_recorded": settings.record_scheduler_actions,
        "jobs": jobs,
        "completed_jobs": completed_jobs,
    }


def _validate_resume(settings: BenchmarkSettings) -> None:
    config_path = settings.output_directory.resolve() / "benchmark_config.json"
    if not config_path.exists():
        return
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    expected = _configuration(
        settings,
        state="running",
        started_at=str(existing.get("started_at", "")),
        benchmark_wall_seconds=0.0,
        jobs=int(existing.get("jobs", 0)),
        completed_jobs=0,
    )
    comparable = (
        "schema_version",
        "manifest_sha256",
        "storm",
        "oracle_backend",
        "stormpy_version",
        "precision",
        "minimum_precision",
        "sandwich_precision",
        "strategy_catalog_revision",
        "sandwich_lk_implementations",
        "projection_override",
        "geometry_tolerance",
        "max_iterations",
        "timeout_seconds",
        "run_timeout_seconds",
        "repetitions",
        "job_order",
        "scheduler_actions_recorded",
    )
    changed = [field for field in comparable if existing.get(field) != expected.get(field)]
    if changed:
        raise ValueError(
            "cannot resume with changed settings: " + ", ".join(changed)
        )


def write_checkpoint(
    settings: BenchmarkSettings,
    rows: Sequence[Mapping[str, Any]],
    *,
    state: str,
    started_at: str,
    benchmark_wall_seconds: float,
    jobs: int,
) -> None:
    output = settings.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: int(row["execution_order"]))
    _write_csv(output / "summary.csv", ordered, SUMMARY_FIELDS)
    aggregate_specs = {
        "aggregate.csv": ("projection",),
        "aggregate_by_expected_status.csv": ("expected_status", "projection"),
        "aggregate_by_family.csv": ("family", "projection"),
    }
    for filename, group_fields in aggregate_specs.items():
        aggregates = aggregate_rows(ordered, group_fields)
        fields = list(aggregates[0]) if aggregates else list(group_fields)
        _write_csv(output / filename, aggregates, fields)
    config = _configuration(
        settings,
        state=state,
        started_at=started_at,
        benchmark_wall_seconds=benchmark_wall_seconds,
        jobs=jobs,
        completed_jobs=len(rows),
    )
    temporary = output / ".benchmark_config.json.tmp"
    temporary.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "benchmark_config.json")
