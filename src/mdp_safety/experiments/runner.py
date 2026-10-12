"""Resumable orchestration of benchmark jobs."""

from __future__ import annotations

from datetime import datetime
from dataclasses import replace
import json
import time
from typing import Any

from .checkpoint import _read_summary, _validate_resume, write_checkpoint
from .execution import execute_command, run_job
from .manifest import build_jobs, load_manifest
from .models import BenchmarkSettings, CommandExecutor, FINAL_FAILURE_STATUSES

def run_benchmark(
    settings: BenchmarkSettings,
    executor: CommandExecutor = execute_command,
) -> int:
    """Run a suite, checkpointing after every job and optionally resuming it."""

    settings.validate()
    instances = load_manifest(settings.manifest)
    if settings.projection_override is not None:
        instances = [replace(instance, projections=settings.projection_override) for instance in instances]
    jobs = build_jobs(instances, settings.repetitions)
    output = settings.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat()
    previous_wall_seconds = 0.0
    rows: list[dict[str, Any]] = []
    if settings.resume:
        _validate_resume(settings)
        rows.extend(_read_summary(output / "summary.csv"))
        config_path = output / "benchmark_config.json"
        if config_path.exists():
            previous_config = json.loads(config_path.read_text(encoding="utf-8"))
            started_at = str(previous_config["started_at"])
            previous_wall_seconds = float(
                previous_config.get("benchmark_wall_seconds", 0.0)
            )
    completed_keys = {
        (str(row["name"]), str(row["projection"]), int(row["repetition"]))
        for row in rows
    }
    run_started = time.perf_counter()
    try:
        for execution_order, job in enumerate(jobs, 1):
            if job.key in completed_keys:
                continue
            progress = f"[{execution_order}/{len(jobs)}]"
            identity = (f"{job.instance.name} | method={job.projection} | "
                        f"repetition={job.repetition}/{settings.repetitions}")
            print(f"{progress} START {identity}", flush=True)
            row = run_job(job, settings, execution_order, executor)
            rows.append(row)
            print(
                f"{progress} DONE {identity} | {row['status']} "
                f"({float(row['wall_seconds']):.3f}s)",
                flush=True,
            )
            write_checkpoint(
                settings,
                rows,
                state="running",
                started_at=started_at,
                benchmark_wall_seconds=(
                    previous_wall_seconds + time.perf_counter() - run_started
                ),
                jobs=len(jobs),
            )
    except BaseException:
        write_checkpoint(
            settings,
            rows,
            state="interrupted",
            started_at=started_at,
            benchmark_wall_seconds=(
                previous_wall_seconds + time.perf_counter() - run_started
            ),
            jobs=len(jobs),
        )
        raise
    write_checkpoint(
        settings,
        rows,
        state="finished",
        started_at=started_at,
        benchmark_wall_seconds=(
            previous_wall_seconds + time.perf_counter() - run_started
        ),
        jobs=len(jobs),
    )
    return int(any(str(row.get("status")) in FINAL_FAILURE_STATUSES for row in rows))
