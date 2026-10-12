"""Isolated verifier processes, whole-run timeouts and per-run metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from ..timeouts import validate_timeout
from .models import BenchmarkJob, BenchmarkSettings, CommandExecutor, DECISIVE_VERDICTS


def _stop_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Reap the verifier and its Storm children after cancellation or timeout."""
    def send(sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        return process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        send(getattr(signal, "SIGKILL", 9))
        return process.communicate()


def execute_command(
    command: list[str], run_timeout: float | None
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Run with an optional deadline; reap the process group on timeout or Ctrl-C."""

    validate_timeout(run_timeout, "run timeout")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=run_timeout)
        return subprocess.CompletedProcess(
            command, process.returncode, stdout, stderr
        ), False
    except subprocess.TimeoutExpired:
        stdout, stderr = _stop_process_group(process)
        return subprocess.CompletedProcess(command, 124, stdout, stderr), True
    except BaseException:
        # No-deadline runs still need reliable Ctrl-C cleanup and checkpointing.
        _stop_process_group(process)
        raise


def _command(job: BenchmarkJob, settings: BenchmarkSettings, run_directory: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mdp_safety",
        str(job.instance.model),
        str(job.instance.specification),
        "--projection",
        job.projection,
        "--precision",
        str(settings.precision),
        "--min-precision",
        str(settings.minimum_precision),
        "--sandwich-precision",
        str(settings.sandwich_precision),
        "--geometry-tolerance",
        str(settings.geometry_tolerance),
        "--max-iterations",
        str(settings.max_iterations),
        "--timeout",
        "none" if settings.storm_timeout_seconds is None else str(settings.storm_timeout_seconds),
        "--storm",
        settings.storm,
        "--output-dir",
        str(run_directory),
    ]
    if not settings.record_scheduler_actions:
        command.append("--omit-scheduler-actions")
    if job.instance.constants:
        command.extend(("--constants", job.instance.constants))
    return command


def _run_directory(job: BenchmarkJob, settings: BenchmarkSettings) -> Path:
    directory = (
        settings.output_directory.resolve()
        / job.instance.name
        / job.projection
    )
    if settings.repetitions > 1:
        directory /= f"repeat-{job.repetition:03d}"
    return directory


def _base_row(job: BenchmarkJob, execution_order: int) -> dict[str, Any]:
    instance = job.instance
    return {
        "suite": instance.suite,
        "family": instance.family,
        "name": instance.name,
        "model": str(instance.model),
        "specification": str(instance.specification),
        "expected_status": instance.expected_status or "",
        "label_matches": "",
        "projection": job.projection,
        "repetition": job.repetition,
        "execution_order": execution_order,
    }


def run_job(
    job: BenchmarkJob,
    settings: BenchmarkSettings,
    execution_order: int,
    executor: CommandExecutor = execute_command,
) -> dict[str, Any]:
    run_directory = _run_directory(job, settings)
    command = _command(job, settings, run_directory)
    started_timestamp = time.time()
    started = time.perf_counter()
    completed, timed_out = executor(command, settings.run_timeout_seconds)
    wall_seconds = time.perf_counter() - started
    result_path = run_directory / "result.json"
    fresh_result = (
        result_path.exists()
        and result_path.stat().st_mtime >= started_timestamp - 1.0
    )
    row = _base_row(job, execution_order)
    if fresh_result and not timed_out:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        status = str(result["status"])
        expected = job.instance.expected_status
        row.update(
            {
                "label_matches": (
                    status == expected
                    if expected and status in DECISIVE_VERDICTS
                    else ""
                ),
                "status": status,
                "elapsed_seconds": result["elapsed_seconds"],
                "wall_seconds": wall_seconds,
                "iterations": result["iterations"],
                "support_queries": len(result["queries"]),
                "oracle_calls": result["oracle_call_count"],
                "weight_selection_calls": result["weight_selection_call_count"],
                "weight_selection_seconds": result["weight_selection_seconds"],
                "weight_selection_cpu_seconds": result[
                    "weight_selection_cpu_seconds"
                ],
                "mean_weight_selection_ms": 1000.0
                * result["weight_selection_seconds"]
                / max(1, result["weight_selection_call_count"]),
                "outer_vertex_enumeration_calls": result.get(
                    "outer_vertex_enumeration_call_count", ""
                ),
                "outer_vertex_enumeration_seconds": result.get(
                    "outer_vertex_enumeration_seconds", ""
                ),
                "maximum_outer_vertices": result.get(
                    "maximum_outer_vertex_count", ""
                ),
                "storm_calls": result["storm_call_count"],
                "states": result["state_count"],
                "choices": result["original_choice_count"],
                "storm_version": result["storm_version"],
                "exit_code": completed.returncode,
                "error": "",
            }
        )
    else:
        row.update(
            {
                "status": "TIMEOUT" if timed_out else "ERROR",
                "elapsed_seconds": "",
                "wall_seconds": wall_seconds,
                "iterations": "",
                "support_queries": "",
                "oracle_calls": "",
                "weight_selection_calls": "",
                "weight_selection_seconds": "",
                "weight_selection_cpu_seconds": "",
                "mean_weight_selection_ms": "",
                "outer_vertex_enumeration_calls": "",
                "outer_vertex_enumeration_seconds": "",
                "maximum_outer_vertices": "",
                "storm_calls": "",
                "states": "",
                "choices": "",
                "storm_version": "",
                "exit_code": completed.returncode,
                "error": (
                    f"run exceeded {settings.run_timeout_seconds:g} seconds"
                    if timed_out
                    else (completed.stderr or completed.stdout).strip()
                ),
            }
        )
    return row
