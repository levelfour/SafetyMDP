"""Execution and analysis primitives for reproducible benchmark suites."""

from .aggregation import aggregate_rows
from .checkpoint import write_checkpoint
from .execution import execute_command, run_job
from .manifest import build_jobs, load_manifest, manifest_digest
from .models import (
    BenchmarkInstance, BenchmarkJob, BenchmarkSettings, CommandExecutor,
    DECISIVE_VERDICTS, DEFAULT_PROJECTIONS, FINAL_FAILURE_STATUSES, SUMMARY_FIELDS,
)
from .runner import run_benchmark

__all__ = [
    "BenchmarkInstance", "BenchmarkJob", "BenchmarkSettings", "CommandExecutor",
    "DECISIVE_VERDICTS", "DEFAULT_PROJECTIONS", "FINAL_FAILURE_STATUSES",
    "SUMMARY_FIELDS", "aggregate_rows", "build_jobs", "execute_command",
    "load_manifest", "manifest_digest", "run_benchmark", "run_job", "write_checkpoint",
]
