"""Public benchmark API re-exported from the experiments package.

Execution, manifests, checkpoints and aggregation have separate implementations.
Callers can inject an executor for isolated orchestration tests.
"""

import platform  # Shared module object for downstream test injection.

from .experiments import (
    BenchmarkInstance, BenchmarkJob, BenchmarkSettings, CommandExecutor,
    DECISIVE_VERDICTS, DEFAULT_PROJECTIONS, FINAL_FAILURE_STATUSES, SUMMARY_FIELDS,
    aggregate_rows, build_jobs, execute_command, load_manifest, manifest_digest,
    run_benchmark, run_job, write_checkpoint,
)
from .experiments.aggregation import _number, _percentile
from .experiments.checkpoint import _configuration, _read_summary, _validate_resume, _write_csv
from .experiments.execution import _base_row, _command, _run_directory
from .experiments.manifest import _load_manifest
from .experiments.models import _infer_expected_status, _infer_family, _resolve

__all__ = [
    "BenchmarkInstance", "BenchmarkJob", "BenchmarkSettings", "CommandExecutor",
    "DECISIVE_VERDICTS", "DEFAULT_PROJECTIONS", "FINAL_FAILURE_STATUSES",
    "SUMMARY_FIELDS", "aggregate_rows", "build_jobs", "execute_command",
    "load_manifest", "manifest_digest", "run_benchmark", "run_job", "write_checkpoint",
]
