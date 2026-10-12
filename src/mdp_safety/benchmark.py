"""Command-line interface for reproducible, resumable benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import benchmarking
from .defaults import DEFAULT_TIMEOUT_SECONDS
from .timeouts import parse_timeout
from .strategies import projection_names


# Kept as a module attribute so tests and downstream callers can inject a local
# executor without coupling to the runner implementation.
_execute = benchmarking.execute_command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdp-safety-bench")
    parser.add_argument("manifest", type=Path, help="JSON array of benchmark instances")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--projections", nargs="+", choices=projection_names(),
        help="replace every instance's method list (e.g. --projections sandwich-lk); "
             "does not edit the manifest",
    )
    parser.add_argument("--precision", type=float, default=1e-7)
    parser.add_argument("--min-precision", type=float, default=1e-12)
    parser.add_argument("--sandwich-precision", type=float, default=1e-6)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--timeout",
        type=parse_timeout,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="timeout per Storm call in seconds, or 'none' to disable (default: 300)",
    )
    parser.add_argument(
        "--run-timeout",
        type=parse_timeout,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="wall-clock timeout for one complete run, or 'none' to disable (default: 300)",
    )
    parser.add_argument("--storm", default="storm")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="number of measured repetitions per instance/method (default: 1)",
    )
    parser.add_argument(
        "--keep-scheduler-actions",
        action="store_true",
        help="retain scheduler action maps in per-run JSON",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a checkpoint whose manifest and settings match exactly",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = benchmarking.BenchmarkSettings(
        manifest=args.manifest.resolve(),
        output_directory=args.output_dir.resolve(),
        precision=args.precision,
        minimum_precision=args.min_precision,
        sandwich_precision=args.sandwich_precision,
        geometry_tolerance=args.geometry_tolerance,
        max_iterations=args.max_iterations,
        storm_timeout_seconds=args.timeout,
        run_timeout_seconds=args.run_timeout,
        storm=args.storm,
        repetitions=args.repetitions,
        record_scheduler_actions=args.keep_scheduler_actions,
        resume=args.resume,
        projection_override=tuple(args.projections) if args.projections is not None else None,
    )
    try:
        return benchmarking.run_benchmark(settings, executor=_execute)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
