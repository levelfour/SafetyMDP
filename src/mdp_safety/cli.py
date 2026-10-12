"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from .defaults import DEFAULT_TIMEOUT_SECONDS
from .timeouts import parse_timeout
from .strategies.registry import projection_help, projection_names
from .specification import load_specification
from .reporting import write_result
from .verifier import SafetyVerifier, VerificationOptions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdp-safety",
        description=(
            "Verify containment of a finite MDP's terminal-reachability "
            "Pareto/achievable region using Storm OVI."
        ),
    )
    parser.add_argument("model", type=Path, help="finite MDP in PRISM language")
    parser.add_argument("specification", type=Path, help="JSON terminal/safe-region specification")
    parser.add_argument(
        "--projection",
        choices=projection_names(),
        default="l2",
        help=projection_help() + "; default: l2",
    )
    parser.add_argument("--storm", default="storm", help="Storm CLI for PRISM model export (queries use Stormpy; default: storm)")
    parser.add_argument("--constants", help="PRISM constant assignments, e.g. N=20,p=0.1")
    parser.add_argument("--precision", type=float, default=1e-7, help="initial absolute OVI precision")
    parser.add_argument("--sandwich-precision", type=float, default=1e-6,
                        help="Pareto indicator tolerance (default: 1e-6): absolute for sandwich-lk, "
                             "relative for sandwich-lk-gauge; "
                             "reaching it alone returns UNKNOWN, not SAFE")
    parser.add_argument(
        "--min-precision", type=float, default=1e-12, help="smallest absolute OVI precision"
    )
    parser.add_argument(
        "--geometry-tolerance", type=float, default=1e-8, help="convex-geometry tolerance"
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--timeout",
        type=parse_timeout,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="timeout per Storm CLI/API operation in seconds, or 'none' to disable (default: 300)",
    )
    parser.add_argument("--output-dir", type=Path, help="write result.json and CSV artifacts here")
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="keep generated DRN/results under OUTPUT_DIR/work (requires --output-dir)",
    )
    parser.add_argument(
        "--omit-scheduler-actions",
        action="store_true",
        help="omit full scheduler maps from result.json (useful for benchmarks)",
    )
    parser.add_argument("--debug", action="store_true", help="show a traceback on errors")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        specification = load_specification(args.specification, args.geometry_tolerance / 100.0)
        terminal_labels = specification.terminal_labels
        terminal_sets = specification.terminal_sets
        terminalize = specification.terminalize
        safe_region = specification.safe_region
        options = VerificationOptions(
            projection=args.projection,
            storm_precision=args.precision,
            sandwich_precision=args.sandwich_precision,
            minimum_storm_precision=args.min_precision,
            geometry_tolerance=args.geometry_tolerance,
            max_iterations=args.max_iterations,
            storm_timeout_seconds=args.timeout,
            keep_working_files=args.keep_work,
            record_scheduler_actions=not args.omit_scheduler_actions,
        )
        if args.keep_work and args.output_dir is None:
            raise ValueError("--keep-work requires --output-dir")
        verifier = SafetyVerifier(options, args.storm)
        work = args.output_dir / "work" if args.keep_work else None
        result = verifier.verify(
            args.model,
            terminal_labels,
            safe_region,
            constants=args.constants,
            working_directory=work,
            terminal_sets=terminal_sets,
            terminalize=terminalize,
        )
        if args.output_dir is not None:
            artifacts = write_result(result, args.output_dir.resolve())
        else:
            artifacts = {}
        output = result.to_dict()
        if artifacts:
            output["artifacts"] = {key: str(path) for key, path in artifacts.items()}
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 2 if result.status == "UNKNOWN" else 0
    except Exception as error:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"mdp-safety: error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
