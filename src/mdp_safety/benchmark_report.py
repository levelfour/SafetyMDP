"""Render benchmark CSV artifacts as publication-oriented Markdown tables."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


from .strategies.registry import METHOD_LABELS as _CURRENT_METHOD_LABELS
from .timeouts import format_timeout

METHOD_LABELS = dict(_CURRENT_METHOD_LABELS)


@dataclass(frozen=True)
class SuiteArtifacts:
    label: str
    directory: Path
    config: Mapping[str, object]
    aggregate: tuple[Mapping[str, str], ...]
    by_expected_status: tuple[Mapping[str, str], ...]
    by_family: tuple[Mapping[str, str], ...]

    @classmethod
    def load(cls, label: str, directory: Path) -> "SuiteArtifacts":
        path = directory.resolve()
        return cls(
            label=label,
            directory=path,
            config=json.loads((path / "benchmark_config.json").read_text(encoding="utf-8")),
            aggregate=tuple(_read_csv(path / "aggregate.csv")),
            by_expected_status=tuple(_read_csv(path / "aggregate_by_expected_status.csv")),
            by_family=tuple(_read_csv(path / "aggregate_by_family.csv")),
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _method(value: str) -> str:
    return METHOD_LABELS.get(value, value)


def _integer(value: str) -> str:
    return str(int(float(value))) if value not in ("", None) else "--"


def _decimal(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}" if value not in ("", None) else "--"


def _method_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    output = [
        "| Method | Completed | SAFE | UNSAFE | UNKNOWN | Timeout | Oracle calls (total / mean completed) | Weight time (s / ms per call) | Wall time (s total / mean) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in sorted(rows, key=lambda item: item["projection"]):
        output.append(
            "| "
            + " | ".join(
                (
                    _method(row["projection"]),
                    f"{_integer(row['completed'])}/{_integer(row['runs'])}",
                    _integer(row["safe"]),
                    _integer(row["unsafe"]),
                    _integer(row["unknown"]),
                    _integer(row["timeouts"]),
                    f"{_integer(row['total_oracle_calls'])} / {_decimal(row['mean_oracle_calls'], 2)}",
                    f"{_decimal(row['total_weight_selection_seconds'])} / {_decimal(row['mean_weight_selection_ms_per_call'])}",
                    f"{_decimal(row['total_wall_seconds'])} / {_decimal(row['mean_wall_seconds'])}",
                )
            )
            + " |"
        )
    return output


def _family_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    output = [
        "| Family | Method | Completed | Timeout | Oracle calls | Weight time (s) | Wall time (s) |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for row in sorted(rows, key=lambda item: (item["family"], item["projection"])):
        output.append(
            "| "
            + " | ".join(
                (
                    row["family"],
                    _method(row["projection"]),
                    f"{_integer(row['completed'])}/{_integer(row['runs'])}",
                    _integer(row["timeouts"]),
                    _integer(row["total_oracle_calls"]),
                    _decimal(row["total_weight_selection_seconds"]),
                    _decimal(row["total_wall_seconds"]),
                )
            )
            + " |"
        )
    return output


def render_report(suites: Iterable[SuiteArtifacts], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        (
            "`Completed` means that a verifier result artifact was returned and may "
            "therefore include `UNKNOWN`. Oracle-call and weight-selection totals use "
            "completed runs only. Total wall time includes the full cutoff charged to "
            "timed-out runs."
        ),
        "",
    ]
    for suite in suites:
        config = suite.config
        lines.extend(
            (
                f"## {suite.label}",
                "",
                (
                    f"Manifest: `{config['manifest']}`. "
                    f"Storm-call timeout: {format_timeout(config['timeout_seconds'])}; "
                    f"whole-run timeout: {format_timeout(config['run_timeout_seconds'])}; "
                    f"precision: {config['precision']}; minimum precision: "
                    f"{config['minimum_precision']}; max iterations: "
                    f"{config['max_iterations']}; repetitions: {config['repetitions']}."
                ),
                "",
                *_method_table(suite.aggregate),
                "",
            )
        )
        statuses = sorted(
            {row["expected_status"] for row in suite.by_expected_status if row["expected_status"]}
        )
        for status in statuses:
            selected = [row for row in suite.by_expected_status if row["expected_status"] == status]
            lines.extend((f"### Expected {status}", "", *_method_table(selected), ""))
        lines.extend(("### By model family", "", *_family_table(suite.by_family), ""))
    return "\n".join(lines).rstrip() + "\n"


def _suite(value: str) -> tuple[str, Path]:
    try:
        label, directory = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("suite must be LABEL=RESULT_DIRECTORY") from error
    if not label or not directory:
        raise argparse.ArgumentTypeError("suite must be LABEL=RESULT_DIRECTORY")
    return label, Path(directory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdp-safety-report")
    parser.add_argument("--suite", action="append", type=_suite, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Finite-MDP safety benchmark results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suites = [SuiteArtifacts.load(label, path) for label, path in args.suite]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(suites, args.title), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
