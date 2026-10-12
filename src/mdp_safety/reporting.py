"""Stable JSON/CSV artifacts for benchmark automation."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .verification.results import SafetyResult


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_result(result: SafetyResult, output_directory: Path) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    result_json = output_directory / "result.json"
    result_json.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    iterations_csv = output_directory / "iterations.csv"
    query_rows = [asdict(query) for query in result.queries]
    fields = list(query_rows[0]) if query_rows else [
        "iteration",
        "attempt",
        "projection",
        "outcome",
    ]
    with iterations_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in query_rows:
            writer.writerow(
                {
                    key: _json_cell(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )

    selections_csv = output_directory / "weight_selections.csv"
    selection_rows = [asdict(selection) for selection in result.selections]
    selection_fields = list(selection_rows[0]) if selection_rows else [
        "iteration",
        "projection",
        "elapsed_seconds",
        "cpu_seconds",
        "metric",
        "score",
    ]
    with selections_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=selection_fields)
        writer.writeheader()
        for row in selection_rows:
            writer.writerow(
                {
                    key: _json_cell(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )

    vertices_csv = output_directory / "outer_vertices.csv"
    _write_points(vertices_csv, result.outer_vertices)
    pareto_csv = output_directory / "pareto_samples.csv"
    _write_points(pareto_csv, result.pareto_samples)
    inner_csv = output_directory / "inner_approximation_points.csv"
    _write_points(inner_csv, result.inner_approximation_points)
    return {
        "result": result_json,
        "iterations": iterations_csv,
        "weight_selections": selections_csv,
        "outer_vertices": vertices_csv,
        "pareto_samples": pareto_csv,
        "inner_approximation_points": inner_csv,
    }


def _write_points(path: Path, points: list[list[float]]) -> None:
    dimension = len(points[0]) if points else 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"p{index + 1}" for index in range(dimension)])
        writer.writerows(points)
