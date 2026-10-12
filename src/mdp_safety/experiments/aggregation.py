"""Pure aggregation of benchmark rows; missing measurements stay missing."""

from __future__ import annotations

from statistics import mean, median, stdev
from typing import Any, Callable, Mapping, Sequence

def _number(row: Mapping[str, Any], field: str, converter: Callable[[Any], Any]) -> Any | None:
    value = row.get(field, "")
    if value in ("", None):
        return None
    return converter(value)


def _percentile(values: Sequence[float], fraction: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str] = ("projection",),
) -> list[dict[str, Any]]:
    """Aggregate runs by method and optional suite/family/expected verdict fields."""

    keys = sorted({tuple(str(row.get(field, "")) for field in group_fields) for row in rows})
    aggregates: list[dict[str, Any]] = []
    for key in keys:
        selected = [
            row
            for row in rows
            if tuple(str(row.get(field, "")) for field in group_fields) == key
        ]
        completed = [row for row in selected if row.get("oracle_calls", "") not in ("", None)]
        calls = [_number(row, "oracle_calls", int) for row in completed]
        selection_calls = [_number(row, "weight_selection_calls", int) for row in completed]
        selection_seconds = [_number(row, "weight_selection_seconds", float) for row in completed]
        wall_seconds = [_number(row, "wall_seconds", float) for row in selected]
        completed_wall = [_number(row, "wall_seconds", float) for row in completed]
        verifier_seconds = [_number(row, "elapsed_seconds", float) for row in completed]
        enumeration_seconds = [_number(row, "outer_vertex_enumeration_seconds", float) for row in completed]
        outer_vertices = [_number(row, "maximum_outer_vertices", int) for row in completed]
        calls = [value for value in calls if value is not None]
        selection_calls = [value for value in selection_calls if value is not None]
        selection_seconds = [value for value in selection_seconds if value is not None]
        wall_seconds = [value for value in wall_seconds if value is not None]
        completed_wall = [value for value in completed_wall if value is not None]
        verifier_seconds = [value for value in verifier_seconds if value is not None]
        enumeration_seconds = [value for value in enumeration_seconds if value is not None]
        outer_vertices = [value for value in outer_vertices if value is not None]
        total_selection_calls = sum(selection_calls)
        total_selection_seconds = sum(selection_seconds)
        repetitions = len({int(row["repetition"]) for row in selected}) if selected else 0
        aggregate: dict[str, Any] = dict(zip(group_fields, key))
        aggregate.update(
            {
                "instances": len({str(row["name"]) for row in selected}),
                "repetitions": repetitions,
                "runs": len(selected),
                "completed": len(completed),
                "safe": sum(row.get("status") == "SAFE" for row in selected),
                "unsafe": sum(row.get("status") == "UNSAFE" for row in selected),
                "unknown": sum(row.get("status") == "UNKNOWN" for row in selected),
                "timeouts": sum(row.get("status") == "TIMEOUT" for row in selected),
                "errors": sum(row.get("status") == "ERROR" for row in selected),
                "label_mismatches": sum(str(row.get("label_matches", "")).lower() == "false" for row in selected),
                "total_oracle_calls": sum(calls),
                "mean_oracle_calls": mean(calls) if calls else "",
                "median_oracle_calls": median(calls) if calls else "",
                "stdev_oracle_calls": stdev(calls) if len(calls) > 1 else 0.0,
                "min_oracle_calls": min(calls) if calls else "",
                "max_oracle_calls": max(calls) if calls else "",
                "mean_suite_oracle_calls": sum(calls) / repetitions if calls and repetitions else "",
                "total_weight_selection_calls": total_selection_calls,
                "total_weight_selection_seconds": total_selection_seconds,
                "mean_weight_selection_seconds_per_instance": mean(selection_seconds) if selection_seconds else "",
                "mean_weight_selection_ms_per_call": 1000.0 * total_selection_seconds / total_selection_calls if total_selection_calls else "",
                "total_outer_vertex_enumeration_seconds": sum(enumeration_seconds),
                "mean_outer_vertex_enumeration_seconds_per_instance": mean(enumeration_seconds) if enumeration_seconds else "",
                "maximum_outer_vertices": max(outer_vertices) if outer_vertices else "",
                "total_wall_seconds": sum(wall_seconds),
                "mean_wall_seconds": mean(wall_seconds) if wall_seconds else "",
                "median_wall_seconds": median(wall_seconds) if wall_seconds else "",
                "stdev_wall_seconds": stdev(wall_seconds) if len(wall_seconds) > 1 else 0.0,
                "p25_wall_seconds": _percentile(wall_seconds, 0.25),
                "p75_wall_seconds": _percentile(wall_seconds, 0.75),
                "min_wall_seconds": min(wall_seconds) if wall_seconds else "",
                "max_wall_seconds": max(wall_seconds) if wall_seconds else "",
                "mean_suite_wall_seconds": sum(wall_seconds) / repetitions if wall_seconds and repetitions else "",
                "completed_total_wall_seconds": sum(completed_wall),
                "completed_mean_wall_seconds": mean(completed_wall) if completed_wall else "",
                "total_verifier_seconds": sum(verifier_seconds),
                "mean_verifier_seconds": mean(verifier_seconds) if verifier_seconds else "",
            }
        )
        aggregates.append(aggregate)
    return aggregates
