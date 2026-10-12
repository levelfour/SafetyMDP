"""Load terminal-set and safe-region JSON independently of CLI execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .safe_regions import SafeRegion, region_from_dict


@dataclass(frozen=True)
class SafetySpecification:
    terminal_labels: list[str]
    safe_region: SafeRegion
    terminal_sets: list[dict[str, Any]] | None = None
    terminalize: bool = False


def specification_from_dict(specification: dict[str, Any], tolerance: float = 1e-9) -> SafetySpecification:
    terminal_sets = specification.get("terminal_sets")
    if terminal_sets is not None:
        if not isinstance(terminal_sets, list) or not terminal_sets:
            raise ValueError("terminal_sets must be a non-empty JSON object array")
        terminal_labels = [
            definition.get("name") if isinstance(definition, dict) else None
            for definition in terminal_sets
        ]
    else:
        terminal_labels = specification["terminal_labels"]
    if not isinstance(terminal_labels, list) or not all(
        isinstance(label, str) and label for label in terminal_labels
    ):
        raise ValueError("terminal_labels must be a non-empty JSON string array")
    terminalize = specification.get("terminalize", False)
    if not isinstance(terminalize, bool):
        raise ValueError("terminalize must be a Boolean")
    safe_region = region_from_dict(
        specification["safe_region"],
        tolerance,
        dimension_hint=len(terminal_labels),
    )
    return SafetySpecification(terminal_labels, safe_region, terminal_sets, terminalize)


def load_specification(path: Path, tolerance: float = 1e-9) -> SafetySpecification:
    return specification_from_dict(json.loads(path.read_text(encoding="utf-8")), tolerance)
