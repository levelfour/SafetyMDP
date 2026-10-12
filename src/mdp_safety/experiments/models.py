"""Benchmark input records, settings, stable CSV fields and statuses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping

from ..defaults import DEFAULT_TIMEOUT_SECONDS
from ..timeouts import validate_timeout
from ..strategies.registry import DEFAULT_PROJECTIONS, validate_projection_names
from ..verification.options import VerificationOptions

FINAL_FAILURE_STATUSES = frozenset({"TIMEOUT", "ERROR"})
DECISIVE_VERDICTS = frozenset({"SAFE", "UNSAFE"})

SUMMARY_FIELDS = (
    "suite",
    "family",
    "name",
    "model",
    "specification",
    "expected_status",
    "label_matches",
    "projection",
    "repetition",
    "execution_order",
    "status",
    "elapsed_seconds",
    "wall_seconds",
    "iterations",
    "support_queries",
    "oracle_calls",
    "weight_selection_calls",
    "weight_selection_seconds",
    "weight_selection_cpu_seconds",
    "mean_weight_selection_ms",
    "outer_vertex_enumeration_calls",
    "outer_vertex_enumeration_seconds",
    "maximum_outer_vertices",
    "storm_calls",
    "states",
    "choices",
    "storm_version",
    "exit_code",
    "error",
)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _infer_expected_status(name: str, specification: Path) -> str | None:
    for candidate in (name.lower(), specification.stem.lower()):
        if candidate.endswith("-unsafe") or "-unsafe-" in candidate:
            return "UNSAFE"
        if candidate.endswith("-safe") or "-safe-" in candidate:
            return "SAFE"
    return None


def _infer_family(name: str) -> str:
    # Strip a dimension/size suffix when inferring a manifest instance's family.
    match = re.match(r"(.+?)-(?:n|d|b)\d", name)
    if match:
        return match.group(1)
    return name


@dataclass(frozen=True)
class BenchmarkInstance:
    """One model/specification pair from a benchmark manifest."""

    name: str
    model: Path
    specification: Path
    projections: tuple[str, ...]
    constants: str | None
    suite: str
    family: str
    expected_status: str | None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        manifest_directory: Path,
        default_suite: str,
    ) -> "BenchmarkInstance":
        try:
            name = str(raw["name"])
            model = _resolve(manifest_directory, str(raw["model"]))
            specification = _resolve(
                manifest_directory, str(raw["specification"])
            )
        except KeyError as error:
            raise ValueError(f"manifest instance is missing {error.args[0]!r}") from error
        if not name:
            raise ValueError("benchmark instance name must not be empty")
        projections = validate_projection_names(
            raw.get("projections", DEFAULT_PROJECTIONS), context=f"{name}: projections",
        )
        expected = raw.get("expected_status")
        if expected is not None:
            expected = str(expected).upper()
            if expected not in {"SAFE", "UNSAFE"}:
                raise ValueError(
                    f"{name}: expected_status must be SAFE or UNSAFE"
                )
        else:
            expected = _infer_expected_status(name, specification)
        constants = raw.get("constants")
        return cls(
            name=name,
            model=model,
            specification=specification,
            projections=projections,
            constants=str(constants) if constants else None,
            suite=str(raw.get("suite", default_suite)),
            family=str(raw.get("family", _infer_family(name))),
            expected_status=expected,
        )


@dataclass(frozen=True)
class BenchmarkJob:
    instance: BenchmarkInstance
    projection: str
    repetition: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.instance.name, self.projection, self.repetition)


@dataclass(frozen=True)
class BenchmarkSettings:
    manifest: Path
    output_directory: Path
    precision: float = 1e-7
    minimum_precision: float = 1e-12
    geometry_tolerance: float = 1e-8
    max_iterations: int = 100
    storm_timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS
    run_timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS
    storm: str = "storm"
    repetitions: int = 1
    record_scheduler_actions: bool = False
    resume: bool = False
    sandwich_precision: float = 1e-6
    projection_override: tuple[str, ...] | None = None

    def validate(self) -> None:
        if self.projection_override is not None:
            validate_projection_names(self.projection_override, context="projection override")
        VerificationOptions(
            storm_precision=self.precision,
            minimum_storm_precision=self.minimum_precision,
            geometry_tolerance=self.geometry_tolerance,
            max_iterations=self.max_iterations,
            storm_timeout_seconds=self.storm_timeout_seconds,
            sandwich_precision=self.sandwich_precision,
        ).validate()
        if type(self.repetitions) is not int or self.repetitions < 1:
            raise ValueError("repetitions must be a positive integer")
        validate_timeout(self.run_timeout_seconds, "run timeout")


CommandExecutor = Callable[
    [list[str], float | None], tuple[subprocess.CompletedProcess[str], bool]
]
