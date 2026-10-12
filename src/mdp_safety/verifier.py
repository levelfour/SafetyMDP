"""Public verifier API and model/working-directory lifecycle.

The CEGAR state machine, measurements and result schema live in verification.
This module also exposes verification options, result records and exceptions.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
from typing import Any

from .drn import DrnError
from .oracle import ReachabilityOracle
from .polytope import PolytopeError
from .regions import RegionError, SafeRegion
from .storm import StormError, StormRunner
from .strategies import ProjectionStrategy, make_projection
from .strategies.sandwich import InnerFacetStrategy
from .verification.engine import VerificationRun
from .verification.options import VerificationOptions
from .verification.results import (
    QueryRecord, SafetyResult, SelectionRecord, _float_list, _pareto_filter,
    build_result, make_query_record,
)


class SafetyVerifier:
    def __init__(
        self,
        options: VerificationOptions | None = None,
        storm_executable: str = "storm",
        *,
        storm_runner: ReachabilityOracle | None = None,
    ) -> None:
        self.options = options or VerificationOptions()
        self.options.validate()
        self.projection: ProjectionStrategy = make_projection(self.options.projection)
        if isinstance(self.projection, InnerFacetStrategy):
            self.projection.precision = self.options.sandwich_precision
        self.storm = storm_runner if storm_runner is not None else StormRunner(
            storm_executable, self.options.storm_timeout_seconds
        )

    def verify(
        self,
        model_path: Path,
        terminal_labels: list[str],
        safe_region: SafeRegion,
        constants: str | None = None,
        working_directory: Path | None = None,
        terminal_sets: list[dict[str, Any]] | None = None,
        terminalize: bool = False,
    ) -> SafetyResult:
        started = time.perf_counter()
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if safe_region.dimension != len(terminal_labels):
            raise ValueError(
                f"safe region has dimension {safe_region.dimension}, but "
                f"{len(terminal_labels)} terminal labels were supplied"
            )
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if working_directory is None:
            temporary = tempfile.TemporaryDirectory(prefix="mdp-safety-")
            work = Path(temporary.name)
        else:
            work = working_directory.resolve()
            work.mkdir(parents=True, exist_ok=True)
        try:
            base_model = self.storm.build_prism(model_path, work / "model.drn", constants)
            if terminal_sets is not None:
                base_model = base_model.with_derived_labels(terminal_sets)
            terminal_map = base_model.validate_terminal_labels(
                terminal_labels, require_absorbing=not terminalize
            )
            state = VerificationRun(
                self.options, self.projection, self.storm, base_model,
                terminal_labels, terminal_map, safe_region, work,
            ).execute()
            result = build_result(
                state, self.options, model_path, terminal_labels, terminalize,
                self.projection.name, self.storm.version, base_model, started,
            )
            result.storm_configuration.update(getattr(self.storm, "configuration", {}))
            return result
        finally:
            if temporary is not None and not self.options.keep_working_files:
                temporary.cleanup()

    def _query_record(self, *args, **kwargs) -> QueryRecord:
        """Build a query record using this verifier's strategy name."""
        return make_query_record(self.projection.name, *args, **kwargs)


__all__ = [
    "DrnError", "PolytopeError", "RegionError", "SafetyResult",
    "SafetyVerifier", "StormError", "VerificationOptions",
]
