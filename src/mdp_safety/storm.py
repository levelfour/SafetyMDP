"""Storm CLI model construction and native-API sound OVI/scheduler queries."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from .defaults import DEFAULT_TIMEOUT_SECONDS
from .timeouts import validate_timeout
from .drn import DrnModel, read_drn, read_scheduler, write_drn
from .storm_api import SCHEDULER_BACKEND, run_query_worker, stormpy_version
from .storm_results import (
    IntervalResult, OracleQueryResult, StormError, SupportResult, probability_interval,
)


SUCCESS_LABEL = "__mdpsafety_success__"
FAILURE_LABEL = "__mdpsafety_failure__"


class StormRunner:
    def __init__(self, executable: str = "storm", timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS) -> None:
        validate_timeout(timeout_seconds, "Storm timeout")
        resolved = shutil.which(executable)
        if resolved is None:
            raise StormError(
                f"Storm executable {executable!r} was not found; install Storm or pass --storm"
            )
        self.executable = resolved
        self.timeout_seconds = timeout_seconds
        self.version = self._version()
        self.api_version = stormpy_version()
        self.api_storm_version: str | None = None

    @property
    def configuration(self) -> dict:
        return {
            "backend": SCHEDULER_BACKEND,
            "stormpy_version": self.api_version,
            "stormpy_storm_version": self.api_storm_version,
            "scheduler_application": "SparseMdp.apply_scheduler",
            "coordinate_model": "DTMC induced from weighted terminal gadget",
            "coordinate_solver": "native optimistic-value-iteration",
        }

    def _version(self) -> str:
        completed = self._run([self.executable, "--version"])
        match = re.search(r"Storm\s+([^\s]+)", completed.stdout)
        if not match:
            raise StormError("could not determine the Storm version")
        return match.group(1)

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise StormError(f"Storm timed out after {self.timeout_seconds:g}s") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise StormError(f"Storm failed (exit {completed.returncode}):\n{detail}")
        return completed

    def build_prism(
        self,
        prism_path: Path,
        output_path: Path,
        constants: str | None = None,
    ) -> DrnModel:
        command = [
            self.executable,
            "--prism",
            str(prism_path),
            "--buildfull",
            "--buildstateval",
            "--explchecks",
            "--digits",
            "17",
            "--exportbuild",
            str(output_path),
            "drn",
        ]
        if constants:
            command.extend(("--constants", constants))
        self._run(command)
        if not output_path.exists():
            raise StormError("Storm did not create the requested explicit model")
        return read_drn(output_path)

    def query(
        self,
        base_model: DrnModel,
        terminal_to_dimension: dict[int, int],
        terminal_labels: list[str],
        weight: np.ndarray,
        precision: float,
        work_directory: Path,
        query_id: str,
    ) -> OracleQueryResult:
        started = time.perf_counter()
        if base_model.validate_terminal_labels(terminal_labels, require_absorbing=False) != terminal_to_dimension:
            raise StormError("terminal labels and coordinate mapping disagree")
        weighted = base_model.weighted_terminal_gadget(
            terminal_to_dimension,
            weight,
            SUCCESS_LABEL,
            FAILURE_LABEL,
        )
        model_path = work_directory / f"weighted-{query_id}.drn"
        result_path = work_directory / f"weighted-{query_id}.result.json"
        scheduler_path = work_directory / f"weighted-{query_id}.scheduler.json"
        write_drn(weighted, model_path)
        request_path = work_directory / f"query-{query_id}.json"
        request = {
            "model_path": str(model_path.resolve()),
            "support_property": f'Pmax=? [F "{SUCCESS_LABEL}"]',
            "precision": precision,
            "scheduler_path": str(scheduler_path.resolve()),
            "support_result_path": str(result_path.resolve()),
            "induced_path": str((work_directory / f"induced-{query_id}.drn").resolve()),
            "terminal_labels": terminal_labels,
            "coordinate_result_paths": [str((work_directory / f"induced-{query_id}-{i}.result.json").resolve())
                                        for i in range(len(terminal_labels))],
        }
        request_path.write_text(json.dumps(request, allow_nan=False) + "\n", encoding="utf-8")
        preparation_seconds = time.perf_counter() - started
        events, command = run_query_worker(request_path, len(terminal_labels), self.timeout_seconds)
        self.api_storm_version = events[0]["storm_version"]

        def interval(event: dict, cap: float, overhead: float = 0.) -> IntervalResult:
            estimate = float(event["estimate"])
            if not np.isfinite(estimate):
                raise StormError("Storm API returned a nonfinite probability")
            lower, upper = probability_interval(estimate, precision, cap)
            return IntervalResult(estimate, lower, upper, precision,
                                  event["elapsed_seconds"] + overhead, command)

        support_interval = interval(events[1], float(np.max(weight)),
                                    preparation_seconds + events[0]["elapsed_seconds"])
        scheduled = read_scheduler(scheduler_path, weighted)
        actions: dict[int, str] = {}
        for state in base_model.states:
            if state.identifier in terminal_to_dimension:
                continue
            # Completion and evaluation already happened on the same native
            # scheduler. The exported map is only for the witness artifact.
            if state.identifier not in scheduled:
                raise StormError(f"native scheduler export omits state {state.identifier}")
            local_index = scheduled[state.identifier]
            if local_index < 0 or local_index >= len(state.actions):
                raise StormError(
                    f"invalid local scheduler choice at state {state.identifier}"
                )
            actions[state.identifier] = state.actions[local_index].identifier
        coordinates = [interval(event, 1., events[2]["elapsed_seconds"] if i == 0 else 0.)
                       for i, event in enumerate(events[3:])]
        return OracleQueryResult(SupportResult(support_interval, scheduled, actions, weighted), coordinates)
