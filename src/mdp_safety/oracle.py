"""Interface of the weighted reachability oracle used by the safety verifier.

Each query includes same-scheduler coordinate bounds. StormRunner implements
this protocol. Injection allows deterministic state-machine
tests without replacing the real Storm implementation used in experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .model import DrnModel
from .storm_results import OracleQueryResult


class ReachabilityOracle(Protocol):
    version: str

    def build_prism(self, prism_path: Path, output_path: Path, constants: str | None = None) -> DrnModel:
        ...

    def query(
        self, base_model: DrnModel, terminal_to_dimension: dict[int, int],
        terminal_labels: list[str],
        weight: np.ndarray, precision: float, work_directory: Path, query_id: str,
    ) -> OracleQueryResult:
        ...
