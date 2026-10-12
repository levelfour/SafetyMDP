"""Per-run approximation state and vertex-enumeration measurements."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from ..polytope import OuterPolytope
from .results import QueryRecord, SelectionRecord


@dataclass
class VerificationState:
    outer: OuterPolytope
    queries: list[QueryRecord] = field(default_factory=list)
    selections: list[SelectionRecord] = field(default_factory=list)
    pareto_samples: list[list[float]] = field(default_factory=list)
    inner_points: list[list[float]] = field(default_factory=list)
    witness: dict[str, Any] | None = None
    status: str = "UNKNOWN"
    reason: str = "maximum number of refinements reached"
    completed_iterations: int = 0
    vertex_enumeration_limited: bool = False
    outer_vertex_enumeration_call_count: int = 0
    outer_vertex_enumeration_seconds: float = 0.0
    maximum_outer_vertex_count: int = 0
    final_vertices: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.final_vertices = np.empty((0, self.outer.dimension), dtype=float)

    def enumerate_vertices(self) -> np.ndarray:
        started = time.perf_counter()
        self.outer_vertex_enumeration_call_count += 1
        try:
            vertices = self.outer.vertices()
        finally:
            self.outer_vertex_enumeration_seconds += time.perf_counter() - started
        self.maximum_outer_vertex_count = max(self.maximum_outer_vertex_count, len(vertices))
        return vertices
