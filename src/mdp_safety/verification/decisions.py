"""Pure refinement decisions, independent of model checking and geometry solvers."""
from __future__ import annotations


def query_outcome(
    lower_violation: float,
    support_upper: float,
    counterexample_value: float,
    tolerance: float,
    allow_inner_refinement: bool,
    inner_progress: bool,
) -> str:
    """Preserve the precedence: witness, strict outer cut, inner progress, retry."""
    if lower_violation > tolerance:
        return "unsafe-witness"
    if support_upper < counterexample_value - tolerance:
        return "refined"
    if allow_inner_refinement and inner_progress:
        return "inner-refined"
    return "retry-with-higher-precision"
