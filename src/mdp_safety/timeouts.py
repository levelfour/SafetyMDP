"""Explicit optional time limits shared by the CLI and experiment settings."""

from __future__ import annotations

import math


def validate_timeout(value: float | None, name: str = "timeout") -> None:
    """Accept None for no deadline, otherwise require positive finite seconds."""
    if value is None:
        return
    try:
        valid = not isinstance(value, bool) and math.isfinite(value) and value > 0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be positive and finite, or None to disable")


def parse_timeout(value: str) -> float | None:
    """CLI spelling: a positive number of seconds or the explicit token 'none'."""
    if value.strip().lower() == "none":
        return None
    seconds = float(value)
    validate_timeout(seconds)
    return seconds


def format_timeout(value: float | None) -> str:
    return "disabled" if value is None else f"{value:g} s"
