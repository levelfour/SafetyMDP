"""Strategy factories, metadata and configuration validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .base import ProjectionStrategy
from .gauge import GaugeStrategy
from .norm import NormProjectionStrategy
from .sandwich_lk import LammelKueferStrategy, LammelKueferGaugeStrategy

_STRATEGIES: dict[str, Callable[[], ProjectionStrategy]] = {}


@dataclass(frozen=True)
class StrategyDefinition:
    """Single source of truth for factories, defaults, help and report labels."""

    name: str
    factory: Callable[[], ProjectionStrategy]
    label: str
    description: str
    default: bool = False
    comparison: bool = True


BUILTIN_STRATEGIES = (
    StrategyDefinition("l2", lambda: NormProjectionStrategy("l2"), r"$L^2$",
                       "Euclidean projection", default=True),
    StrategyDefinition("linf", lambda: NormProjectionStrategy("linf"), r"$L^\infty$",
                       "L-infinity projection", default=True),
    StrategyDefinition("gauge", GaugeStrategy, "Gauge", "safe-region gauge", default=True),
    StrategyDefinition("sandwich-lk", LammelKueferStrategy, "Sandwich (LK epsilon)",
                       "LK maximum epsilon", default=True),
    StrategyDefinition("sandwich-lk-gauge", LammelKueferGaugeStrategy, "Sandwich (LK gauge)",
                       "LK maximum relative gauge", default=True),
)
DEFAULT_PROJECTIONS = tuple(item.name for item in BUILTIN_STRATEGIES if item.default)
COMPARISON_PROJECTIONS = tuple(item.name for item in BUILTIN_STRATEGIES if item.comparison)
METHOD_LABELS = {item.name: item.label for item in BUILTIN_STRATEGIES}


def projection_help() -> str:
    return "; ".join(f"{item.name} ({item.description})" for item in BUILTIN_STRATEGIES)


def register_projection(name: str, factory: Callable[[], ProjectionStrategy]) -> None:
    """Register a strategy; external experiments can call this before verification."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("projection strategy name cannot be empty")
    _STRATEGIES[normalized] = factory


def projection_names() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def make_projection(name: str) -> ProjectionStrategy:
    # Do not misreport a KeyError raised *inside* a plugin as an unknown name.
    try:
        factory = _STRATEGIES[name.strip().lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown projection {name!r}; available: {', '.join(projection_names())}"
        ) from error
    return factory()


def validate_projection_names(values: object, *, context: str = "projections") -> tuple[str, ...]:
    """Reject stale configurations before creating output or running Storm."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{context} must be a non-empty array")
    if any(not isinstance(name, str) or name not in _STRATEGIES for name in values):
        raise ValueError(
            f"{context} must use registered strategies; available: {', '.join(projection_names())}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"{context} must contain distinct registered strategies")
    return tuple(values)


for definition in BUILTIN_STRATEGIES:
    register_projection(definition.name, definition.factory)
