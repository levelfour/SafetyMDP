"""Upper facets of a downward hull, without enumerating dominated box corners.

For scaled samples s >= 0, translate the hull by +1. Its polar is the
bounded, full-dimensional polytope {a >= 0: (s + 1) @ a <= 1}.
Every nonzero polar vertex gives a facet of conv(samples) - R_+^d.
All geometry here is floating point, not a formal completeness certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import HalfspaceIntersection, QhullError

from .safe_regions.base import RegionError, Vector


@dataclass(frozen=True)
class InnerFacet:
    normal: Vector
    bound: float
    support_point: Vector


def downward_hull_facets(samples: Vector) -> tuple[InnerFacet, ...]:
    """Return nonnegative, L2-unit facet normals in deterministic lexical order.

The downward hull is unbounded below; no x >= 0 faces are included.
For probability inputs, intersecting it with the positive orthant gives the
usual downward achievable region. No random Qhull perturbation is used.
"""
    points = np.asarray(samples, dtype=float)
    if (points.ndim != 2 or not len(points) or points.shape[1] < 1
            or not np.all(np.isfinite(points)) or np.any(points < 0)):
        raise RegionError("inner facet construction requires finite nonnegative samples")
    points = np.unique(points, axis=0)
    dimension = points.shape[1]
    if dimension == 1:
        index = int(np.argmax(points[:, 0]))
        return (InnerFacet(np.ones(1), float(points[index, 0]), points[index].copy()),)

    scales = np.max(points, axis=0)
    scales[scales == 0] = 1.0
    scaled = points / scales
    shifted = scaled + 1.0
    halfspaces = np.vstack((
        np.column_stack((shifted, -np.ones(len(points)))),
        np.column_stack((-np.eye(dimension), np.zeros(dimension))),
    ))
    # Each shifted coordinate is in [1, 2], so this is strictly interior.
    interior = np.full(dimension, 1.0 / (4.0 * dimension))
    try:
        vertices = HalfspaceIntersection(halfspaces, interior).intersections
    except (QhullError, ValueError) as error:
        raise RegionError(f"inner facet enumeration failed: {error}") from error
    if len(vertices) < dimension + 1 or not np.all(np.isfinite(vertices)):
        raise RegionError("inner polar enumeration returned invalid vertices")

    facets = []
    for vertex in vertices:
        if np.linalg.norm(vertex) < 1e-12:
            continue  # The polar origin represents no finite upper facet.
        if np.min(vertex) < -1e-9 or np.max(shifted @ vertex - 1.0) > 1e-8:
            raise RegionError("inner polar vertex failed feasibility validation")
        normal_scaled = np.maximum(vertex, 0.0)
        # Validate the polar-to-facet correspondence before changing scales.
        scaled_support = float(np.max(scaled @ normal_scaled))
        if abs(scaled_support - (1.0 - np.sum(normal_scaled))) > 1e-8:
            raise RegionError("inner polar vertex does not expose a supporting facet")
        # Divide in two stages to avoid overflow for very small coordinates.
        normal = normal_scaled * (np.min(scales) / scales)
        size = float(np.max(normal))
        if size <= 0 or not np.isfinite(size):
            raise RegionError("inner facet normal is numerically degenerate")
        normal /= size
        normal /= np.linalg.norm(normal)
        values = points @ normal
        index = int(np.argmax(values))
        # Recompute support from the original, unrounded lower-bound points.
        facets.append(InnerFacet(normal, float(values[index]), points[index].copy()))
    if not facets:
        raise RegionError("inner facet enumeration produced no upper facets")
    # Resolve exact indicator ties reproducibly, independently of Qhull order.
    # Indicator ranking belongs to the strategy, not to facet enumeration.
    facets.sort(key=lambda facet: (tuple(np.round(facet.normal, 14)), tuple(facet.normal)))
    return tuple(facets)
