"""Stereographic S^3 → R^3 and Poincaré-ball Möbius primitives (standalone, for notebooks)."""
from __future__ import annotations

import numpy as np


def stereographic_s3_to_r3(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    *,
    pole: str = "north",
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stereographic projection of unit 4-vectors (on ``S^3``) to ``R^3``.

    ``pole='north'`` uses chart from ``(0,0,0,1)``: ``y_k = x_k / (1 - x_4)``, ``k=1,2,3``.
    ``pole='south'`` uses ``y_k = x_k / (1 + x_4)``.
    """
    if pole == "north":
        d = np.maximum(eps, 1.0 - x4)
    elif pole == "south":
        d = np.maximum(eps, 1.0 + x4)
    else:
        raise ValueError("pole must be 'north' or 'south'")
    return x1 / d, x2 / d, x3 / d


def mobius_ball_add(
    x1: float,
    x2: float,
    x3: float,
    y1: float,
    y2: float,
    y3: float,
) -> tuple[float, float, float]:
    """Gyrovector Möbius addition in the unit Poincaré ball (curvature -1)."""
    x2n = x1 * x1 + x2 * x2 + x3 * x3
    y2n = y1 * y1 + y2 * y2 + y3 * y3
    xy = x1 * y1 + x2 * y2 + x3 * y3
    den = 1 + 2 * xy + x2n * y2n
    if den < 1e-30:
        return 0.0, 0.0, 0.0
    inv = 1.0 / den
    c1 = 1 + 2 * xy + y2n
    c2 = 1 - x2n
    return (c1 * x1 + c2 * y1) * inv, (c1 * x2 + c2 * y2) * inv, (c1 * x3 + c2 * y3) * inv


def mobius_ball_to_origin(
    ax: float,
    ay: float,
    az: float,
    x1: float,
    x2: float,
    x3: float,
) -> tuple[float, float, float]:
    """Isometry sending ``a`` to ``0``: ``x ↦ (-a) ⊕ x``."""
    return mobius_ball_add(-ax, -ay, -az, x1, x2, x3)


def ball_to_origin_array(
    ax: float,
    ay: float,
    az: float,
    pts: np.ndarray,
) -> np.ndarray:
    """Apply ``mobius_ball_to_origin`` row-wise; ``pts`` is ``N×3``."""
    out = np.empty_like(pts, dtype=np.float64)
    for i in range(pts.shape[0]):
        out[i, 0], out[i, 1], out[i, 2] = mobius_ball_to_origin(ax, ay, az, float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
    return out
