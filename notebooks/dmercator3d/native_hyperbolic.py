"""Lorentz hyperboloid helpers for (unit direction, Inf.Hyp.Rad) rows from D-Mercator `inf_coord`.

Assumes ``Inf.Hyp.Rad`` is the hyperbolic radial coordinate ρ in the usual upper-sheet
parametrization in ℝ^{D+2} with signature ``(-, +, …, +)``:

    X = ( cosh(ρ),  sinh(ρ) · ŝ )

with ŝ a unit vector in ℝ^{D+1} (the normalized ``Inf.Pos.*`` block).

Geodesic distance on **unit curvature** (sectional curvature −1):

    d(u, v) = acosh( max(1+ε, −⟨u, v⟩) ),   ⟨u,v⟩ = −u₀v₀ + Σᵢ uᵢvᵢ .

``Inf.Kappa`` does **not** appear in geodesic length; it enters the PS / Fermi–Dirac
**connection probability**, which is not the same object as hyperbolic distance.

If inference uses a global length scale R (curvature −1/R²), multiply ``d`` by R.
The ``radius_H^D+1`` header field is a model hyperparameter—whether it equals this R
depends on the D-Mercator implementation; treat scaling as a documentation/paper check
when you need numbers in physical units.
"""
from __future__ import annotations

import numpy as np


def lorentz_inner(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minkowski inner product; last axis is ambient dimension (D+2 for native D)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return -u[..., 0] * v[..., 0] + np.sum(u[..., 1:] * v[..., 1:], axis=-1)


def directions_hyp_to_lorentz(s_unit: np.ndarray, hyp_rad: np.ndarray) -> np.ndarray:
    """Stack Lorentz coordinates (N × (D+2)) from N×(D+1) unit rows ``s_unit`` and radii ``hyp_rad``."""
    s = np.asarray(s_unit, dtype=np.float64)
    rho = np.asarray(hyp_rad, dtype=np.float64).reshape(-1, 1)
    nrm = np.linalg.norm(s, axis=1, keepdims=True)
    nrm = np.where(nrm == 0.0, 1.0, nrm)
    s = s / nrm
    c0 = np.cosh(rho)
    tail = np.sinh(rho) * s
    return np.concatenate([c0, tail], axis=1)


def lorentz_geodesic_distance(u: np.ndarray, v: np.ndarray) -> float:
    inner = float(lorentz_inner(np.asarray(u).reshape(1, -1), np.asarray(v).reshape(1, -1))[0])
    # On-sheet points satisfy −⟨u,v⟩ ≥ 1; float noise can dip slightly below 1 for u ≈ v.
    z = max(1.0, float(-inner))
    return float(np.arccosh(z))


def lorentz_geodesic_pairwise(X: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Geodesic distance for each pair (i[k], j[k]) into rows of ``X``."""
    u = X[np.asarray(i, dtype=int)]
    v = X[np.asarray(j, dtype=int)]
    inner = lorentz_inner(u, v)
    z = np.maximum(-inner, 1.0)
    return np.arccosh(z)
