from __future__ import annotations

import torch


def minkowski_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Lorentzian inner product ⟨u,v⟩ = -u₀v₀ + Σᵢ uᵢvᵢ (last dim ambient d+1)."""
    return -u[..., 0] * v[..., 0] + (u[..., 1:] * v[..., 1:]).sum(dim=-1)


def lorentz_distance(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Geodesic distance on the unit hyperboloid model matching -⟨u,v⟩ ≥ 1:
    d(u,v) = acosh(clamp(-⟨u,v⟩, min=1+eps)).
    """
    inner = minkowski_inner(u, v)
    z = torch.clamp(-inner, min=1.0 + eps)
    return torch.acosh(z)


def fermi_dirac_logits(
    dist: torch.Tensor, r: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Log-odds for a positive edge: higher when distance is small.
    logit = (r - d) / t  with t > 0.
    """
    return (r - dist) / t.clamp_min(1e-6)


def lorentz_to_poincare(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Stereographic projection from hyperboloid sheet (x₀ > 0) to the open unit d-ball.
    y = x_space / (x₀ + 1)
    """
    x0 = x[..., 0:1]
    space = x[..., 1:]
    denom = (x0 + 1.0).clamp_min(eps)
    return space / denom


def naive_ball_to_3d(y: torch.Tensor) -> torch.Tensor:
    """Take first 3 Euclidean coordinates of ball embedding (heuristic for viz)."""
    out = torch.zeros(y.shape[:-1] + (3,), device=y.device, dtype=y.dtype)
    m = min(3, y.shape[-1])
    out[..., :m] = y[..., :m]
    return out
