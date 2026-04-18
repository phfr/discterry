from __future__ import annotations

import torch
import torch.nn as nn
from geoopt import Lorentz, ManifoldParameter

from src.manifold_ops import fermi_dirac_logits, lorentz_distance


class LorentzNodeEmbedding(nn.Module):
    """Learnable node embeddings on the Lorentz hyperboloid (ambient dim = spatial_dim + 1)."""

    def __init__(
        self,
        num_nodes: int,
        spatial_dim: int,
        *,
        std: float = 0.02,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        # k must match embedding dtype or random_normal raises (see geoopt Lorentz.random_normal).
        self.manifold = Lorentz(k=torch.tensor(1.0, dtype=dtype))
        self.spatial_dim = spatial_dim
        amb = spatial_dim + 1
        # geoopt: random_normal(n, dim, std=..., dtype=...)
        x = self.manifold.random_normal(num_nodes, amb, std=std, dtype=dtype)
        self.emb = ManifoldParameter(x, manifold=self.manifold)
        # Decoder: learnable Fermi–Dirac margin and temperature (t > 0).
        self.dec_r = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.dec_log_t = nn.Parameter(torch.tensor(0.0, dtype=dtype))

    def forward(self, node_idx: torch.Tensor) -> torch.Tensor:
        return self.emb[node_idx]

    def temperature(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.dec_log_t) + 1e-4

    def link_logits(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        u = self.emb[src]
        v = self.emb[dst]
        d = lorentz_distance(u, v)
        return fermi_dirac_logits(d, self.dec_r, self.temperature())

    def project_embeddings(self) -> None:
        with torch.no_grad():
            self.manifold.projx(self.emb.data)
