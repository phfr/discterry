from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    edges_path: Path = Path("edges.tsv")
    spatial_dim: int = 16  # hyperbolic dimension d; ambient Lorentz coords are d+1
    batch_size: int = 8192
    epochs: int = 20
    lr: float = 0.02
    weight_decay: float = 0.0
    val_ratio: float = 0.05
    test_ratio: float = 0.1
    neg_sampling_ratio: float = 1.0  # negatives per positive in RandomLinkSplit
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float64"  # float64 | float32
    toy_edges: int | None = None  # if set, keep at most this many undirected edges after dedupe
    toy_nodes: int | None = None  # if set, keep only nodes with rank < N by degree (after toy_edges filter)
    checkpoint_dir: Path = Path("checkpoints")
    grad_clip: float | None = 10.0
    log_every: int = 50  # batches between loss prints inside tqdm postfix

    def torch_dtype(self):
        import torch

        return torch.float64 if self.dtype == "float64" else torch.float32
