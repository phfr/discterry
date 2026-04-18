from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.manifold_ops import lorentz_to_poincare, naive_ball_to_3d
from src.model import LorentzNodeEmbedding


def export_embeddings(
    checkpoint: Path,
    out_csv: Path,
    *,
    include_ball: bool = False,
    include_naive_3d: bool = False,
    chunk_rows: int = 8192,
) -> None:
    load_kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    ckpt = torch.load(checkpoint, **load_kw)
    idx_to_protein: list[str] = ckpt["idx_to_protein"]
    spatial_dim: int = int(ckpt["spatial_dim"])
    dtype_str: str = ckpt.get("dtype", "float64")
    dt = torch.float64 if dtype_str == "float64" else torch.float32

    model = LorentzNodeEmbedding(
        num_nodes=len(idx_to_protein),
        spatial_dim=spatial_dim,
        dtype=dt,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    amb = spatial_dim + 1
    n = len(idx_to_protein)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = ["protein_id"] + [f"x{i}" for i in range(amb)]
    if include_ball:
        header += [f"p{i}" for i in range(spatial_dim)]
    if include_naive_3d:
        header += ["viz_x", "viz_y", "viz_z"]

    first_chunk = True
    for start in tqdm(range(0, n, chunk_rows), desc="export chunks"):
        end = min(start + chunk_rows, n)
        idx = torch.arange(start, end, dtype=torch.long)
        with torch.no_grad():
            x = model.emb[idx].detach().cpu().to(dt)
            rows = {"protein_id": [idx_to_protein[i] for i in range(start, end)]}
            for j in range(amb):
                rows[f"x{j}"] = x[:, j].numpy()
            if include_ball:
                y = lorentz_to_poincare(x)
                for j in range(spatial_dim):
                    rows[f"p{j}"] = y[:, j].numpy()
            if include_naive_3d:
                y = lorentz_to_poincare(x)
                v3 = naive_ball_to_3d(y)
                rows["viz_x"] = v3[:, 0].numpy()
                rows["viz_y"] = v3[:, 1].numpy()
                rows["viz_z"] = v3[:, 2].numpy()
        df = pd.DataFrame(rows)
        df.to_csv(
            out_csv,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False


def _parse_args():
    p = argparse.ArgumentParser(description="Export Lorentz embeddings to CSV.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--include-ball", action="store_true")
    p.add_argument("--include-naive-3d", action="store_true")
    p.add_argument("--chunk-rows", type=int, default=8192)
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    export_embeddings(
        a.checkpoint,
        a.out_csv,
        include_ball=a.include_ball,
        include_naive_3d=a.include_naive_3d,
        chunk_rows=a.chunk_rows,
    )


if __name__ == "__main__":
    main()
