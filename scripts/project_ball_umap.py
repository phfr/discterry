#!/usr/bin/env python3
"""
Optional: UMAP from Poincaré ball columns (Euclidean on ℝᵈ) to 3D for visualization.
Requires: pip install umap-learn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import umap
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "umap-learn is not installed. Install with: pip install umap-learn"
    ) from e


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", type=Path, required=True, help="CSV with p0..p{d-1} columns")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="euclidean")
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)
    pcols = sorted([c for c in df.columns if c.startswith("p") and c[1:].isdigit()], key=lambda x: int(x[1:]))
    if not pcols:
        raise SystemExit("No p0, p1, ... columns found; export with --include-ball first.")
    X = df[pcols].to_numpy(dtype=np.float64)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        verbose=True,
    )
    emb3 = reducer.fit_transform(X)
    out = df[["protein_id"]].copy() if "protein_id" in df.columns else pd.DataFrame()
    out["umap_x"] = emb3[:, 0]
    out["umap_y"] = emb3[:, 1]
    out["umap_z"] = emb3[:, 2]
    out.to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)


if __name__ == "__main__":
    main()
