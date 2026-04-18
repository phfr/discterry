# Hyperbolic PPI embeddings (Lorentz)

Train **Lorentz (hyperboloid)** node embeddings on a tab-separated edge list (`source`, `target` only). Uses **PyTorch Geometric** `RandomLinkSplit`, **geoopt** `Lorentz` + `RiemannianAdam`, link prediction with **BCE** on a Fermi-style logit of hyperbolic distance.

## Layout

- `src/config.py` — defaults and dtype
- `src/data.py` — TSV ingest, undirected dedupe, toy subsample, split
- `src/manifold_ops.py` — Minkowski inner, stable `arcosh` distance, Lorentz → Poincaré (ball)
- `src/model.py` — `LorentzNodeEmbedding`
- `src/train.py` — training, AUC/AP, checkpoints, `tqdm`
- `src/export.py` — CSV export with optional ball / naive 3D columns
- `src/export_arrow.py` — Arrow IPC (`nodes.arrow`, `edges.arrow`, `meta.json`) for the WebGPU viz (`pyarrow`, no PyG import)
- `viz/` — Vite + TypeScript WebGPU viewer (see [`viz/README.md`](viz/README.md))
- `scripts/project_ball_umap.py` — optional UMAP on ball coords (requires `umap-learn`)

## Run (you execute in WSL / your env)

From the repo root (e.g. `cd /mnt/c/Users/john/Documents/philipp/hbol`):

```bash
# Toy smoke test
python -m src.train --edges edges.tsv --toy-edges 2000 --epochs 3 --batch-size 2048

# Full graph (defaults: spatial dim 16, batch 8192, float64)
python -m src.train --edges edges.tsv --epochs 20 --device cuda

# Export embeddings after training
python -m src.export --checkpoint checkpoints/best.pt --out-csv embeddings.csv --include-ball --include-naive-3d

# Arrow pack for WebGPU viz (writes viz/public/data/)
python -m src.export_arrow --checkpoint checkpoints/best.pt --edges edges.tsv --out-dir viz/public/data
```

Flags: `--spatial-dim 32`, `--fp32`, `--lr`, `--seed`, `--val-ratio`, `--test-ratio`, `--checkpoint-dir`, etc. See `python -m src.train --help`.

## Notes

- **Float64** is the default for manifold stability; use `--fp32` if you need speed on the GPU.
- After each optimizer step the code calls **`projx`** on embedding rows so points stay on the hyperboloid.
- `edges.tsv` may contain extra columns; only **`source`** and **`target`** are read (`usecols`).
