# D-Mercator 3D analysis notebooks

Independent suite for **native 3-hyperbolic** D-Mercator runs (e.g. `d-mercator-run/d3/`) where `edges_GC.inf_coord` has **four** `Inf.Pos.*` columns (`DIMENSION 3`). This folder does **not** import [`notebooks/dmercator/`](../dmercator/); use [`dmercator3d_io.py`](dmercator3d_io.py) instead.

## Run layout

Artifacts live under `d-mercator-run/<subdir>/` (default **`d3`**):

- `edges_GC.inf_coord` — vertex table, variable `Inf.Pos.1` … `Inf.Pos.{D+1}`
- `edges_GC.edge` — undirected edgelist

Set `RUN_SUBDIR = "d3"` in each notebook, or export **`DMERCATOR_RUN`** before Jupyter.

## Working directory

Start Jupyter with **cwd = `notebooks/dmercator3d/`** so `import dmercator3d_io` and `import ball_projection` resolve.

## Suggested order

1. `00_load_and_sanity.ipynb` — paths, parse, `cache/merged.parquet`
2. `01_exploratory_qa.ipynb`
3. `02_sphere_S3_views.ipynb` … `02d_ball_focus_mobius.ipynb`, `viz_plotly_3d.ipynb`
4. `03_link_prediction_sanity.ipynb`
5. `07_umap_*.ipynb` … `14_triplet_distance_sanity.ipynb`

## Python dependencies

**Core:** `pandas`, `numpy`, `networkx`, `matplotlib`, `scipy`, `pyarrow`, `jupyter`

| Notebooks | Extra install |
|-----------|----------------|
| UMAP (`07`–`09`) | `umap-learn` |
| Clustering / metrics (`10`–`11`, `13`) | `scikit-learn` (Louvain: `networkx.community.louvain_communities`, NX ≥ 2.8) |
| `viz_plotly_3d.ipynb` | `plotly` (optional `kaleido`) |
| Optional HDBSCAN cell in `10` | `hdbscan` |

## Helpers

- [`dmercator3d_io.py`](dmercator3d_io.py) — `parse_inf_coord`, `normalize_direction_nd`, `paths_for_run`, …
- [`ball_projection.py`](ball_projection.py) — stereographic `S^3 → R^3`, Möbius ball map
- [`native_hyperbolic.py`](native_hyperbolic.py) — Lorentz hyperboloid map from `(unit Inf.Pos.*, Inf.Hyp.Rad)` and geodesic distance (used in `03_link_prediction_sanity.ipynb`)

See also [pipeline.md](../../pipeline.md) for how `inf_coord` relates to the disk export in `notebooks/dmercator/`.
