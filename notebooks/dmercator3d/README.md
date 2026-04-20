# D-Mercator 3D analysis notebooks

Independent suite for **native 3D hyperbolic** D-Mercator runs (e.g. `d-mercator-run/d3/`) where `edges_GC.inf_coord` has **four** `Inf.Pos.*` columns (`DIMENSION 3`). Typical input is a **PPI (or other) undirected edgelist** plus the embedding produced by [D-Mercator](https://github.com/networkgeometry/d-mercator).

This folder does **not** import [`notebooks/dmercator/`](../dmercator/); use [`dmercator3d_io.py`](dmercator3d_io.py) for IO and paths.

## Run layout

Artifacts live under `d-mercator-run/<subdir>/` (default **`d3`**):

| File | Role |
|------|------|
| `edges_GC.inf_coord` | Vertex table: `Inf.Kappa`, `Inf.Hyp.Rad`, `Inf.Pos.1` … `Inf.Pos.{D+1}` |
| `edges_GC.edge` | Undirected edgelist |

Set `RUN_SUBDIR = "d3"` in a notebook, or export **`DMERCATOR_RUN`** before Jupyter so all `paths_for_run()` calls agree.

## Working directory

Start Jupyter with **cwd = `notebooks/dmercator3d/`** so local imports resolve (`dmercator3d_io`, `ball_projection`, `native_hyperbolic`).

## Cached data

Notebook **`00`** writes **`cache/merged.parquet`** (coordinates + graph **degree**). Later notebooks load this file to avoid re-parsing `inf_coord`. The `cache/` directory is **tracked in git** so clones can run downstream notebooks without re-running **`00`** (regenerate and commit when inputs change).

## Notebooks (suggested order)

| # | Notebook | What it does |
|---|----------|----------------|
| **00** | [`00_load_and_sanity.ipynb`](00_load_and_sanity.ipynb) | Paths, `parse_inf_coord`, `load_edges_graph`, merge **degree**, write `cache/merged.parquet`, sanity checks |
| **01** | [`01_exploratory_qa.ipynb`](01_exploratory_qa.ipynb) | Distributions / correlations on raw vs normalized features |
| **02** | [`02_sphere_S3_views.ipynb`](02_sphere_S3_views.ipynb) | Stereographic **S³ → ℝ³** scatter; subsampled views colored by `Inf.Hyp.Rad` |
| **02b** | [`02b_ball_views_seeds.ipynb`](02b_ball_views_seeds.ipynb) | Same ball chart with **seed** proteins highlighted |
| **02c** | [`02c_ball_focus_euclidean.ipynb`](02c_ball_focus_euclidean.ipynb) | **Euclidean** lerp between two points in ℝ³ (diagnostic, not hyperbolic geodesic) |
| **02d** | [`02d_ball_focus_mobius.ipynb`](02d_ball_focus_mobius.ipynb) | **Möbius** ball map sending a focus protein toward the origin |
| **Viz** | [`viz_plotly_3d.ipynb`](viz_plotly_3d.ipynb) | Optional **Plotly** interactive 3D (same stereographic pipeline) |
| **03** | [`03_link_prediction_sanity.ipynb`](03_link_prediction_sanity.ipynb) | **Native hyperbolic** geodesic scores vs random non-edges (uses `native_hyperbolic.py`) |
| **07** | [`07_umap_features_hyp_rad.ipynb`](07_umap_features_hyp_rad.ipynb) | **UMAP** on **[direction \| log1p(rad) \| log1p(κ)]**; multiple color views (radius, density, κ, degree) |
| **08** | [`08_umap_direction_only.ipynb`](08_umap_direction_only.ipynb) | Same pipeline, **direction-only** features |
| **09** | [`09_umap_colored_by_degree.ipynb`](09_umap_colored_by_degree.ipynb) | Same as **07** features; scatter colored by **degree** |
| **10** | [`10_clustering_kmeans_hdbscan.ipynb`](10_clustering_kmeans_hdbscan.ipynb) | **Full-graph** **[U \| log1p(rad)]**: KMeans sweep, silhouette (optional subsampled score), Davies–Bouldin; **HDBSCAN** parameter grid + diagnostics; optional ARI vs KMeans |
| **11** | [`11_graph_vs_embedding_communities.ipynb`](11_graph_vs_embedding_communities.ipynb) | **Louvain** on PPI **vs** Louvain on **mutual kNN** in feature space; **ARI** (all overlap vertices, no random vertex cap) |
| **12** | [`12_hub_geometry_correlations.ipynb`](12_hub_geometry_correlations.ipynb) | Pearson correlations: **degree**, `Inf.Hyp.Rad`, `Inf.Kappa`, stereographic **`r_ball`**; **exact betweenness** on full induced subgraph vs geometry |
| **13** | [`13_embedding_neighbor_overlap.ipynb`](13_embedding_neighbor_overlap.ipynb) | kNN in embedding vs graph neighbors — **precision@K** (random **3000**-vertex subsample for runtime; edit in notebook) |
| **14** | [`14_triplet_distance_sanity.ipynb`](14_triplet_distance_sanity.ipynb) | Triangle-inequality violation rates: **Euclidean**, **squared Euclidean**, **cosine** (full **X** and **U** only), **angular** geodesic on **S³** (shared random triples) |

There are no `04`–`06` notebooks in this folder; numbering matches the broader analysis sequence.

## Python dependencies

**Core:** `pandas`, `numpy`, `networkx`, `matplotlib`, `scipy`, `pyarrow`, `jupyter`

| Area | Extra install |
|------|----------------|
| UMAP (**07**–**09**) | `umap-learn` |
| kNN / clustering / metrics (**03**, **10**, **11**, **13**) | `scikit-learn` |
| Louvain (**11**) | **NetworkX ≥ 2.8** (`nx.community.louvain_communities`) |
| **10** optional density clustering | `hdbscan` |
| **viz_plotly_3d.ipynb** | `plotly` (optional `kaleido` for static export) |

**14** uses only NumPy. **12** betweenness uses NetworkX on the induced subgraph (can be slow on very large graphs; exact `k=None`).

## Helpers (Python modules)

| Module | Purpose |
|--------|---------|
| [`dmercator3d_io.py`](dmercator3d_io.py) | `repo_root`, `get_run_dir`, `paths_for_run`, `parse_inf_coord`, `normalize_direction_nd`, `load_edges_graph`, `load_merged_parquet` / `save_merged_parquet` |
| [`ball_projection.py`](ball_projection.py) | Stereographic **S³ → ℝ³** and **Möbius** maps on the ball |
| [`native_hyperbolic.py`](native_hyperbolic.py) | Lorentz / hyperboloid position from `(unit Inf.Pos.*, Inf.Hyp.Rad)` and **hyperbolic geodesic distance** (used in **03**) |

## See also

- [pipeline.md](../../pipeline.md) — how `inf_coord` relates to the Poincaré-disk export in `notebooks/dmercator/`
- Upstream embedding tool: [networkgeometry/d-mercator](https://github.com/networkgeometry/d-mercator)
