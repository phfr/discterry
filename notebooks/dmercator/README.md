# D-Mercator PPI notebooks

Notebooks for exploring [D-Mercator](https://www.nature.com/articles/s41467-023-43337-5) outputs on a protein–protein interaction graph, using the same hyperbolic picture as [Poincaré embeddings](https://arxiv.org/abs/1705.08039).

## Run layout

Artifacts for one inference run live under `d-mercator-run/<name>/` (today only `d2/`):

- `edges_GC.inf_coord` — vertex table and embedding coordinates  
- `edges_GC.edge` — edgelist  
- `edges_GC.inf_log` — optional human-readable log  

## Switching runs

In the first code cell of any notebook, set `RUN_SUBDIR = "d3"` (or your folder name), **or** set environment variable `DMERCATOR_RUN` before starting Jupyter (same string as the subfolder name under `d-mercator-run/`).

## Suggested order

1. `00_load_and_sanity.ipynb` — paths, load, sanity checks, writes `cache/merged.parquet`  
2. `01_exploratory_qa.ipynb` — distributions and correlations  
3. `02_disk_views.ipynb` — disk projections and sampled PPI edges (matplotlib)  
4. `03_link_prediction_sanity.ipynb` — crude edge vs non-edge checks  
5. `04_export_disk_web_bundle.ipynb` — export `nodes.parquet` / `edges.parquet` / `meta.json` for **`viz/discterry/public/data/`** (2D disk) and, via `export_discterry_public_data_3d`, **`viz/discterry/public/data3d/`** for **`#3d`** ball mode  
6. `viz_*.ipynb` — one notebook per rendering stack for comparable exports  

## Python environment

From the repo root (or this folder), install **core** dependencies first:

```bash
pip install pandas numpy networkx matplotlib scipy pyarrow jupyter
```

Optional stacks (install only what you need):

| Notebook | Extra install |
|----------|----------------|
| `viz_matplotlib_raster_vector.ipynb` | (core only) |
| `viz_datashader.ipynb` | `datashader` (optional `colorcet` for nicer ramps) |
| `viz_plotly_kaleido.ipynb` | `plotly` `kaleido` |
| `viz_bokeh.ipynb` | `bokeh` |
| `viz_holoviews_datashader.ipynb` | `holoviews` `datashader` `param` — **heavy**; safe to skip |

## Working directory

Open Jupyter with the notebook folder as the current working directory (`notebooks/dmercator/`) so `dmercator_io` imports resolve.

## HoloViews + Datashader

`viz_holoviews_datashader.ipynb` is optional: it may be slow or fail on constrained machines. The README above marks it as skippable; the notebook itself is a short stub with import checks.
