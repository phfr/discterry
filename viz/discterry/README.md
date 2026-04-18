# Discterry — Poincaré disk / ball (WebGPU)

**Discterry** is an interactive view of a PPI graph run: choose a **focus** protein and **seed** proteins. **2D (default):** Poincaré **disk** (same logic as [`notebooks/dmercator/02d_disk_focus_mobius.ipynb`](../notebooks/dmercator/02d_disk_focus_mobius.ipynb)). **3D:** Poincaré **ball** chart from a **D=3** D-Mercator run, enabled when the page URL hash is **`#3d`**. Rendering uses **Three.js `WebGPURenderer` only** (no WebGL fallback).

## Requirements

- **WebGPU**-capable browser (recent Chrome / Edge, or others with WebGPU enabled).
- **2D mode** — static files under **`public/data/`**:
  - `nodes.parquet` — columns `vertex` (string), `x`, `y` (float32, orthographic disk)
  - `edges.parquet` — columns `src`, `dst` (int32 row indices into `nodes`)
  - optional `meta.json` — `default_focus`, `default_seeds` for first load
- **3D mode (`…/index.html#3d`)** — separate bundle under **`public/data3d/`** (same `edges.parquet` schema; `nodes.parquet` includes `x`, `y`, `z` and four `inf_pos_*` columns from the d3 `inf_coord` export).

Generate **2D** files with the first code cell in **[`notebooks/dmercator/04_export_disk_web_bundle.ipynb`](../notebooks/dmercator/04_export_disk_web_bundle.ipynb)**. Generate **3D** files with the **`export_discterry_public_data_3d("d3")`** cell (writes `public/data3d/`; does not overwrite `public/data/`). Run notebooks with cwd `notebooks/dmercator/` so `dmercator_io` imports.

Or from the repo root:

```bash
python -c "import sys; sys.path.insert(0, 'notebooks/dmercator'); import dmercator_io as dm; dm.export_discterry_public_data_3d('d3')"
```

(adapt path / `cd` so `d-mercator-run/d3/edges_GC.inf_coord` resolves.)

## Commands

```bash
cd viz/discterry
npm install
npm run dev
```

Open the printed local URL (append **`#3d`** for the 3D ball). Edits to seeds/focus apply when you **Apply seeds** or pick focus from the list; the 3D view orbits with drag and zooms with the wheel (`R` / `F` reset / fit camera).

```bash
npm run build
npm run preview
```

## Layout

- **`src/math/`** — disk Möbius map, disk/ball Poincaré geodesics, `R_SAFE` / rim constants  
- **`src/data/loadBundle.ts`** / **`loadBundle3d.ts`** — Parquet → typed arrays (`parquet-wasm` + `apache-arrow`)  
- **`src/model/`** — seed edge filter, `computeScene` / `computeScene3d` (geodesic line buffers)  
- **`src/viz/DiskView.tsx`** / **`BallView3d.tsx`** — WebGPU scenes (disk orthographic vs ball perspective + orbit)

See **`CONCEPT.md`** for a full walkthrough of data and geometry.
