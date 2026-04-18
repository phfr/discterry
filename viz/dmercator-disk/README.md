# D-Mercator Möbius disk (WebGPU)

Interactive **Poincaré disk** view of a D-Mercator run: choose a **focus** protein and **seed** proteins (same logic as [`notebooks/dmercator/02d_disk_focus_mobius.ipynb`](../notebooks/dmercator/02d_disk_focus_mobius.ipynb)). Rendering uses **Three.js `WebGPURenderer` only** (no WebGL fallback).

## Requirements

- **WebGPU**-capable browser (recent Chrome / Edge, or others with WebGPU enabled).
- Static data files under **`public/data/`**:
  - `nodes.parquet` — columns `vertex` (string), `x`, `y` (float32, orthographic disk)
  - `edges.parquet` — columns `src`, `dst` (int32 row indices into `nodes`)
  - optional `meta.json` — `default_focus`, `default_seeds` for first load

Generate these with **[`notebooks/dmercator/04_export_disk_web_bundle.ipynb`](../notebooks/dmercator/04_export_disk_web_bundle.ipynb)** (run with cwd `notebooks/dmercator/` so `dmercator_io` imports).

## Commands

```bash
cd viz/dmercator-disk
npm install
npm run dev
```

Open the printed local URL. Click **Update view** after editing focus or seeds.

```bash
npm run build
npm run preview
```

## Layout

- **`src/math/`** — Möbius map, Poincaré geodesic sampling, `R_SAFE` / rim constants  
- **`src/data/loadBundle.ts`** — Parquet → typed arrays (`parquet-wasm` + `apache-arrow`)  
- **`src/model/`** — seed edge filter, `computeScene` (geodesic line buffers)  
- **`src/viz/DiskView.tsx`** — WebGPU scene and batched `LineSegments` / `Points`
