# Pipeline: edgelist → D-Mercator → export → Discterry

This document traces **end-to-end** how a protein–protein interaction (PPI) **edge list** becomes coordinates on disk, how that relates to the **D-Mercator** inference described in the paper below, how **`notebooks/dmercator/04_export_disk_web_bundle.ipynb`** packages data for the web app, and **what mathematics the frontend applies** for visualization (it does **not** re-run the embedding).

**Primary reference (open access):**  
García-Pérez, G., Allard, A., Serrano, M.Á. *et al.* “The D-Mercator method for the multidimensional hyperbolic embedding of real networks.” *Nat Commun* **14**, 7193 (2023). DOI [10.1038/s41467-023-43337-5](https://doi.org/10.1038/s41467-023-43337-5) — article: [nature.com/articles/s41467-023-43337-5](https://www.nature.com/articles/s41467-023-43337-5), PDF: [s41467-023-43337-5.pdf](https://www.nature.com/articles/s41467-023-43337-5.pdf).  
You may also have a local copy (e.g. `c:\Users\john\Downloads\s41467-023-43337-5.pdf` on Windows).

---

## 1. What problem D-Mercator solves (paper, in one pass)

Real networks often show **heterogeneous degrees**, **hierarchy**, and **metric structure** that low-dimensional random-geometric models capture well in **hyperbolic** space. Classic **Mercator** (and related **PS** models) embeds in the **2D hyperbolic disk** (one radial popularity-like coordinate and an angular coordinate). **D-Mercator** generalizes this to **native dimension** \(D\): each vertex carries a **hyperbolic radial** coordinate, a **\(D\)-dimensional similarity direction** (a point on a **\(D\)-sphere**), and a **popularity / curvature parameter** \(\kappa\) that shapes connection probabilities in the model. Inference fits these latent variables to the observed graph (which edges exist) under the chosen statistical model.

**Why hyperbolic (intuition):** in hyperbolic space, volume grows exponentially with radius, matching the exponential growth of branches in many real networks; “similarity” and “popularity” can be separated geometrically. **Why multidimensional:** some systems need more than one angular degree of freedom to explain connectivity beyond a single hierarchy axis; D-Mercator estimates that structure rather than forcing everything into a single 2D hyperbolic plane.

**How (at the level of artifacts):** the standalone D-Mercator program (not shipped in this repo) reads an **edgelist**, runs the optimization / sampling procedure described in the paper, and writes text/binary outputs. This repository **consumes** those outputs from a run folder such as `d-mercator-run/d2/`.

---

## 2. Files produced by a D-Mercator run (inputs to this repo)

Under `d-mercator-run/<run>/` the layout used here is:

| File | Role |
|------|------|
| **`edges_GC.edge`** | Plain **undirected edgelist**: two whitespace-separated vertex IDs per line (comments `#` allowed). This is the **topology** the model was fit to. |
| **`edges_GC.inf_coord`** | **Inferred coordinates and parameters** per vertex, plus a **header** with global hyperparameters (`beta`, `mu`, vertex count, radii in model space, etc.). |
| **`edges_GC.inf_log`** | Optional human-readable log from the run. |

The **`*.inf_coord`** body columns (after the header) are the bridge from the paper’s latent variables to our notebooks:

| Column | Meaning (paper-aligned) |
|--------|-------------------------|
| **`Vertex`** | Vertex name (e.g. gene symbol). |
| **`Inf.Kappa`** | Inferred **\(\kappa\)** (popularity / composite curvature parameter in the PS-type model family). High \(\kappa\) typically pushes the vertex toward the **bulk** of the disk in native coordinates; the paper discusses interpretation in terms of connectivity and model fit. |
| **`Inf.Hyp.Rad`** | Inferred **hyperbolic radial** coordinate in the native \((D{+}1)\)-dimensional hyperbolic representation (the “radial” part of the position along the hyperboloid). |
| **`Inf.Pos.1` … `Inf.Pos.(D+1)`** | Inferred **similarity direction** in \(\mathbb{R}^{D+1}\); for **\(D=2\)** (as in a typical run header `DIMENSION 2`) there are **three** components. Before use in maps, directions are treated as **unnormalized 3-vectors** that the code **normalizes to the unit sphere** \(S^2\) (see `dmercator_io.normalize_pos`). |

The header also records **`beta`**, **`mu`**, and model radii (`radius_S^D`, `radius_H^D+1`, …): these are **global** parameters of the fitted **\(S^1 \times \mathbb{H}^2\)**-type geometry (two-dimensional similarity circle times hyperbolic radial line), not per-vertex columns in the table.

**Important:** the inference is **full 3D direction + hyperbolic radius + \(\kappa\)**. The **web export** in this repo (next section) uses only a **2D disk chart** derived from the **angular** part; **`Inf.Hyp.Rad` is not written into `nodes.parquet`**. Downstream analyses that need the full native coordinates should read `inf_coord` or `merged.parquet` from notebook `00`, not only the Discterry bundle.

---

## 3. From `inf_coord` to disk \((x,y)\) in Python (`dmercator_io`)

Module: `notebooks/dmercator/dmercator_io.py`.

1. **`parse_inf_coord`** reads the file, extracts **`meta`** (regex on comments: `nb_vertices`, `beta`, `mu`, `dimension`), and builds a **`DataFrame`** with the vertex table above.

2. **`normalize_pos(df)`** takes `Inf.Pos.1..3`, stacks them as an \(N\times 3\) matrix, **L2-normalizes each row** to unit length. Each vertex is now a point **\(\hat{\mathbf{s}} \in S^2\)** (unit sphere in similarity space).

3. **`ortho_xy_disk(df)`** (used by the export notebook) returns  
   \[
   x = \hat{s}_1,\quad y = \hat{s}_2
   \]  
   i.e. the **first two components** of the normalized direction. Because \(\hat{s}_1^2+\hat{s}_2^2+\hat{s}_3^2=1\), necessarily \(\hat{s}_1^2+\hat{s}_2^2 \le 1\), so **\((x,y)\) lies in the closed unit disk** \(\overline{\mathbb{D}}\). In practice inferred directions yield **strict** interior points suitable for the Poincaré disk visualization.

**Why this projection:** Discterry and the notebooks share a **single 2D complex coordinate** \(z=x+iy\) per vertex so that **Möbius automorphisms** of the **Poincaré disk** (focus changes, true geodesics) are implemented in TypeScript without carrying a third display coordinate. The **price** is that **radial hyperbolic distance** from the native model is **not** encoded in `x,y` alone—those scalars are a **similarity-subspace chart**, not a literal plot of \((r_{\mathbb{H}},\theta)\) from the paper’s hyperboloid. For PPI **navigation and neighborhood structure** in the viewer, this chart is still useful when paired with **hyperbolic geodesics in the disk** (see §6).

**Alternatives in the same module (not used for Discterry export):** `stereo_disk_north` / `stereo_disk_south` stereographically project the sphere from a pole onto \(\mathbb{R}^2\); different charts change the 2D picture but not the raw inference.

---

## 4. Export notebook: `04_export_disk_web_bundle.ipynb`

**Working directory:** `notebooks/dmercator/` (so `import dmercator_io as dm` resolves).

**Steps (mirrors the code cells):**

1. **`RUN_SUBDIR`** — folder under `d-mercator-run/` (default `d2`, or env `DMERCATOR_RUN`).
2. **`paths = dm.paths_for_run(RUN_SUBDIR)`** — resolves `edges_GC.inf_coord` and `edges_GC.edge`.
3. **`parse_inf_coord`** → `meta`, `df`.
4. **`load_edges_graph`** — `networkx.Graph` from the edgelist.
5. **`ortho_xy_disk(df)`** → NumPy arrays `x`, `y` (`float32` in Parquet).
6. **Vertices table** — `vertex` strings in table order; stable index \(0..N-1\).
7. **Edges table** — for each undirected edge \((u,v)\) in the graph, map names to indices **`src`**, **`dst`**; skip edges if an endpoint is missing from the coordinate table (defensive).
8. **Write** with PyArrow:
   - `viz/discterry/public/data/nodes.parquet` — columns `vertex`, `x`, `y`.
   - `viz/discterry/public/data/edges.parquet` — columns `src`, `dst` (`int32`).
   - `viz/discterry/public/data/meta.json` — `run_subdir`, counts, **`default_focus`** / **`default_seeds`** for first paint in the UI.

**What the frontend never sees from this step:** `Inf.Kappa`, `Inf.Hyp.Rad`, and the third normalized component \(\hat{s}_3\) (it is fixed by \(x,y\) on the sphere only up to sign ambiguity—in practice the inferred vector determines the sign). If you need \(\kappa\) or native radius in the browser, extend the Parquet schema and loader accordingly.

---

## 5. Discterry frontend: load → scene → GPU

Code lives under `viz/discterry/` (see also `viz/discterry/CONCEPT.md` for a viewer-centric glossary).

### 5.1 Load (`src/data/loadBundle.ts`)

Fetches **`nodes.parquet`**, **`edges.parquet`**, optional **`meta.json`**. Builds **`GraphBundle`**: parallel `Float32Array` for `x`, `y`, edge `src`/`dst`, `nameToIndex`, and `vertex[]` names.

### 5.2 Focus: Möbius recentering (`src/z0FromProtein.ts`, `src/math/mobius.ts`)

User picks a **focus** protein \(F\). Its stored coordinate is \(z_0^{\text{raw}} = x[j] + i y[j]\). **`clampZ0`** slightly scales \(z_0\) inward if \(|z_0|\) is extremely close to 1 (avoids a vanishing denominator in the map).

The **disk automorphism** (same formula as in Poincaré models and in notebook `02d_disk_focus_mobius.ipynb`):

\[
T_{z_0}(z) = \frac{z - z_0}{1 - \overline{z_0}\, z}.
\]

**What:** sends \(z_0 \mapsto 0\) and preserves the Poincaré metric as an isometry of the disk.  
**Why:** recenter the plot on a chosen gene without recomputing the embedding.  
**How:** `mobiusZ` / `mobiusDiskArrays` applied to every vertex to produce \((w_x, w_y)\) in the **\(W\)**-plane.

### 5.3 Scene assembly (`src/model/computeScene.ts`, `src/model/graphFilter.ts`)

- **Seeds** — user-supplied name set; determines which vertices are highlighted and which edges are drawn.
- **Edges** — only **seed-touching** edges are triangulated as polylines:
  - **Both endpoints seeds** → “both-seed” list (emphasized).
  - **Exactly one seed** → “one-seed” list (faint).
- For each drawn edge, **`appendGeodesicLineSegments`**:
  1. Reads endpoints in **original** \(z\)-space from the bundle.
  2. Maps endpoints to **\(W\)** with the **same** \(z_0\).
  3. Samples **`poincareGeodesicXY`** between \(w_1\) and \(w_2\) in the **Poincaré disk** (`src/math/poincareGeodesic.ts` — upper half-plane route, same mathematics as the reference notebook).

**Why geodesics, not straight chords:** in the disk model, **hyperbolic straight lines** are **circular arcs** orthogonal to the boundary (or diameters). Drawing Euclidean segments would be geometrically misleading for “distance” in the model.

**Rim cull** — display-only: vertices with \(|W|\) in an outer annulus can be hidden unless they are seeds or touch a drawn edge (`rimCullEps` slider).  
**Boundary clip** — edges whose endpoints have \(|z|\) or \(|W|\) beyond `EDGE_Z_BOUND` (~0.999) skip geodesic construction for numerical stability.

### 5.4 Focus animation (`App.tsx`, `src/math/focusAnimPath.ts`)

When changing focus, **`z0`** is interpolated along a **true hyperbolic geodesic** between the old and new focus positions **in the disk** (not a Euclidean straight line in \((\mathrm{Re}\,z,\mathrm{Im}\,z)\)). Each frame calls **`computeScene`** with the intermediate \(z_0\) and pushes buffers via **`applySceneBuffers`**.

### 5.5 Viewer-only transforms (`src/viz/DiskView.tsx`, `src/math/viewerMobius.ts`)

These **do not** change the stored graph; they only change how the same buffers are viewed:

| Control | Mathematics |
|--------|-------------|
| **Wheel** | Orthographic **zoom** (camera half-extent). |
| **Shift + wheel** | **Radial display warp** \(w \mapsto |w|^{\gamma-1} w\) with \(|w|=1\) fixed (rim-preserving “FOV-like” stretch of the interior). |
| **Drag** | Euclidean **pan** of the orthographic camera. |
| **Shift + drag** | Left-composition with a small **disk automorphism** in **SU(1,1)** form (incremental **Möbius** “hyperbolic pan”). |

**Recenter on focus change** clears **pan** and **viewer Möbius** (not orthographic zoom / not rim \(\gamma\)) so that after a new \(z_0\), the origin of \(W\) lies at the screen center—consistent with “focus = data origin.”

### 5.6 Drawing order (WebGPU)

Roughly: grid/crosshair → faint blue one-seed geodesics → green non-seed nodes → red both-seed geodesics (thick line material) → red seed disks on top; optional HTML labels projected with the same camera.

---

## 6. Mental model checklist

| Stage | Where | Coordinates |
|-------|--------|----------------|
| Topology | `*.edge` | Graph \(G=(V,E)\) |
| Inference | D-Mercator binary (external) | \((\kappa, r_{\mathbb{H}}, \hat{\mathbf{s}} \in S^2)\) per vertex + globals \(\beta,\mu,\ldots\) |
| 2D chart | `ortho_xy_disk` | \(z = x + iy\) from first two components of \(\hat{\mathbf{s}}\) |
| Web bundle | Parquet | Same \(x,y\) + indexed edges |
| Focus view | `mobiusZ` | \(W = T_{z_0}(z)\) |
| Edges in view | `poincareGeodesic` | Arcs in the \(W\)-disk |

---

## 7. Related paths in this repo

| Piece | Path |
|-------|------|
| IO + charts | `notebooks/dmercator/dmercator_io.py` |
| Export notebook | `notebooks/dmercator/04_export_disk_web_bundle.ipynb` |
| Notebook index | `notebooks/dmercator/README.md` |
| Disk / Möbius math in Python | `notebooks/dmercator/02d_disk_focus_mobius.ipynb` (and `02_disk_views.ipynb`) |
| Viewer concepts | `viz/discterry/CONCEPT.md` |
| Viewer implementation | `viz/discterry/src/` |
