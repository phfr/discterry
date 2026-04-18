# Concepts: data, geometry, and rendering in **Discterry**

This document explains **what Discterry does**, from the Parquet files on disk to the pixels on screen. It is written for readers who may not know **hyperbolic geometry** or **complex analysis**; every technical term is introduced when it first appears.

---

## 1. What problem this viewer solves

An upstream **PPI disk embedding** places a **network** (genes or proteins as **vertices**, interactions as **edges**) into a **round diagram**: each vertex gets a 2D position inside the **unit disk** (the interior of a circle of radius 1). **Discterry** does **not** re-run that embedding; it **reads** those positions and the edge list, then lets you:

- Pick a **focus** vertex (a protein of interest).
- Pick a **seed** set (a list of vertices you care about).
- **Re-center** the picture so the focus sits at the **middle of the disk** in a geometry-aware way.
- Draw **only** vertices and edges that involve seeds (plus the focus), so the picture stays readable on large graphs.

So: **Parquet = stored geometry + topology**; **browser = Möbius transform + filtering + drawing**.

---

## 2. Files on disk and what each column means

### 2.1 `public/data/nodes.parquet`

One row per vertex. The app loads **three** columns (see `src/data/loadBundle.ts`):

| Column   | Type   | Meaning |
|----------|--------|---------|
| `vertex` | string | Human-readable name (e.g. gene symbol). Must match what you type for focus/seeds. |
| `x`      | float  | Real part of the vertex position in the **orthographic disk model** (see §4). |
| `y`      | float  | Imaginary part of that same position. |

Together, \((x, y)\) is a point in the **open** unit disk: strictly inside the unit circle. In code we often write **\(z = x + i y\)** as a **complex number**—only a bookkeeping trick so one pair \((x,y)\) is one number \(z\) with **real part** \(x\) and **imaginary part** \(y\). No physics is implied.

The loader builds:

- `vertex[]` — names in row order.
- `x[]`, `y[]` — parallel `Float32Array`s of coordinates.
- `nameToIndex` — map from trimmed string name → row index (used for seeds, focus, and tooltips).

### 2.2 `public/data/edges.parquet`

One row per undirected edge, as **indices** into `nodes` (same order as `vertex`):

| Column | Type  | Meaning |
|--------|-------|---------|
| `src`  | int32 | Index of one endpoint (0 … N−1). |
| `dst`  | int32 | Index of the other endpoint. |

The app does **not** assume a particular ordering of endpoints; an edge \((i,j)\) is the same as \((j,i)\) for drawing.

### 2.3 `public/data/meta.json` (optional)

Small JSON for **defaults** on first load: `default_focus` (string), `default_seeds` (array of strings). Loaded separately from the graph (`src/data/loadBundle.ts`).

---

## 3. Graph terms (discrete math, not hyperbolic)

- **Vertex / node** — one row in `nodes.parquet`; identified by **index** \(i\) and **name** `vertex[i]`.
- **Edge** — one row in `edges.parquet`; connects `src[ei]` and `dst[ei]`.
- **Degree** — number of edges incident to a vertex (each edge counted once). Shown in the UI list; computed by scanning all edges once.
- **Seed** — a vertex whose name appears in the **applied** seed list (after you apply the text box). Stored as a **mask** `isSeed[i] ∈ {0,1}`.
- **Seed-touching edge** — an edge with **at least one** seed endpoint.
- **Both-seed edge** — both endpoints are seeds (used for **red** emphasis in the viewer).
- **One-seed edge** — exactly one endpoint is a seed (drawn as **blue**, faint additive lines).

---

## 4. The unit disk as a “sheet” for the network

Imagine the open unit disk as a **flat round sheet** (radius 1). Every vertex has coordinates \((x,y)\) with \(x^2 + y^2 < 1\). That is the **orthographic disk** picture produced upstream (export notebook + `dmercator_io` under `notebooks/dmercator/`); **Discterry** **trusts** those numbers.

**Why a disk?** The Poincaré disk is a standard **model of hyperbolic space**: distances inside the disk are **not** Euclidean distances; “straight lines” in that world are **circular arcs** orthogonal to the boundary circle (or diameters). Discterry draws those **true hyperbolic geodesics** between mapped endpoints (§10–§11), not straight Euclidean chords in the disk—except when a geodesic happens to be a diameter segment, it **looks** straight.

You do **not** need the full theory: treat “geodesic” as **the correct curved connector** for this geometry, implemented by the code in `src/math/poincareGeodesic.ts` (same idea as `02d_disk_focus_mobius.ipynb`).

---

## 5. Complex numbers in one minute

A **complex number** \(z = x + i y\) is a pair \((x,y)\) with a multiplication rule. For this app:

- **Addition** — add \(x\) and add \(y\) separately (like vectors).
- **Complex conjugate** of \(z = x+iy\) is \(\bar z = x - iy\) (flip the sign of \(y\)).
- **Modulus** \(|z| = \sqrt{x^2+y^2}\) is the **distance from the origin** in the usual Euclidean plane.

The **Möbius map** (§6) is written with complex arithmetic; the TypeScript implementation expands it into real operations on `re` / `im` (`src/math/mobius.ts`).

---

## 6. Focus and the Möbius automorphism (“hyperbolic recentering”)

### 6.1 Choosing the focus parameter \(z_0\)

When you pick **focus** name \(F\):

1. Look up index \(j\) with `nameToIndex`.
2. Read \(z_0^\text{raw} = (x[j], y[j])\) from the bundle.
3. Apply **`clampZ0`** (`R_SAFE` in `src/math/constants.ts`): if \(|z_0^\text{raw}|\) is **extremely** close to 1, scale the vector slightly inward so the later formula stays numerically stable. For typical data the clamp is a **no-op**; it only fires for pathological rim points.

The result is **`z0`** passed everywhere as `Complex { re, im }` (`z0FromProtein`).

### 6.2 The map \(T_{z_0}(z)\)

For each vertex coordinate \(z\) (same \((x,y)\) as in the file), the app computes

\[
w = T_{z_0}(z) = \frac{z - z_0}{1 - \overline{z_0}\, z}.
\]

This is a **disk automorphism**: it maps the open unit disk to itself, is **invertible**, preserves **hyperbolic** structure, and sends **\(z_0 \mapsto 0\)**. So after the map:

- The **focus protein** sits at the **origin** \((0,0)\)—the crosshair center in the plot.
- All other vertices move to new positions \(w\) inside the same unit disk.

Implementation: `mobiusZ` / `mobiusDiskArrays` in `src/math/mobius.ts`.

**Intuition:** you are choosing a **legal change of coordinates** on the sheet that **puts your chosen gene at the center** without tearing the disk; hyperbolic “straight lines” stay geodesics after the map.

---

## 7. After the map: coordinates \(W\)

Write \(W = u + i v\) for the image point (the code stores `wx[i]`, `wy[i]`). These are still inside the unit disk for all vertices coming from a valid embedding.

**Node drawing** uses \((wx, wy)\) (with a tiny \(z\) offset for seeds in WebGL—see rendering §12).

---

## 8. Seeds and which edges are drawn

Given the **applied** seed name set:

1. **`buildSeedMask`** — `isSeed[i]=1` if `vertex[i]` is in the set (exact string match after trim).
2. **`classifySeedEdges`** — scans every edge:
   - both endpoints seeds → edge index goes to list **`both`**;
   - exactly one → list **`one`**;
   - neither → ignored for **line** drawing (still in the graph for degree, not drawn as a colored geodesic here).

So **edges drawn** are only **seed-touching** edges. Non-seed vertices can still appear as **green points** if they are endpoints of such edges (see §9 rim logic).

---

## 9. Rim cull (display-only, not a change to the math)

**Rimcull** slider sets `rimCullEps` (default from `RIM_CULL_EPS`). After Möbius, define

\[
\text{rim} = 1 - \max(0, \texttt{rimCullEps}).
\]

A vertex is considered **in the outer band** if \(|W| > \text{rim}\) (Euclidean modulus in the \(W\)-plane).

- **Hidden** if it is in that band **and** not a seed **and** not an endpoint of any **seed-touching** edge.
- **Seeds** and **vertices on seed-touching edges** are **always drawn** even in the band (so the picture does not lose the subgraph you care about).

This is **purely visual clutter control**; it does not change the underlying \(z\) or \(W\) values used when an object **is** drawn.

---

## 10. Boundary clip for **edge** polylines (`EDGE_Z_BOUND`)

Separate from rim cull: before building a geodesic polyline, the code checks endpoints in **original** \(z\)-space and in **mapped** \(w\)-space. If either endpoint has modulus **≥ `EDGE_Z_BOUND`** (0.999), that edge’s geodesic is **skipped** (see `appendGeodesicLineSegments` in `src/model/computeScene.ts`).

**Why:** very near the unit circle, the half-plane construction and projective steps are **numerically fragile**; skipping matches the spirit of the reference notebook clip.

**Stats:** `edgesSeedTouching` counts both+one lists; `edgesDrawn` counts edges that actually produced polyline segments; **`edgesSkippedBoundary`** = difference (shown in the UI as “edges skip clip”).

---

## 11. Building the scene: `computeScene`

`computeScene(bundle, z0, seedNames, rimCullEps)` returns **`SceneBuffers`**:

1. **Masks** — `isSeed`, `both`, `one`, and `seedEdgeVertex` (vertex on at least one seed-touching edge).
2. **Map all vertices** — fill `wx`, `wy` via `mobiusDiskArrays`.
3. **Points** — for each vertex, decide `show` from rim rule (§9). If shown:
   - if seed → append \((wx, wy, 0.004)\) to **seed** buffer + name to **`seedLabels`**;
   - else → append \((wx, wy, 0)\) to **other** buffer.
4. **Lines** — for each edge index in `both` and `one`, call `appendGeodesicLineSegments`:
   - Read endpoints \(z_1, z_2\) in **original** disk from `x,y`.
   - Map to \(w_1, w_2\) with the **same** `z0`.
   - Run **`poincareGeodesicXY`** on \((w_1, w_2)\) to get `GEODESIC_N` samples along the **true** Poincaré geodesic in the \(W\)-disk.
   - Emit **line segment** pairs \((w_k, w_{k+1})\) as 3D points \((x,y,0)\) for the GPU line list.

Constants: `GEODESIC_N` samples per geodesic; `EDGE_Z_BOUND` for skip; `R_SAFE` only affects `z0` via `clampZ0`.

---

## 12. Rendering pipeline (`DiskView.tsx` + Three.js WebGPU)

The UI passes **`SceneBuffers`** into `DiskView`. A **WebGPU** renderer draws a fixed orthographic camera on the \((x,y)\) plane (disk fits the view).

Typical **draw order** (conceptually bottom → top):

1. **Optional crosshair** — two faint `LineSegments` on the axes (toggle in UI).
2. **Blue** — one-seed edges (`LineSegments`, additive blending, low opacity).
3. **Green points** — non-seed vertices in the scene buffer (`Points`, `depthWrite: false` so large sprites do not erase neighbors).
4. **Red thick lines** — both-seed edges via **`LineSegments2`** + **`Line2NodeMaterial`** (wide lines; `linewidth` in **pixels**).
5. **Red seed markers** — instanced **filled circles** (`InstancedMesh` + `CircleGeometry`), drawn with **no depth test** so they always read on top.
6. **HTML overlay** (optional) — seed **name labels** at projected screen positions from the same \((wx, wy)\).

The canvas lives under a full-viewport host; labels sit in a `pointer-events: none` div aligned to the canvas.

---

## 13. UI flow (how pieces connect)

1. **Load** — fetch Parquet + optional `meta.json`; build `GraphBundle`.
2. **Seeds text** — draft vs **applied** list (Apply button validates names exist).
3. **Focus** — clicking a name sets **applied focus**; `z0FromProtein` recomputes `z0`.
4. **`useMemo`** — recomputes `computeScene` whenever bundle, focus, applied seeds, or rim slider change.
5. **Stats box** — reads `SceneBuffers.stats` (nodes rendered, rim hides, edge counts, skip clip).

Tooltips on the name list show **loaded scalars** for that row: name, index, \(x,y,|z|,\) angle, degree.

---

## 14. Glossary (quick reference)

| Term | Short meaning |
|------|----------------|
| **Unit disk** | Points with \(x^2+y^2 < 1\); the plot window. |
| **Orthographic disk \((x,y)\)** | Coordinates stored in Parquet; the embedding’s output before any focus map. |
| **Complex \(z\)** | Pair \((x,y)\) with fixed multiplication rules; not physical. |
| **Focus \(z_0\)** | The chosen vertex’s disk position (after `clampZ0`); center of the Möbius map. |
| **\(W = T_{z_0}(z)\)** | Coordinates after recentering; focus at origin. |
| **Geodesic** | Shortest path in hyperbolic sense; drawn as a circular arc (or diameter) in the disk. |
| **Seed** | Highlighted vertex set; determines which edges and neighbors appear. |
| **Rimcull** | Hides far-out non-seed vertices (in \(W\)) unless they touch a drawn seed edge. |
| **Skip clip** | Edges skipped because an endpoint is too close to the unit circle numerically. |

---

## 15. Where to read the code

| Topic | Location |
|-------|-----------|
| Parquet → arrays | `src/data/loadBundle.ts` |
| Focus → `z0` | `src/z0FromProtein.ts`, `src/math/mobius.ts` (`clampZ0`, `mobiusZ`) |
| Seeds / edge classes | `src/model/graphFilter.ts` |
| Scene assembly | `src/model/computeScene.ts` |
| Geodesic math | `src/math/poincareGeodesic.ts` |
| Constants | `src/math/constants.ts` |
| GPU drawing | `src/viz/DiskView.tsx` |
| Controls & stats | `src/App.tsx` |

The authoritative narrative for the **notebook** side of the same mathematics is `notebooks/dmercator/02d_disk_focus_mobius.ipynb` (markdown cells + code).
