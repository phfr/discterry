"""IO helpers for D-Mercator PPI outputs under ``d-mercator-run/<run>/``."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def repo_root() -> Path:
    """Repository root (``hbol``): ``notebooks/dmercator`` → ``notebooks`` → root."""
    return Path(__file__).resolve().parent.parent.parent


# ``viz/discterry`` defaults — keep in sync with ``02d_disk_focus_mobius.ipynb`` (``FOCUS_NODE`` / ``SEED_NODES``).
DEFAULT_DISCTERRY_FOCUS = "STAC3"
DEFAULT_DISCTERRY_SEEDS: tuple[str, ...] = (
    "CATSPER1",
    "CYSRT1",
    "CATSPERD",
    "CACNA1H",
    "STAC3",
    "CACNB3",
    "CACNG6",
    "CACNA1S",
    "CATSPER4",
    "CACNG8",
    "CACNG2",
    "CACNA1C",
    "KRTAP1-3",
    "NOTCH2NLA",
    "CACNG1",
    "CACNA1F",
    "CACNA2D1",
)


def get_run_dir(subdir: Optional[str] = None) -> Path:
    """Return ``<repo>/d-mercator-run/<subdir>``.

    ``subdir`` defaults to ``DMERCATOR_RUN`` env or ``"d2"``.
    """
    s = (subdir or os.environ.get("DMERCATOR_RUN", "d2")).strip()
    return repo_root() / "d-mercator-run" / s


def paths_for_run(subdir: Optional[str] = None) -> Dict[str, Path]:
    """Paths for the standard ``edges_GC`` rootname under a run directory."""
    base = get_run_dir(subdir)
    rootname = "edges_GC"
    return {
        "inf_coord": base / f"{rootname}.inf_coord",
        "edge": base / f"{rootname}.edge",
        "inf_log": base / f"{rootname}.inf_log",
    }


def _parse_coord_meta(raw: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    m = re.search(r"nb\.\s*vertices:\s*(\d+)", raw)
    if m:
        meta["nb_vertices"] = int(m.group(1))
    m = re.search(r"beta:\s*([0-9.eE+-]+)", raw, re.IGNORECASE)
    if m:
        meta["beta"] = float(m.group(1))
    m = re.search(r"mu:\s*([0-9.eE+-]+)", raw, re.IGNORECASE)
    if m:
        meta["mu"] = float(m.group(1))
    m = re.search(r"DIMENSION\s+(\d+)", raw)
    if m:
        meta["dimension"] = int(m.group(1))
    m = re.search(r"radius_S\^D:\s*([0-9.eE+-]+)", raw)
    if m:
        meta["radius_s_d"] = float(m.group(1))
    m = re.search(r"radius_H\^D\+1\s+([0-9.eE+-]+)", raw)
    if m:
        meta["radius_h_d1"] = float(m.group(1))
    m = re.search(r"kappa_min:\s*([0-9.eE+-]+)", raw, re.IGNORECASE)
    if m:
        meta["kappa_min"] = float(m.group(1))
    return meta


def parse_inf_coord(path: Path | str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Parse ``*.inf_coord``: comment header/footer and ``Vertex`` table.

    Returns ``(meta, df)`` with columns
    ``Vertex``, ``Inf.Kappa``, ``Inf.Hyp.Rad``, ``Inf.Pos.1``, ``Inf.Pos.2``, ``Inf.Pos.3``.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta = _parse_coord_meta(raw)
    meta["inf_coord_path"] = str(path.resolve())
    lines = raw.splitlines()
    i0: Optional[int] = None
    for i, line in enumerate(lines):
        if "Vertex" in line and "Inf.Kappa" in line:
            i0 = i + 1
            break
    if i0 is None:
        raise ValueError(f"No Vertex / Inf.Kappa header in {path}")

    rows: list[tuple[Any, ...]] = []
    for j in range(i0, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            break
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            kappa = float(parts[-5])
            hyp_rad = float(parts[-4])
            p1, p2, p3 = float(parts[-3]), float(parts[-2]), float(parts[-1])
        except ValueError:
            continue
        name = " ".join(parts[:-5]).strip()
        if not name:
            continue
        rows.append((name, kappa, hyp_rad, p1, p2, p3))

    df = pd.DataFrame(
        rows,
        columns=["Vertex", "Inf.Kappa", "Inf.Hyp.Rad", "Inf.Pos.1", "Inf.Pos.2", "Inf.Pos.3"],
    )
    return meta, df


def load_edges_graph(edge_path: Path | str) -> nx.Graph:
    """Load whitespace-separated edgelist (two protein IDs per line) into an undirected graph."""
    edge_path = Path(edge_path)
    G = nx.Graph()
    with edge_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            G.add_edge(parts[0], parts[1])
    return G


def normalize_pos(df: pd.DataFrame) -> np.ndarray:
    """Unit directions (N×3) from ``Inf.Pos.*`` columns."""
    cols = ["Inf.Pos.1", "Inf.Pos.2", "Inf.Pos.3"]
    p = df[cols].to_numpy(dtype=np.float64)
    n = np.linalg.norm(p, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return p / n


def stereo_disk_north(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Stereographic projection from north pole ``(0,0,1)`` onto ``z=0`` plane."""
    p = normalize_pos(df)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    eps = 1e-12
    denom = np.maximum(eps, 1.0 - z)
    u = x / denom
    v = y / denom
    return u, v


def stereo_disk_south(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Stereographic projection from south pole ``(0,0,-1)`` onto ``z=0`` plane."""
    p = normalize_pos(df)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    eps = 1e-12
    denom = np.maximum(eps, 1.0 + z)
    u = x / denom
    v = y / denom
    return u, v


def ortho_xy_disk(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Drop the third coordinate after normalizing to the unit sphere (points lie in the unit disk)."""
    p = normalize_pos(df)
    return p[:, 0].copy(), p[:, 1].copy()


def load_merged_parquet(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def save_merged_parquet(df: pd.DataFrame, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def export_discterry_public_data(subdir: Optional[str] = None) -> Dict[str, Any]:
    """Write ``viz/discterry/public/data/{nodes,edges}.parquet`` and ``meta.json``.

    ``nodes.parquet`` includes disk chart ``x,y`` plus all per-vertex D-Mercator inference
    columns from ``*.inf_coord`` (``inf_kappa``, ``inf_hyp_rad``, ``inf_pos_*``).
    ``meta.json`` merges run defaults with header fields from the coord file (``beta``,
    ``mu``, ``dimension``, radii, etc.).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = paths_for_run(subdir)
    coord_meta, df = parse_inf_coord(paths["inf_coord"])
    g = load_edges_graph(paths["edge"])
    x, y = ortho_xy_disk(df)

    repo = repo_root()
    out = repo / "viz" / "discterry" / "public" / "data"
    out.mkdir(parents=True, exist_ok=True)

    vertices = [str(v).strip() for v in df["Vertex"].tolist()]
    name_to_i = {name: i for i, name in enumerate(vertices)}

    kappa = df["Inf.Kappa"].to_numpy(dtype=np.float64).astype(np.float32)
    hyp = df["Inf.Hyp.Rad"].to_numpy(dtype=np.float64).astype(np.float32)
    p1 = df["Inf.Pos.1"].to_numpy(dtype=np.float64).astype(np.float32)
    p2 = df["Inf.Pos.2"].to_numpy(dtype=np.float64).astype(np.float32)
    p3 = df["Inf.Pos.3"].to_numpy(dtype=np.float64).astype(np.float32)

    nodes_tbl = pa.table(
        {
            "vertex": pa.array(vertices, type=pa.string()),
            "x": pa.array(np.asarray(x, dtype=np.float32), type=pa.float32()),
            "y": pa.array(np.asarray(y, dtype=np.float32), type=pa.float32()),
            "inf_kappa": pa.array(kappa, type=pa.float32()),
            "inf_hyp_rad": pa.array(hyp, type=pa.float32()),
            "inf_pos_1": pa.array(p1, type=pa.float32()),
            "inf_pos_2": pa.array(p2, type=pa.float32()),
            "inf_pos_3": pa.array(p3, type=pa.float32()),
        }
    )

    src: list[int] = []
    dst: list[int] = []
    for u, v in g.edges():
        uu, vv = str(u).strip(), str(v).strip()
        if uu not in name_to_i or vv not in name_to_i:
            continue
        src.append(name_to_i[uu])
        dst.append(name_to_i[vv])

    edges_tbl = pa.table(
        {
            "src": pa.array(src, type=pa.int32()),
            "dst": pa.array(dst, type=pa.int32()),
        }
    )

    pq.write_table(nodes_tbl, out / "nodes.parquet")
    pq.write_table(edges_tbl, out / "edges.parquet")

    run_key = (subdir or os.environ.get("DMERCATOR_RUN", "d2")).strip()
    focus = (
        DEFAULT_DISCTERRY_FOCUS.strip()
        if DEFAULT_DISCTERRY_FOCUS.strip() in name_to_i
        else (vertices[0] if vertices else "")
    )
    default_seeds = [s.strip() for s in DEFAULT_DISCTERRY_SEEDS if s.strip() in name_to_i]
    if not default_seeds and vertices:
        default_seeds = [vertices[0]]

    meta: Dict[str, Any] = {
        "run_subdir": run_key,
        "n_vertices": len(vertices),
        "n_edges": len(src),
        "default_focus": focus,
        "default_seeds": default_seeds,
    }
    for k, v in coord_meta.items():
        if k == "nb_vertices":
            meta["mercator_nb_vertices_header"] = v
        else:
            meta[k] = v

    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote nodes:", out / "nodes.parquet", "n_vertices=", len(vertices))
    print("Wrote edges:", out / "edges.parquet", "n_edges=", len(src))
    print("Wrote meta:", out / "meta.json")
    return meta
