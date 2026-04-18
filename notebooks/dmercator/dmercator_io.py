"""IO helpers for D-Mercator PPI outputs under ``d-mercator-run/<run>/``."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _inf_pos_column_count(header_line: str) -> int:
    """Count ``Inf.Pos.N`` tokens in the Vertex table header (D+1 for native D-Mercator)."""
    return len(re.findall(r"Inf\.Pos\.\d+", header_line))


def parse_inf_coord(path: Path | str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Parse ``*.inf_coord``: comment header/footer and ``Vertex`` table.

    Returns ``(meta, df)`` with columns ``Vertex``, ``Inf.Kappa``, ``Inf.Hyp.Rad``,
    and ``Inf.Pos.1`` … ``Inf.Pos.{n}`` where *n* is inferred from the header row
    (3 for D=2, 4 for D=3, etc.).
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta = _parse_coord_meta(raw)
    meta["inf_coord_path"] = str(path.resolve())
    lines = raw.splitlines()
    header_line = ""
    i0: Optional[int] = None
    for i, line in enumerate(lines):
        if "Vertex" in line and "Inf.Kappa" in line:
            header_line = line
            i0 = i + 1
            break
    if i0 is None:
        raise ValueError(f"No Vertex / Inf.Kappa header in {path}")

    n_pos = _inf_pos_column_count(header_line)
    if n_pos < 3:
        raise ValueError(f"Expected at least 3 Inf.Pos columns in header of {path}, got {n_pos}")
    if "dimension" not in meta:
        meta["dimension"] = int(n_pos - 1)

    pos_cols = [f"Inf.Pos.{k}" for k in range(1, n_pos + 1)]
    tail = 2 + n_pos  # kappa, hyp_rad, positions

    rows: list[tuple[Any, ...]] = []
    for j in range(i0, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            break
        parts = line.split()
        if len(parts) < tail + 1:
            continue
        try:
            tail_nums = [float(parts[k]) for k in range(-tail, 0)]
        except ValueError:
            continue
        kappa = tail_nums[0]
        hyp_rad = tail_nums[1]
        pos_vals = tail_nums[2:]
        if len(pos_vals) != n_pos:
            continue
        name = " ".join(parts[: -tail]).strip()
        if not name:
            continue
        rows.append((name, kappa, hyp_rad, *pos_vals))

    df = pd.DataFrame(rows, columns=["Vertex", "Inf.Kappa", "Inf.Hyp.Rad", *pos_cols])
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


def inf_pos_columns(df: pd.DataFrame) -> List[str]:
    """Sorted list of ``Inf.Pos.*`` column names present in ``df``."""
    cols = [c for c in df.columns if re.fullmatch(r"Inf\.Pos\.\d+", c)]
    return sorted(cols, key=lambda c: int(c.split(".")[-1]))


def normalize_pos(df: pd.DataFrame) -> np.ndarray:
    """Unit directions (N×K) from all ``Inf.Pos.*`` columns, L2-normalized per row."""
    cols = inf_pos_columns(df)
    if not cols:
        raise ValueError("no Inf.Pos columns in DataFrame")
    p = df[cols].to_numpy(dtype=np.float64)
    n = np.linalg.norm(p, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return p / n


def stereo_s3_to_r3(p4: np.ndarray) -> np.ndarray:
    """Stereographic image of ``S^3`` (rows unit in R^4) into ``R^3``, pole ``(0,0,0,1)``.

    For row ``(x,y,z,w)``, map ``(x,y,z) / (1 - w + eps)``. Output shape ``(N, 3)``.
    """
    x, y, z, w = p4[:, 0], p4[:, 1], p4[:, 2], p4[:, 3]
    eps = 1e-12
    denom = np.maximum(eps, 1.0 - w)
    return np.stack([x / denom, y / denom, z / denom], axis=1)


def poincare_ball_xyz_from_native(
    df: pd.DataFrame,
    coord_meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tier-A chart: ``S^3`` direction → ``R^3`` via stereographic pole ``(0,0,0,1)``, then radial warp.

    Display coordinates lie in the open unit ball: ``rho * u_hat`` with
    ``rho = 0.92 * tanh(Inf.Hyp.Rad / sigma)``, ``sigma = radius_h_d1`` from header
    when present else median hyperbolic radius.

    This is a **visual** chart for Discterry ``#3d``; geodesics in the viewer use the
    true Poincaré ball metric in these display coordinates, not the native D-Mercator
    hyperboloid distances.
    """
    u = normalize_pos(df)
    if u.shape[1] != 4:
        raise ValueError(f"Tier-A 3d export expects 4 Inf.Pos columns (D=3), got {u.shape[1]}")

    v = stereo_s3_to_r3(u)
    hyp = df["Inf.Hyp.Rad"].to_numpy(dtype=np.float64)
    sigma = float(coord_meta.get("radius_h_d1") or np.median(hyp) or 10.0)
    sigma = max(sigma, 1e-6)
    rho = 0.92 * np.tanh(hyp / sigma)
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    vn = np.maximum(vn, 1e-12)
    uhat = v / vn
    ball = rho.reshape(-1, 1) * uhat
    bx, by, bz = ball[:, 0], ball[:, 1], ball[:, 2]
    return bx.astype(np.float32), by.astype(np.float32), bz.astype(np.float32)


def stereo_disk_north(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Stereographic projection from north pole ``(0,0,1)`` onto ``z=0`` plane."""
    p = normalize_pos(df)
    if p.shape[1] < 3:
        raise ValueError("stereo_disk_north needs at least 3 Inf.Pos columns")
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    eps = 1e-12
    denom = np.maximum(eps, 1.0 - z)
    u = x / denom
    v = y / denom
    return u, v


def stereo_disk_south(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Stereographic projection from south pole ``(0,0,-1)`` onto ``z=0`` plane."""
    p = normalize_pos(df)
    if p.shape[1] < 3:
        raise ValueError("stereo_disk_south needs at least 3 Inf.Pos columns")
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    eps = 1e-12
    denom = np.maximum(eps, 1.0 + z)
    u = x / denom
    v = y / denom
    return u, v


def ortho_xy_disk(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Drop the third coordinate after normalizing to the unit sphere (points lie in the unit disk)."""
    p = normalize_pos(df)
    if p.shape[1] < 3:
        raise ValueError("ortho_xy_disk needs at least 3 Inf.Pos columns")
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
    pos_cols = inf_pos_columns(df)
    pos_arrays = {
        f"inf_pos_{i + 1}": df[c].to_numpy(dtype=np.float64).astype(np.float32) for i, c in enumerate(pos_cols)
    }

    col_dict: Dict[str, Any] = {
        "vertex": pa.array(vertices, type=pa.string()),
        "x": pa.array(np.asarray(x, dtype=np.float32), type=pa.float32()),
        "y": pa.array(np.asarray(y, dtype=np.float32), type=pa.float32()),
        "inf_kappa": pa.array(kappa, type=pa.float32()),
        "inf_hyp_rad": pa.array(hyp, type=pa.float32()),
    }
    col_dict.update(pos_arrays)
    nodes_tbl = pa.table(col_dict)

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


def export_discterry_public_data_3d(subdir: str = "d3") -> Dict[str, Any]:
    """Write ``viz/discterry/public/data3d/{nodes,edges}.parquet`` and ``meta.json``.

    Expects D=3 native coordinates (four ``Inf.Pos`` columns). ``x,y,z`` are Tier-A
    Poincaré-ball display coordinates (see ``poincare_ball_xyz_from_native``).
    Does not overwrite ``public/data/`` (2D bundle).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = paths_for_run(subdir)
    coord_meta, df = parse_inf_coord(paths["inf_coord"])
    pos_cols = inf_pos_columns(df)
    if len(pos_cols) != 4:
        raise ValueError(
            f"export_discterry_public_data_3d expects 4 Inf.Pos columns (D=3 run), got {len(pos_cols)}: {pos_cols}"
        )

    g = load_edges_graph(paths["edge"])
    x, y, z = poincare_ball_xyz_from_native(df, coord_meta)

    repo = repo_root()
    out = repo / "viz" / "discterry" / "public" / "data3d"
    out.mkdir(parents=True, exist_ok=True)

    vertices = [str(v).strip() for v in df["Vertex"].tolist()]
    name_to_i = {name: i for i, name in enumerate(vertices)}

    kappa = df["Inf.Kappa"].to_numpy(dtype=np.float64).astype(np.float32)
    hyp = df["Inf.Hyp.Rad"].to_numpy(dtype=np.float64).astype(np.float32)
    pos_arrays = {
        f"inf_pos_{i + 1}": df[c].to_numpy(dtype=np.float64).astype(np.float32) for i, c in enumerate(pos_cols)
    }

    col_dict: Dict[str, Any] = {
        "vertex": pa.array(vertices, type=pa.string()),
        "x": pa.array(np.asarray(x, dtype=np.float32), type=pa.float32()),
        "y": pa.array(np.asarray(y, dtype=np.float32), type=pa.float32()),
        "z": pa.array(np.asarray(z, dtype=np.float32), type=pa.float32()),
        "inf_kappa": pa.array(kappa, type=pa.float32()),
        "inf_hyp_rad": pa.array(hyp, type=pa.float32()),
    }
    col_dict.update(pos_arrays)
    nodes_tbl = pa.table(col_dict)

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

    run_key = subdir.strip()
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
        "discterry_mode": "3d",
        "n_vertices": len(vertices),
        "n_edges": len(src),
        "default_focus": focus,
        "default_seeds": default_seeds,
        "dimension": 3,
    }
    for k, v in coord_meta.items():
        if k == "nb_vertices":
            meta["mercator_nb_vertices_header"] = v
        else:
            meta[k] = v

    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote 3d nodes:", out / "nodes.parquet", "n_vertices=", len(vertices))
    print("Wrote 3d edges:", out / "edges.parquet", "n_edges=", len(src))
    print("Wrote 3d meta:", out / "meta.json")
    return meta
