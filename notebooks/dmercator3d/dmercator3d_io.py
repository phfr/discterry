"""Standalone IO for D-Mercator runs with variable ``Inf.Pos.*`` column count (e.g. D=3 → four positions).

Independent of ``notebooks/dmercator/dmercator_io.py``. Intended cwd: ``notebooks/dmercator3d/``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def repo_root() -> Path:
    """``hbol`` root: ``notebooks/dmercator3d`` → ``notebooks`` → root."""
    return Path(__file__).resolve().parent.parent.parent


def get_run_dir(subdir: Optional[str] = None) -> Path:
    """``<repo>/d-mercator-run/<subdir>``. Defaults to ``DMERCATOR_RUN`` env or ``\"d3\"``."""
    s = (subdir or os.environ.get("DMERCATOR_RUN", "d3")).strip()
    return repo_root() / "d-mercator-run" / s


def paths_for_run(subdir: Optional[str] = None) -> Dict[str, Path]:
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


def _infer_n_pos_from_header(header_line: str) -> int:
    """Count ``Inf.Pos.N`` tokens on the Vertex table header line."""
    tokens = re.findall(r"Inf\.Pos\.\d+", header_line)
    if not tokens:
        raise ValueError("No Inf.Pos.N columns found on Vertex header line")
    nums: list[int] = []
    for t in tokens:
        m = re.search(r"(\d+)$", t)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        raise ValueError("Could not parse Inf.Pos column indices")
    return max(nums)


def pos_columns(n_pos: int) -> List[str]:
    return [f"Inf.Pos.{k}" for k in range(1, n_pos + 1)]


def parse_inf_coord(path: Path | str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Parse ``*.inf_coord`` with **variable** number of ``Inf.Pos.*`` columns.

    Rows: ``Vertex``, ``Inf.Kappa``, ``Inf.Hyp.Rad``, ``Inf.Pos.1`` … ``Inf.Pos.{n_pos}``.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta = _parse_coord_meta(raw)
    meta["inf_coord_path"] = str(path.resolve())

    lines = raw.splitlines()
    header_line: Optional[str] = None
    i0: Optional[int] = None
    for i, line in enumerate(lines):
        if "Vertex" in line and "Inf.Kappa" in line and "Inf.Hyp.Rad" in line:
            header_line = line
            i0 = i + 1
            break
    if i0 is None or header_line is None:
        raise ValueError(f"No Vertex / Inf.Kappa header in {path}")

    n_pos = _infer_n_pos_from_header(header_line)
    meta["n_pos"] = n_pos
    tail = 2 + n_pos
    cols = ["Vertex", "Inf.Kappa", "Inf.Hyp.Rad", *pos_columns(n_pos)]

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
        name = " ".join(parts[:-tail]).strip()
        if not name:
            continue
        kappa, hyp_rad = tail_nums[0], tail_nums[1]
        pos_vals = tuple(tail_nums[2:])
        rows.append((name, kappa, hyp_rad, *pos_vals))

    df = pd.DataFrame(rows, columns=cols)
    return meta, df


def load_edges_graph(edge_path: Path | str) -> nx.Graph:
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


def normalize_direction_nd(df: pd.DataFrame) -> np.ndarray:
    """Unit vectors (N × n_pos) from all ``Inf.Pos.*`` columns present in ``df``."""
    cols = [c for c in df.columns if c.startswith("Inf.Pos.")]
    cols.sort(key=lambda c: int(c.split(".")[-1]))
    if not cols:
        raise ValueError("No Inf.Pos.* columns in dataframe")
    p = df[cols].to_numpy(dtype=np.float64)
    n = np.linalg.norm(p, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return p / n


def load_merged_parquet(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def save_merged_parquet(df: pd.DataFrame, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
