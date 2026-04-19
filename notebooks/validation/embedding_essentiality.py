"""Join D-Mercator ``Vertex`` symbols with SNAP human essentiality (via NCBI Gene ID → Symbol)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "resources"


def load_gene_id_to_symbol(path: Optional[Path] = None) -> pd.Series:
    """NCBI ``Homo_sapiens.gene_info.gz`` → Series index GeneID, value Symbol."""
    path = path or RES / "Homo_sapiens.gene_info.gz"
    df = pd.read_csv(path, sep="\t", compression="gzip", dtype=str, low_memory=False)
    # NCBI header line begins with ``#tax_id``; strip ``#`` from column names.
    df.columns = [str(c).lstrip("#") for c in df.columns]
    if "tax_id" in df.columns:
        df = df[df["tax_id"].astype(str) == "9606"]
    df["GeneID"] = pd.to_numeric(df["GeneID"], errors="coerce")
    df = df.dropna(subset=["GeneID", "Symbol"])
    df["GeneID"] = df["GeneID"].astype(int)
    s = df.drop_duplicates(subset=["GeneID"]).set_index("GeneID")["Symbol"]
    return s


def load_snap_essentiality(path: Optional[Path] = None) -> pd.DataFrame:
    """SNAP ``G-HumanEssential.tsv.gz`` → columns ``gene_id``, ``gene_symbol``, ``essential`` (0/1)."""
    path = path or RES / "G-HumanEssential.tsv.gz"
    raw = pd.read_csv(path, sep="\t", compression="gzip")
    id_col = "Gene ID"
    lab_col = [c for c in raw.columns if "Essentiality" in c][0]
    gmap = load_gene_id_to_symbol()
    gid = raw[id_col].astype(int)
    sym = gid.map(gmap)
    essential = raw[lab_col].astype(str).str.strip().eq("Essential").astype(np.int8)
    out = pd.DataFrame({"gene_id": gid, "gene_symbol": sym, "essential": essential})
    return out.dropna(subset=["gene_symbol"]).copy()


def normalize_vertex_key(s: pd.Series) -> pd.Series:
    """Uppercase string vertex for matching HGNC-style symbols."""
    return s.astype(str).str.strip().str.upper()


def attach_essentiality(
    df: pd.DataFrame,
    *,
    ess: Optional[pd.DataFrame] = None,
    vertex_col: str = "Vertex",
) -> pd.DataFrame:
    """Left-join essentiality on ``Vertex`` (uppercased) vs ``gene_symbol`` (uppercased)."""
    ess = ess if ess is not None else load_snap_essentiality()
    e = (
        ess.assign(_k=normalize_vertex_key(ess["gene_symbol"]))
        .drop_duplicates(subset=["_k"])
        .rename(columns={"gene_id": "essentiality_ncbi_gene_id"})[
            ["_k", "essential", "essentiality_ncbi_gene_id"]
        ]
    )
    left = df.copy()
    left["_k"] = normalize_vertex_key(left[vertex_col])
    m = left.merge(e, on="_k", how="left", suffixes=("", "_ess"))
    m.drop(columns=["_k"], inplace=True)
    return m


def disk_radius_from_ortho_xy(df: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean radius in the orthographic disk chart √(x²+y²) (third component fixed by sphere)."""
    return np.sqrt(np.asarray(x, dtype=np.float64) ** 2 + np.asarray(y, dtype=np.float64) ** 2)
