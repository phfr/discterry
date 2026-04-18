"""
Export node Lorentz coordinates + PPI edges to Apache Arrow IPC files for the WebGPU viz.

Writes to --out-dir:
  - nodes.arrow   : columns id (int32), lorentz (fixed_size_list float32 length d+1), name (utf8)
  - edges.arrow   : columns src, dst (uint32) into node id
  - meta.json       : hyperbolic_dim, ambient_dim, node_count, edge_count
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from src.model import LorentzNodeEmbedding


def _read_edges_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["source", "target"],
        dtype=str,
    )
    df = df.dropna()
    df["source"] = df["source"].str.strip()
    df["target"] = df["target"].str.strip()
    return df


def _canonicalize_pairs(s: pd.Series, t: pd.Series) -> list[tuple[str, str]]:
    lo = np.where(s.values <= t.values, s.values, t.values)
    hi = np.where(s.values <= t.values, t.values, s.values)
    mask = lo != hi
    ed = pd.DataFrame({"lo": lo[mask], "hi": hi[mask]}).drop_duplicates()
    return list(zip(ed["lo"].astype(str), ed["hi"].astype(str)))


def export_arrow(
    checkpoint: Path,
    edges_path: Path,
    out_dir: Path,
) -> None:
    load_kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    ckpt = torch.load(checkpoint, **load_kw)
    idx_to_protein: list[str] = ckpt["idx_to_protein"]
    spatial_dim: int = int(ckpt["spatial_dim"])
    dtype_str: str = ckpt.get("dtype", "float64")
    dt = torch.float64 if dtype_str == "float64" else torch.float32

    amb = spatial_dim + 1
    protein_to_idx = {p: i for i, p in enumerate(idx_to_protein)}

    model = LorentzNodeEmbedding(
        num_nodes=len(idx_to_protein),
        spatial_dim=spatial_dim,
        dtype=dt,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    with torch.no_grad():
        x = model.emb.data.cpu().numpy().astype(np.float32, copy=False)

    ids = np.arange(len(idx_to_protein), dtype=np.int32)
    names = pa.array(idx_to_protein, type=pa.string())
    lorentz = pa.FixedSizeListArray.from_arrays(
        pa.array(x.reshape(-1), type=pa.float32()), amb
    )
    nodes_tbl = pa.table(
        {
            "id": pa.array(ids, type=pa.int32()),
            "lorentz": lorentz,
            "name": names,
        }
    )

    df = _read_edges_tsv(edges_path)
    pairs = _canonicalize_pairs(df["source"], df["target"])
    src_list: list[int] = []
    dst_list: list[int] = []
    for a, b in pairs:
        ia = protein_to_idx.get(a)
        ib = protein_to_idx.get(b)
        if ia is None or ib is None:
            continue
        src_list.append(ia)
        dst_list.append(ib)

    edges_tbl = pa.table(
        {
            "src": pa.array(np.asarray(src_list, dtype=np.uint32)),
            "dst": pa.array(np.asarray(dst_list, dtype=np.uint32)),
        }
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with pa.OSFile(str(out_dir / "nodes.arrow"), "wb") as sink:
        with ipc.new_file(sink, nodes_tbl.schema) as writer:
            writer.write_table(nodes_tbl)

    with pa.OSFile(str(out_dir / "edges.arrow"), "wb") as sink:
        with ipc.new_file(sink, edges_tbl.schema) as writer:
            writer.write_table(edges_tbl)

    meta = {
        "hyperbolic_dim": spatial_dim,
        "ambient_dim": amb,
        "node_count": len(idx_to_protein),
        "edge_count": len(src_list),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Export graph to Arrow IPC for WebGPU viz.")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--edges", type=Path, default=Path("edges.tsv"))
    p.add_argument("--out-dir", type=Path, default=Path("viz/public/data"))
    a = p.parse_args()
    export_arrow(a.checkpoint, a.edges, a.out_dir)
    print(f"Wrote nodes.arrow, edges.arrow, meta.json -> {a.out_dir.resolve()}")


if __name__ == "__main__":
    main()
