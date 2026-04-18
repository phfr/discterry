from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from src.config import TrainConfig


@dataclass
class GraphBundle:
    """Holds split graphs and node id ↔ index maps."""

    train_data: Data
    val_data: Data
    test_data: Data
    idx_to_protein: list[str]
    protein_to_idx: dict[str, int]


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
    """Unique undirected edges as (u, v) with u < v lexicographically."""
    lo = np.where(s.values <= t.values, s.values, t.values)
    hi = np.where(s.values <= t.values, t.values, s.values)
    mask = lo != hi
    ed = pd.DataFrame({"lo": lo[mask], "hi": hi[mask]}).drop_duplicates()
    return list(zip(ed["lo"].astype(str), ed["hi"].astype(str)))


def _subsample_pairs(pairs: list[tuple[str, str]], max_edges: int, seed: int) -> list[tuple[str, str]]:
    if len(pairs) <= max_edges:
        return pairs
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(pairs), size=max_edges, replace=False)
    return [pairs[i] for i in pick]


def _top_degree_nodes(pairs: list[tuple[str, str]], max_nodes: int) -> set[str]:
    deg: dict[str, int] = {}
    for a, b in pairs:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    ranked = sorted(deg.keys(), key=lambda x: (-deg[x], x))
    keep = set(ranked[:max_nodes])
    return keep


def _filter_pairs_by_nodes(pairs: list[tuple[str, str]], nodes: set[str]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in pairs if a in nodes and b in nodes]


def _build_index_maps(pairs: list[tuple[str, str]]) -> tuple[list[str], dict[str, int]]:
    verts = set()
    for a, b in pairs:
        verts.add(a)
        verts.add(b)
    sorted_nodes = sorted(verts)
    protein_to_idx = {p: i for i, p in enumerate(sorted_nodes)}
    return sorted_nodes, protein_to_idx


def build_graph_bundle(cfg: TrainConfig) -> GraphBundle:
    df = _read_edges_tsv(Path(cfg.edges_path))
    pairs = _canonicalize_pairs(df["source"], df["target"])

    if cfg.toy_edges is not None:
        pairs = _subsample_pairs(pairs, cfg.toy_edges, cfg.seed)

    if cfg.toy_nodes is not None:
        keep = _top_degree_nodes(pairs, cfg.toy_nodes)
        pairs = _filter_pairs_by_nodes(pairs, keep)

    idx_to_protein, protein_to_idx = _build_index_maps(pairs)
    u = np.array([protein_to_idx[a] for a, _ in pairs], dtype=np.int64)
    v = np.array([protein_to_idx[b] for _, b in pairs], dtype=np.int64)

    edge_index = torch.tensor(
        np.stack([np.concatenate([u, v]), np.concatenate([v, u])], axis=0),
        dtype=torch.long,
    )
    num_nodes = len(idx_to_protein)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    rls_kw: dict = dict(
        num_val=cfg.val_ratio,
        num_test=cfg.test_ratio,
        is_undirected=True,
        neg_sampling_ratio=cfg.neg_sampling_ratio,
    )
    sig = inspect.signature(RandomLinkSplit.__init__)
    if "add_negative_train_samples" in sig.parameters:
        rls_kw["add_negative_train_samples"] = True
    elif "add_neg_train_samples" in sig.parameters:
        rls_kw["add_neg_train_samples"] = True
    splitter = RandomLinkSplit(**rls_kw)
    train_data, val_data, test_data = splitter(data)

    return GraphBundle(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        idx_to_protein=idx_to_protein,
        protein_to_idx=protein_to_idx,
    )
