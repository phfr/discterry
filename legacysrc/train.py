from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from geoopt.optim import RiemannianAdam
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.config import TrainConfig
from src.data import build_graph_bundle
from src.model import LorentzNodeEmbedding


def _edge_batches(
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
):
    n = edge_label_index.size(1)
    perm = torch.randperm(n, device=edge_label_index.device) if shuffle else torch.arange(n, device=edge_label_index.device)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        src = edge_label_index[0, idx].to(device)
        dst = edge_label_index[1, idx].to(device)
        y = edge_label[idx].float().to(device)
        yield src, dst, y


@torch.no_grad()
def _evaluate_split(
    model: LorentzNodeEmbedding,
    data,
    device: torch.device,
    batch_size: int,
) -> tuple[float | None, float | None]:
    model.eval()
    logits_list = []
    y_list = []
    ei = data.edge_label_index
    el = data.edge_label
    for src, dst, y in _edge_batches(ei, el, batch_size, shuffle=False, device=device):
        logits_list.append(model.link_logits(src, dst).detach().cpu())
        y_list.append(y.detach().cpu())
    if not logits_list:
        return None, None
    logits = torch.cat(logits_list).numpy()
    y = torch.cat(y_list).numpy()
    if y.min() == y.max():
        return None, None
    auc = float(roc_auc_score(y, logits))
    ap = float(average_precision_score(y, logits))
    return auc, ap


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    if str(cfg.device).lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(cfg.device)

    dt = cfg.torch_dtype()
    bundle = build_graph_bundle(cfg)
    model = LorentzNodeEmbedding(
        num_nodes=len(bundle.idx_to_protein),
        spatial_dim=cfg.spatial_dim,
        dtype=dt,
    ).to(device=device, dtype=dt)

    opt_man = RiemannianAdam([model.emb], lr=cfg.lr)
    opt_dec = torch.optim.Adam([model.dec_r, model.dec_log_t], lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.checkpoint_dir / "best.pt"
    last_path = cfg.checkpoint_dir / "last.pt"

    best_ap = -1.0
    epoch_pbar = tqdm(range(cfg.epochs), desc="epochs", leave=True)
    for epoch in epoch_pbar:
        model.train()
        train_ei = bundle.train_data.edge_label_index
        train_el = bundle.train_data.edge_label
        n_batches = (train_ei.size(1) + cfg.batch_size - 1) // cfg.batch_size
        batch_pbar = tqdm(
            _edge_batches(train_ei, train_el, cfg.batch_size, shuffle=True, device=device),
            total=n_batches,
            desc=f"train e{epoch}",
            leave=False,
        )
        running_loss = 0.0
        n_seen = 0
        for bi, (src, dst, y) in enumerate(batch_pbar):
            opt_man.zero_grad()
            opt_dec.zero_grad()
            logits = model.link_logits(src, dst)
            loss = loss_fn(logits, y)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.emb, cfg.grad_clip)
                torch.nn.utils.clip_grad_norm_([model.dec_r, model.dec_log_t], cfg.grad_clip)
            opt_man.step()
            opt_dec.step()
            model.project_embeddings()

            bs = y.numel()
            running_loss += float(loss.detach()) * bs
            n_seen += bs
            if bi % max(1, cfg.log_every) == 0:
                batch_pbar.set_postfix(loss=f"{float(loss.detach()):.4f}")

        train_loss = running_loss / max(1, n_seen)
        val_auc, val_ap = _evaluate_split(model, bundle.val_data, device, cfg.batch_size)
        te_auc, te_ap = _evaluate_split(model, bundle.test_data, device, cfg.batch_size)

        postfix = {"tr_loss": f"{train_loss:.4f}"}
        if val_auc is not None:
            postfix["v_auc"] = f"{val_auc:.4f}"
            postfix["v_ap"] = f"{val_ap:.4f}"
        if te_auc is not None:
            postfix["te_auc"] = f"{te_auc:.4f}"
            postfix["te_ap"] = f"{te_ap:.4f}"
        epoch_pbar.set_postfix(**postfix)

        payload = {
            "model": model.state_dict(),
            "idx_to_protein": bundle.idx_to_protein,
            "spatial_dim": cfg.spatial_dim,
            "dtype": cfg.dtype,
        }
        torch.save(payload, last_path)
        if val_ap is not None and val_ap > best_ap:
            best_ap = val_ap
            torch.save(payload, best_path)

    meta = {
        "best_val_ap": best_ap,
        "checkpoint_dir": str(cfg.checkpoint_dir),
    }
    (cfg.checkpoint_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train Lorentz PPI embeddings.")
    p.add_argument("--edges", type=Path, default=Path("edges.tsv"))
    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--neg-sampling-ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--fp32", action="store_true", help="Shortcut for --dtype float32")
    p.add_argument("--toy-edges", type=int, default=None)
    p.add_argument("--toy-nodes", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-grad-clip", action="store_true")
    a = p.parse_args()
    dtype = "float32" if a.fp32 else a.dtype
    return TrainConfig(
        edges_path=a.edges,
        spatial_dim=a.spatial_dim,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        val_ratio=a.val_ratio,
        test_ratio=a.test_ratio,
        neg_sampling_ratio=a.neg_sampling_ratio,
        seed=a.seed,
        device=a.device,
        dtype=dtype,
        toy_edges=a.toy_edges,
        toy_nodes=a.toy_nodes,
        checkpoint_dir=a.checkpoint_dir,
        grad_clip=None if a.no_grad_clip else a.grad_clip,
        log_every=a.log_every,
    )


def main() -> None:
    cfg = _parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
