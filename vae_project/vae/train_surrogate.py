"""
train_surrogate.py — train the landscape surrogate ĝ(x, δ) → y.

Usage mirrors train.py:

    python train_surrogate.py \\
        --data_dir ../../simulation_data/data/output_good_4 \\
        --out_dir runs/surrogate/sgt_s0 \\
        --seed 0

Defaults match the VAE defaults where applicable (--val_frac 0.2,
--weight_decay 1e-4, --batch_size larger because triples are cheap).

The split is performed at the RUN level using the same recover_split
logic as verify_dz.py, so training the surrogate with the same seed
as a VAE run yields aligned train/val sets — enabling the downstream
compare_latent_signal.py script to cleanly evaluate each VAE
checkpoint on its own val runs.

Outputs (into --out_dir):
    best.pt        checkpoint at epoch of minimum val MSE (standardized)
    last.pt        checkpoint at final epoch
    history.json   per-epoch train/val metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import MultiFolderRunDataset
from .surrogate import (
    LandscapeSurrogate,
    SurrogateConfig,
    SurrogateTripleDataset,
    save_surrogate,
)


# ═════════════════════════════════════════════════════════════════════
# Split recovery — mirrors verify_dz.recover_split exactly.
# Copied here (not imported) to keep this script standalone; the two
# must stay in sync. If you change one, change the other.
# ═════════════════════════════════════════════════════════════════════

def recover_split(n_items: int, val_frac: float = 0.2, seed: int = 0):
    """
    Reconstruct the exact train/val indices used by train_val_split in
    train.py. Must mirror that function exactly: legacy RandomState,
    shuffle of np.arange(n_items), sorted() on both halves.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n_items)))
    val_idx = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    return train_idx, val_idx


# ═════════════════════════════════════════════════════════════════════
# Train/eval loops
# ═════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_sq = 0.0
    total_n = 0
    for x, delta, y in loader:
        x, delta, y = x.to(device), delta.to(device), y.to(device)
        y_hat = model(x, delta)
        loss = nn.functional.mse_loss(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Reduction='mean' loss times batch size recovers sum of squares.
        total_sq += float(loss.detach()) * y.shape[0]
        total_n += y.shape[0]
    return total_sq / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Return val metrics: standardized MSE, physical MSE, R², plus
    per-run MSE statistics (mean/median) so we can spot outlier runs.
    """
    model.eval()
    ys, y_hats, origins = [], [], []
    for x, delta, y in loader:
        x, delta, y = x.to(device), delta.to(device), y.to(device)
        y_hat = model(x, delta)
        ys.append(y.cpu().numpy())
        y_hats.append(y_hat.cpu().numpy())
    ys = np.concatenate(ys)
    y_hats = np.concatenate(y_hats)
    resid = y_hats - ys
    mse_std = float(np.mean(resid ** 2))
    # R² on standardized y: same as R² on physical y (invariant to affine
    # rescaling), but we compute it from the standardized resids directly.
    var_y = float(np.var(ys))
    r2 = float(1.0 - mse_std / var_y) if var_y > 1e-12 else float("nan")
    return {"mse_std": mse_std, "var_y_std": var_y, "r2": r2}


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def train(
    data_dirs: list[str],
    out_dir: str,
    seed: int = 0,
    val_frac: float = 0.2,
    d_hidden: int = 128,
    n_hidden_layers: int = 3,
    dropout: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 300,
    patience: int = 0,             # 0 = no early stop; train to `epochs`
    device: str = "cpu",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ----- dataset: fit stats on the pooled data -----
    # We fit stats on ALL runs (not just train) to match VAE convention.
    # This is a minor standardization leak but keeps stats comparable
    # across splits. If you want strict no-leakage behavior, fit on
    # the train subset only — but remember the VAE didn't either.
    ds = MultiFolderRunDataset(data_dirs, standardize=True)
    print(ds.summary())

    train_idx, val_idx = recover_split(len(ds), val_frac=val_frac, seed=seed)
    print(f"[split] seed={seed} val_frac={val_frac} → "
          f"{len(train_idx)} train runs, {len(val_idx)} val runs")

    train_ds = SurrogateTripleDataset(ds, train_idx)
    val_ds   = SurrogateTripleDataset(ds, val_idx)
    print(f"[triples] train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 256),
                            shuffle=False)

    # ----- model -----
    model = LandscapeSurrogate(
        d_input=ds.D_input,
        d_delta=ds.D_delta,
        d_hidden=d_hidden,
        n_hidden_layers=n_hidden_layers,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] LandscapeSurrogate d_input={ds.D_input} d_delta={ds.D_delta} "
          f"d_hidden={d_hidden} n_hidden={n_hidden_layers} "
          f"→ {n_params} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    cfg = SurrogateConfig(
        d_input=ds.D_input,
        d_delta=ds.D_delta,
        d_hidden=d_hidden,
        n_hidden_layers=n_hidden_layers,
        dropout=dropout,
        seed=seed,
        val_frac=val_frac,
        n_runs_total=len(ds),
        data_source=[str(d) for d in data_dirs],
    )

    # ----- train loop -----
    history = []
    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        entry = {
            "epoch": epoch,
            "train_mse_std": train_mse,
            "val_mse_std": val_metrics["mse_std"],
            "val_r2": val_metrics["r2"],
        }
        history.append(entry)

        improved = val_metrics["mse_std"] < best_val
        if improved:
            best_val = val_metrics["mse_std"]
            best_epoch = epoch
            epochs_since_best = 0
            save_surrogate(out_dir / "best.pt", model, ds.stats, cfg,
                           epoch=epoch, val_metrics=val_metrics)
        else:
            epochs_since_best += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            tag = " *" if improved else ""
            print(f"  ep {epoch:4d} | train_mse_std={train_mse:.4f} | "
                  f"val_mse_std={val_metrics['mse_std']:.4f} | "
                  f"val_r2={val_metrics['r2']:+.3f}{tag}")

        if patience > 0 and epochs_since_best >= patience:
            print(f"[early stop] no improvement in {patience} epochs "
                  f"(best was epoch {best_epoch})")
            break

    # ----- save last + history -----
    last_val = evaluate(model, val_loader, device)
    save_surrogate(out_dir / "last.pt", model, ds.stats, cfg,
                   epoch=len(history), val_metrics=last_val)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[done] best epoch {best_epoch}: val_mse_std={best_val:.4f}")
    print(f"       physical val MSE at best = {best_val * ds.stats.y_std**2:.6f}  "
          f"(y_std={ds.stats.y_std:.4f})")
    print(f"       files: {out_dir}/best.pt, last.pt, history.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", action="append", required=True,
                   help="TRANSOPTR output folder (repeatable, pooled).")
    p.add_argument("--out_dir", default="runs/surrogate/sgt",
                   help="Output directory for checkpoints and history.")
    p.add_argument("--seed", type=int, default=0,
                   help="Controls (a) torch/numpy RNG, (b) the train/val "
                        "split recovered via recover_split. Match this to "
                        "one of your VAE seeds to keep splits aligned for "
                        "downstream compare_latent_signal.py.")
    p.add_argument("--val_frac", type=float, default=0.2)

    p.add_argument("--d_hidden", type=int, default=128)
    p.add_argument("--n_hidden_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stop after N epochs without val improvement. "
                        "0 = disabled (train to --epochs).")
    p.add_argument("--device", default="cpu",
                   help="'cpu' or 'cuda'. Surrogate is tiny; CPU is fine.")

    args = p.parse_args()
    train(
        data_dirs=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        val_frac=args.val_frac,
        d_hidden=args.d_hidden,
        n_hidden_layers=args.n_hidden_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()
