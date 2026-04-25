"""
VAE training loop.

Phase 5 logging: we track recon, KL, mu norm, sigma mean, and per-dim KL
separately — because posterior collapse, KL explosion, and dead dimensions
are all invisible in the total-loss curve but trivially visible when the
components are plotted apart.

Checkpointing: the standardization stats are part of the model contract
and are saved with every checkpoint.

Usage:
    # Single folder (production):
    python -m vae.train --data_dir data/output_good_4 --d_latent 5 --epochs 100

    # Multiple folders (as new data comes in, just add more --data_dir flags):
    python -m vae.train \\
        --data_dir data/output_good_4 \\
        --data_dir data/output_good_5 \\
        --d_latent 5 --epochs 100

    # Legacy single-file mode (back-compat; one .npz only):
    python -m vae.train --npz data/.../x_centroid_data.npz --d_latent 5 --epochs 100
"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .data import MultiFolderRunDataset, RunDataset, collate_runs
from .decoder import MisalignmentDecoder
from .encoder import DeepSetsEncoder
from .model import VAE, ELBOMetrics


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------
def train_val_split(ds: Dataset, val_frac: float = 0.2, seed: int = 0):
    """
    Split by *run*, not by envelope, so no run has rows in both splits.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(ds))))
    val_idx = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    return Subset(ds, train_idx), Subset(ds, val_idx)


# ----------------------------------------------------------------------
# Single-epoch evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(vae: VAE, loader: DataLoader) -> dict:
    """Average metrics over a loader. Returns means."""
    vae.eval()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    per_dim_kl_sum = None
    mu_norm_sum = 0.0
    sigma_sum = 0.0
    recon_det_sum = 0.0   # reconstruction at z=mu (no sampling)

    for pairs, mask, deltas, _ in loader:
        B = pairs.size(0)
        # Stochastic ELBO (as during training)
        _, m = vae.elbo_loss(pairs, mask, deltas)
        totals["total"] += m.total * B
        totals["recon"] += m.recon * B
        totals["kl"] += m.kl * B
        totals["n"] += B
        mu_norm_sum += m.mu_norm * B
        sigma_sum += m.sigma_mean * B
        pd = np.asarray(m.per_dim_kl)
        per_dim_kl_sum = pd * B if per_dim_kl_sum is None else per_dim_kl_sum + pd * B

        # Deterministic reconstruction (z = mu)
        out = vae.forward(pairs, mask, sample=False)
        recon_det = ((out["delta_hat"] - deltas) ** 2).sum(dim=1).mean().item()
        recon_det_sum += recon_det * B

    n = totals["n"]
    return {
        "total": totals["total"] / n,
        "recon": totals["recon"] / n,
        "recon_deterministic": recon_det_sum / n,
        "kl": totals["kl"] / n,
        "mu_norm": mu_norm_sum / n,
        "sigma_mean": sigma_sum / n,
        "per_dim_kl": (per_dim_kl_sum / n).tolist(),
    }


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train(
    data_dirs: Sequence[str | Path] | None = None,
    npz_path: str | Path | None = None,
    out_dir: str | Path = "runs/vae",
    d_latent: int = 5,
    d_hidden: int = 128,
    beta: float = 1.0,
    beta_warmup_epochs: int = 0,    # ramp beta from 0 to target over N epochs
    free_bits: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    epochs: int = 50,
    val_frac: float = 0.2,
    grad_clip: float = 10.0,
    seed: int = 0,
    log_every: int = 1,
):
    """
    Train a VAE on TRANSOPTR tuning runs.

    Data source is exactly one of:
      - `data_dirs`: one or more folders (each containing x_centroid_data.npz
        and/or y_centroid_data.npz). Preferred for production.
      - `npz_path`: a single .npz file. Back-compat for older experiments.

    Everything else is a hyperparameter with sensible defaults documented in
    the CLI section below.
    """
    if (data_dirs is None) == (npz_path is None):
        raise ValueError(
            "Provide exactly one of `data_dirs` (list of folders) "
            "or `npz_path` (single file)."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- data --------
    if data_dirs is not None:
        ds: Dataset = MultiFolderRunDataset(data_dirs, standardize=True)
        data_source_desc = [str(Path(d)) for d in data_dirs]
        print(ds.summary() if hasattr(ds, "summary") else f"Dataset with {len(ds)} runs")
    else:
        ds = RunDataset(npz_path, standardize=True)
        data_source_desc = [str(npz_path)]

    train_set, val_set = train_val_split(ds, val_frac=val_frac, seed=seed)
    # Reuse training stats on val (already automatic: both Subsets share the
    # same underlying dataset, so they share its stats).

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_runs, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_runs,
    )

    # -------- model --------
    enc = DeepSetsEncoder(
        d_input=ds.D_input + 1, d_hidden=d_hidden, d_latent=d_latent,
    )
    dec = MisalignmentDecoder(
        d_latent=d_latent, d_hidden=d_hidden, d_delta=ds.D_delta,
    )
    vae = VAE(enc, dec, beta=beta, free_bits=free_bits)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)

    # -------- logging setup --------
    history: list[dict] = []
    best_val = math.inf

    print(f"Training: n_train={len(train_set)}, n_val={len(val_set)}, "
          f"D_input={ds.D_input}, D_delta={ds.D_delta}, d_latent={d_latent}")

    # -------- epochs --------
    for epoch in range(1, epochs + 1):
        # KL warmup: ramp beta linearly from 0 to target
        if beta_warmup_epochs > 0 and epoch <= beta_warmup_epochs:
            vae.beta = beta * (epoch / beta_warmup_epochs)
        else:
            vae.beta = beta

        vae.train()
        ep_total = ep_recon = ep_kl = 0.0
        ep_n = 0
        for pairs, mask, deltas, _ in train_loader:
            opt.zero_grad()
            loss, m = vae.elbo_loss(pairs, mask, deltas)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            opt.step()
            B = pairs.size(0)
            ep_total += m.total * B
            ep_recon += m.recon * B
            ep_kl += m.kl * B
            ep_n += B

        train_metrics = {
            "total": ep_total / ep_n,
            "recon": ep_recon / ep_n,
            "kl": ep_kl / ep_n,
        }
        val_metrics = evaluate(vae, val_loader)

        row = {
            "epoch": epoch,
            "beta": vae.beta,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, list)},
            "val_per_dim_kl": val_metrics["per_dim_kl"],
        }
        history.append(row)

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            pd_kl_str = "[" + ", ".join(f"{k:.2f}" for k in val_metrics["per_dim_kl"]) + "]"
            alive = sum(1 for k in val_metrics["per_dim_kl"] if k > 0.01)
            print(
                f"ep {epoch:3d} | "
                f"train recon {train_metrics['recon']:.3f} kl {train_metrics['kl']:.3f} | "
                f"val recon {val_metrics['recon']:.3f} "
                f"(det {val_metrics['recon_deterministic']:.3f}) "
                f"kl {val_metrics['kl']:.3f} | "
                f"|mu| {val_metrics['mu_norm']:.2f} sigma {val_metrics['sigma_mean']:.2f} | "
                f"alive {alive}/{d_latent} pd_kl {pd_kl_str}"
            )

        # Checkpoint best
        if val_metrics["recon_deterministic"] < best_val:
            best_val = val_metrics["recon_deterministic"]
            _save_checkpoint(out_dir / "best.pt", vae, ds, epoch, val_metrics, data_source_desc, val_frac, seed)

    # Always save last
    _save_checkpoint(out_dir / "last.pt", vae, ds, epoch, val_metrics, data_source_desc, val_frac, seed)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Best val recon_det = {best_val:.4f}. Artifacts in {out_dir}")
    return history


def _save_checkpoint(
    path: Path,
    vae: VAE,
    ds: Dataset,
    epoch: int,
    val_metrics: dict,
    data_source: list[str],
    val_frac: float,
    seed: int,
):
    torch.save(
        {
            "model_state": vae.state_dict(),
            "stats": ds.stats.to_dict(),
            "config": {
                "d_input": ds.D_input,
                "d_delta": ds.D_delta,
                "d_latent": vae.encoder.d_latent,
                "d_hidden": vae.encoder.d_hidden,
                "beta": vae.beta,
                "data_source": data_source,
                "n_runs": len(ds),
                "seed": seed,
                "val_frac": val_frac,
            },
            "epoch": epoch,
            "val_metrics": {
                k: v for k, v in val_metrics.items()
                if not isinstance(v, (list, np.ndarray))
            },
        },
        path,
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Train the VAE. Use --data_dir (repeatable) for production training;\n"
            "--npz is kept for single-file back-compat."
        ),
    )
    # Data source (exactly one required; argparse enforces non-overlap via logic in train()).
    src = p.add_argument_group("data source (pick one)")
    src.add_argument(
        "--data_dir", action="append", default=None,
        help=(
            "Path to a TRANSOPTR output folder containing x_centroid_data.npz "
            "and/or y_centroid_data.npz. Can be passed multiple times to pool "
            "data from several runs: --data_dir data/output_good_4 "
            "--data_dir data/output_good_5 ..."
        ),
    )
    src.add_argument(
        "--npz", default=None,
        help="(Back-compat) Path to a single .npz file. Mutually exclusive with --data_dir.",
    )

    p.add_argument("--out_dir", default="runs/vae")
    p.add_argument("--d_latent", type=int, default=5)
    p.add_argument("--d_hidden", type=int, default=64)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta_warmup_epochs", type=int, default=10)
    p.add_argument("--free_bits", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if (args.data_dir is None) == (args.npz is None):
        p.error("Provide exactly one of --data_dir (repeatable) or --npz.")

    kwargs = vars(args)
    data_dir = kwargs.pop("data_dir")
    npz = kwargs.pop("npz")
    train(data_dirs=data_dir, npz_path=npz, **kwargs)


if __name__ == "__main__":
    main()
