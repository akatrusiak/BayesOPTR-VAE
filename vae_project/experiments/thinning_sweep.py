"""
Thinning vs δ-count experiment.

Two questions:
  A. At fixed δ count, does thinning pairs-per-run hurt?
  B. At a fixed total-pair budget (the real constraint during data generation),
     are you better off with fewer δs and thick runs, or more δs and thin runs?

Usage:
    cd vae_project_updated
    python -m experiments.thinning_sweep \
        --data_dir ../data/output_good_4 \
        --out_dir runs/thinning \
        [--epochs 40] [--d_latent 5] [--d_hidden 64] [--seed 0]

Writes one line per config to <out_dir>/results.jsonl plus a summary print.

Design notes:
  * `max_runs` is applied deterministically after loading by taking the first
    N globally-assigned run_ids. Because runs are sorted by (file, local_id)
    order during loading, the subset is reproducible and covers both axes
    evenly up to the chosen count (loaders iterate x-file then y-file per
    folder, so the first N runs skew toward x at small N — not ideal, but
    fine for a same-budget comparison because every config sees the same
    subset).
  * Same train/val split seed across configs.
  * Stats are refit inside each dataset (no cross-config stat sharing).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Make the 'vae' package importable when running this file as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vae.data import MultiFolderRunDataset, collate_runs
from vae.decoder import MisalignmentDecoder
from vae.encoder import DeepSetsEncoder
from vae.model import VAE


# ----------------------------------------------------------------------
def _evaluate_det_recon(vae: VAE, loader: DataLoader) -> float:
    """Deterministic recon (z = mu), summed over dims, mean over batch."""
    vae.eval()
    s, n = 0.0, 0
    with torch.no_grad():
        for pairs, mask, deltas, _ in loader:
            out = vae.forward(pairs, mask, sample=False)
            err = ((out["delta_hat"] - deltas) ** 2).sum(dim=1)
            s += err.sum().item()
            n += err.numel()
    return s / max(n, 1)


def _run_train(
    ds,
    train_idx: list[int],
    val_idx: list[int],
    *,
    d_latent: int,
    d_hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    beta_warmup_epochs: int,
    free_bits: float,
    seed: int,
) -> dict:
    """Train once; return best deterministic val recon + final metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_set = Subset(ds, train_idx)
    val_set = Subset(ds, val_idx)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_runs, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_runs,
    )

    enc = DeepSetsEncoder(d_input=ds.D_input + 1, d_hidden=d_hidden, d_latent=d_latent)
    dec = MisalignmentDecoder(d_latent=d_latent, d_hidden=d_hidden, d_delta=ds.D_delta)
    vae = VAE(enc, dec, beta=beta, free_bits=free_bits)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    best_val = math.inf
    last_train_recon = math.nan
    for epoch in range(1, epochs + 1):
        # beta warmup
        if beta_warmup_epochs > 0 and epoch <= beta_warmup_epochs:
            vae.beta = beta * (epoch / beta_warmup_epochs)
        else:
            vae.beta = beta

        vae.train()
        ep_recon, ep_n = 0.0, 0
        for pairs, mask, deltas, _ in train_loader:
            opt.zero_grad()
            loss, m = vae.elbo_loss(pairs, mask, deltas)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
            opt.step()
            B = pairs.size(0)
            ep_recon += m.recon * B
            ep_n += B
        last_train_recon = ep_recon / ep_n

        val_recon_det = _evaluate_det_recon(vae, val_loader)
        if val_recon_det < best_val:
            best_val = val_recon_det

    return {
        "best_val_recon_det": best_val,
        "final_train_recon": last_train_recon,
    }


def _deterministic_run_subset(
    ds_full: MultiFolderRunDataset, n_keep: int, seed: int
) -> list[int]:
    """
    Pick `n_keep` run indices, balanced across axes when possible.

    We don't want the x/y split to be lopsided just because the loader
    iterates x-file before y-file. For a fair "few δs" comparison, sample
    half from x-runs and half from y-runs (deterministically via `seed`).
    """
    x_idx = [i for i, lbl in enumerate(ds_full.run_labels) if lbl.axis == "x"]
    y_idx = [i for i, lbl in enumerate(ds_full.run_labels) if lbl.axis == "y"]
    rng = np.random.RandomState(seed)
    rng.shuffle(x_idx)
    rng.shuffle(y_idx)
    n_x = n_keep // 2
    n_y = n_keep - n_x
    # Cap at availability
    n_x = min(n_x, len(x_idx))
    n_y = min(n_y, len(y_idx))
    chosen = sorted(x_idx[:n_x] + y_idx[:n_y])
    return chosen


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", action="append", required=True)
    p.add_argument("--out_dir", default="runs/thinning")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--d_latent", type=int, default=5)
    p.add_argument("--d_hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta_warmup_epochs", type=int, default=5)
    p.add_argument("--free_bits", type=float, default=0.3)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results_path.unlink(missing_ok=True)

    # ----- Experiment A: thinning sweep at full δ count -----
    # Full data mean is ~673 pairs/run; we sweep caps that are progressively
    # more aggressive.
    exp_a_caps = [15, 30, 60, 120, 250, None]

    # ----- Experiment B: compute-matched at a fixed total-pair budget -----
    # Budgets chosen to fit within available data.
    # For each budget, three points: (few δs, thick), (medium), (many δs, thin).
    # Each tuple is (n_runs, pairs_per_run).
    exp_b_configs = [
        ("budget_4800", [(40, 120), (80, 60), (160, 30)]),
        ("budget_9600", [(40, 240), (80, 120), (160, 60)]),
        ("budget_19200", [(48, 400), (120, 160), (240, 80)]),
    ]

    # ----- Load full dataset once for train/val split consistency -----
    # We'll rebuild with thinning per-config, but the split pattern comes
    # from the full-size dataset so indices stay consistent.
    ds_full_untrimmed = MultiFolderRunDataset(args.data_dir, standardize=False)
    N = len(ds_full_untrimmed)

    # Train/val split over ALL run indices, once. When a config uses a
    # subset (n_runs < N), we intersect with this split so val items are
    # the same wherever they appear.
    rng = np.random.RandomState(args.seed)
    shuffled = np.arange(N)
    rng.shuffle(shuffled)
    n_val = int(round(args.val_frac * N))
    global_val = set(shuffled[:n_val].tolist())
    global_train = set(shuffled[n_val:].tolist())

    def split_for_subset(run_subset: list[int]) -> tuple[list[int], list[int]]:
        subset_set = set(run_subset)
        train_idx = sorted(subset_set & global_train)
        val_idx = sorted(subset_set & global_val)
        # Guard against degenerate empty splits at very small n_runs:
        if not val_idx:
            # Fall back to first 20% of subset as val
            k = max(1, int(round(args.val_frac * len(run_subset))))
            val_idx = sorted(run_subset[:k])
            train_idx = sorted([i for i in run_subset if i not in set(val_idx)])
        return train_idx, val_idx

    results: list[dict] = []

    def log_and_append(cfg: dict):
        results.append(cfg)
        with open(results_path, "a") as f:
            f.write(json.dumps(cfg) + "\n")
        print(
            f"  [{cfg['experiment']}] "
            f"n_runs={cfg['n_runs']:>3} "
            f"max_pairs={str(cfg['max_pairs']):>5} "
            f"total_pairs={cfg['total_pairs']:>7} "
            f"| best_val_recon_det={cfg['best_val_recon_det']:.3f} "
            f"(train_recon={cfg['final_train_recon']:.3f}) "
            f"[{cfg['wall_s']:.1f}s]"
        )

    # ============================================================
    # Experiment A
    # ============================================================
    print("\n=== Experiment A: thinning sweep at full δ count ===")
    full_runs = list(range(N))  # all available
    for cap in exp_a_caps:
        ds = MultiFolderRunDataset(
            args.data_dir, standardize=True,
            max_pairs_per_run=cap, subsample_strategy="uniform",
        )
        total_pairs = int(sum(len(r) for r in ds._run_row_idx))
        train_idx, val_idx = split_for_subset(full_runs)
        t0 = time.time()
        metrics = _run_train(
            ds, train_idx, val_idx,
            d_latent=args.d_latent, d_hidden=args.d_hidden,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, beta=args.beta,
            beta_warmup_epochs=args.beta_warmup_epochs,
            free_bits=args.free_bits, seed=args.seed,
        )
        log_and_append({
            "experiment": "A",
            "n_runs": len(full_runs),
            "max_pairs": cap,
            "total_pairs": total_pairs,
            "wall_s": time.time() - t0,
            **metrics,
        })

    # ============================================================
    # Experiment B
    # ============================================================
    print("\n=== Experiment B: compute-matched (fewer δs × thick vs more δs × thin) ===")
    for budget_name, configs in exp_b_configs:
        print(f"\n-- {budget_name} --")
        for n_runs, cap in configs:
            if n_runs > N:
                print(f"  (skipping n_runs={n_runs}; only {N} available)")
                continue
            ds = MultiFolderRunDataset(
                args.data_dir, standardize=True,
                max_pairs_per_run=cap, subsample_strategy="uniform",
            )
            run_subset = _deterministic_run_subset(ds, n_runs, args.seed)
            total_pairs = int(sum(len(ds._run_row_idx[i]) for i in run_subset))
            train_idx, val_idx = split_for_subset(run_subset)
            if len(train_idx) < 2 or len(val_idx) < 1:
                print(f"  (skipping n_runs={n_runs}; too few per split)")
                continue
            t0 = time.time()
            metrics = _run_train(
                ds, train_idx, val_idx,
                d_latent=args.d_latent, d_hidden=args.d_hidden,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, beta=args.beta,
                beta_warmup_epochs=args.beta_warmup_epochs,
                free_bits=args.free_bits, seed=args.seed,
            )
            log_and_append({
                "experiment": "B",
                "budget": budget_name,
                "n_runs": n_runs,
                "max_pairs": cap,
                "total_pairs": total_pairs,
                "wall_s": time.time() - t0,
                **metrics,
            })

    # ============================================================
    # Summary
    # ============================================================
    print("\n=== Summary ===")
    print(f"Wrote {len(results)} rows to {results_path}")
    # Sort Experiment A rows by cap (None last)
    rows_a = [r for r in results if r["experiment"] == "A"]
    rows_a.sort(key=lambda r: (r["max_pairs"] is None, r["max_pairs"] or 0))
    print("\nExperiment A (fixed n_runs, sweep max_pairs):")
    print(f"  {'cap':>6}  {'total_pairs':>11}  {'val_recon_det':>13}")
    for r in rows_a:
        print(f"  {str(r['max_pairs']):>6}  {r['total_pairs']:>11}  {r['best_val_recon_det']:>13.3f}")

    rows_b = [r for r in results if r["experiment"] == "B"]
    if rows_b:
        print("\nExperiment B (fixed total-pair budget):")
        # Group by budget
        for budget_name, _ in exp_b_configs:
            bgroup = [r for r in rows_b if r.get("budget") == budget_name]
            if not bgroup:
                continue
            print(f"  -- {budget_name} --")
            print(f"    {'n_runs':>6}  {'max_pairs':>9}  {'total':>7}  {'val_recon_det':>13}")
            for r in sorted(bgroup, key=lambda x: x["n_runs"]):
                print(
                    f"    {r['n_runs']:>6}  {str(r['max_pairs']):>9}  "
                    f"{r['total_pairs']:>7}  {r['best_val_recon_det']:>13.3f}"
                )


if __name__ == "__main__":
    main()
