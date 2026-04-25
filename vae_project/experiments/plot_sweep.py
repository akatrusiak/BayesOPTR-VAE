#!/usr/bin/env python3
"""
Aggregate and plot results from a multi-seed d_z sweep.

Reads history.json + best.pt from each {out_root}/vae_dz{DZ}_s{SEED}/ directory,
produces a 4-panel figure with mean line and min/max shading across seeds, and
prints/saves a summary table of best val_recon_deterministic per d_z
(mean ± std across seeds).

Example:
    python plot_sweep.py --out_root runs/sweep --d_latents 5 8 10 12 --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", default="runs/sweep")
    p.add_argument("--d_latents", type=int, nargs="+", default=[5, 8, 10, 12])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--fig_path", default="sweep_curves.png")
    p.add_argument("--csv_path", default="sweep_summary.csv")
    p.add_argument("--baseline", type=float, default=27.0,
                   help="Predict-the-mean baseline value (default 27 = D_delta).")
    return p.parse_args()


def load_run(run_dir: Path):
    """Return (history_list, best_metrics_dict) or (None, None) if incomplete."""
    h_path = run_dir / "history.json"
    b_path = run_dir / "best.pt"
    if not (h_path.exists() and b_path.exists()):
        return None, None
    with open(h_path) as f:
        history = json.load(f)
    ckpt = torch.load(b_path, map_location="cpu", weights_only=False)
    return history, ckpt


def stack_curves(runs_for_dz, key):
    """Stack a per-epoch metric across seeds into shape (n_seeds, n_epochs)."""
    arrs = [np.array([e[key] for e in h]) for h, _ in runs_for_dz]
    # Truncate to shortest length (in case some seeds were cut short)
    n = min(len(a) for a in arrs)
    return np.stack([a[:n] for a in arrs], axis=0), np.arange(1, n + 1)


def main():
    args = parse_args()
    root = Path(args.out_root)

    colors = {5: "#1f77b4", 8: "#ff7f0e", 10: "#2ca02c", 12: "#d62728"}
    # Fallback palette if d_latents differ from defaults
    fallback = plt.cm.tab10.colors
    for i, dz in enumerate(args.d_latents):
        if dz not in colors:
            colors[dz] = fallback[i % len(fallback)]

    # Collect runs grouped by d_z
    runs = {dz: [] for dz in args.d_latents}
    for dz in args.d_latents:
        for seed in args.seeds:
            rd = root / f"vae_dz{dz}_s{seed}"
            h, ck = load_run(rd)
            if h is None:
                print(f"[warn] missing or incomplete: {rd}")
                continue
            runs[dz].append((h, ck))
        print(f"dz={dz}: loaded {len(runs[dz])}/{len(args.seeds)} seeds")

    # --- Figure: 4 panels, mean curve with min/max band ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("val_recon_deterministic", "Reconstruction loss",
         "Val deterministic recon  (dashed = train recon)", axes[0, 0]),
        ("val_kl", "Val KL", "KL divergence vs prior", axes[0, 1]),
        ("val_mu_norm", "||μ_φ||", "Posterior mean norm (val avg)", axes[1, 0]),
        ("val_sigma_mean", "mean σ_φ", "Posterior std (val avg)", axes[1, 1]),
    ]

    for dz in args.d_latents:
        if not runs[dz]:
            continue
        c = colors[dz]
        for key, ylabel, title, ax in panels:
            stacked, ep = stack_curves(runs[dz], key)
            mean = stacked.mean(axis=0)
            lo, hi = stacked.min(axis=0), stacked.max(axis=0)
            ax.fill_between(ep, lo, hi, color=c, alpha=0.18)
            ax.plot(ep, mean, color=c, lw=1.8, label=f"d_z={dz}  (n={len(runs[dz])})")

        # Overlay train_recon as dashed on the top-left panel
        tr_stacked, ep = stack_curves(runs[dz], "train_recon")
        axes[0, 0].plot(ep, tr_stacked.mean(axis=0), "--", color=c, lw=1.0, alpha=0.7)

    axes[0, 0].axhline(args.baseline, color="k", linestyle=":", alpha=0.6,
                       label=f"predict-mean baseline ({args.baseline})")
    axes[0, 0].set_ylim(20, 32)

    for _, ylabel, title, ax in panels:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"d_z sweep — mean curves with min/max band across seeds "
                 f"({len(args.seeds)} seeds per d_z)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.fig_path, dpi=110, bbox_inches="tight")
    print(f"\nwrote {args.fig_path}")

    # --- Summary table: best val_recon_det per seed, then mean±std per d_z ---
    print(f"\n{'='*70}")
    print(f"{'d_z':>4} {'n_seeds':>8} {'best_recon_det (mean±std)':>28} "
          f"{'range':>16} {'best_epochs':>18}")
    print("-" * 80)

    csv_lines = ["d_z,n_seeds,mean_best_recon_det,std_best_recon_det,"
                 "min_best_recon_det,max_best_recon_det,best_epochs"]
    for dz in args.d_latents:
        if not runs[dz]:
            continue
        bests, epochs = [], []
        for h, _ in runs[dz]:
            vrd = np.array([e["val_recon_deterministic"] for e in h])
            bests.append(float(vrd.min()))
            epochs.append(int(np.argmin(vrd)) + 1)
        b = np.array(bests)
        print(f"{dz:>4d} {len(b):>8d}   {b.mean():>10.4f} ± {b.std(ddof=0):<7.4f}"
              f"     [{b.min():.3f}, {b.max():.3f}]   "
              f"{epochs!s:>18}")
        csv_lines.append(f"{dz},{len(b)},{b.mean():.6f},{b.std(ddof=0):.6f},"
                         f"{b.min():.6f},{b.max():.6f},"
                         f"{'|'.join(str(e) for e in epochs)}")
    print("=" * 80)
    print(f"baseline (predict mean of delta) = {args.baseline}")

    with open(args.csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"wrote {args.csv_path}")


if __name__ == "__main__":
    main()
