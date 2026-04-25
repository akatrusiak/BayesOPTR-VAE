"""
plot_transmission_histograms.py — transmission distribution across all runs.

Reads every *centroid_data.npz in --data_dir and plots:
  (1) histogram of all SA-trajectory transmission values (linear + log overlay)
  (2) histogram of per-run T_last  (= the last SA step per run, what verify_dz
      uses as T_true)
  (3) histogram of per-run T_max   (= peak transmission reached in the run)
  (4) scatter of T_last vs T_max   (points below the diagonal are runs whose
      final step wasn't their best)

Usage:
    python plot_transmission_histograms.py \\
        --data_dir path/to/output_good_4 \\
        --data_dir path/to/output_good_5 \\
        --data_dir path/to/output_good_6 \\
        --out transmission_histograms.png

The --attractor flag (default 0.027) draws a vertical marker at a
transmission value of interest (e.g., the verification-pipeline attractor
we found in the diagnostic). Set to a negative value to hide the marker.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", action="append", required=True,
                   help="Folder containing one or both of x_centroid_data.npz, "
                        "y_centroid_data.npz. Repeatable; all folders are pooled.")
    p.add_argument("--out", default="transmission_histograms.png",
                   help="Output figure path.")
    p.add_argument("--attractor", type=float, default=0.027,
                   help="Reference transmission value to mark on the plots "
                        "(the verification-attractor T we observed). "
                        "Set <0 to hide.")
    p.add_argument("--bins_all", type=int, default=150,
                   help="Bin count for the all-points histogram.")
    p.add_argument("--bins_per_run", type=int, default=50,
                   help="Bin count for the per-run histograms.")
    p.add_argument("--t_max", type=float, default=None,
                   help="Upper x-limit for histograms (default: slightly above "
                        "the observed max).")
    p.add_argument("--full_hist_only", action="store_true",default=False,
                   help="If set, only make the full all-points histogram")
    return p.parse_args()


def discover_files(data_dirs: list[str]) -> list[Path]:
    """Collect every x_centroid_data.npz / y_centroid_data.npz under each data_dir."""
    files: list[Path] = []
    for d in data_dirs:
        dp = Path(d)
        if not dp.is_dir():
            raise FileNotFoundError(f"Not a directory: {dp}")
        for name in ["x_centroid_data.npz", "y_centroid_data.npz"]:
            f = dp / name
            if f.exists():
                files.append(f)
    if not files:
        raise FileNotFoundError(
            f"No centroid_data.npz files found under any of: {data_dirs}"
        )
    return files


def load_transmissions(files: list[Path]):
    """
    Load all transmission values, plus per-run last/max summaries.

    Returns:
        all_pts:        flat array of every SA-trajectory transmission value
        per_run_last:   array, one last-value per run
        per_run_max:    array, one max-value per run
    """
    all_pts = []
    per_run_last = []
    per_run_max = []

    for f in files:
        with np.load(f, allow_pickle=False) as d:
            t = d["transmission"]
            mid = d["misalignment_id"]
        all_pts.append(t)
        for uid in np.unique(mid):
            rows = np.where(mid == uid)[0]
            rows.sort()  # SA-time ordering
            per_run_last.append(float(t[rows[-1]]))
            per_run_max.append(float(t[rows].max()))

    all_pts = np.concatenate(all_pts)
    per_run_last = np.array(per_run_last)
    per_run_max = np.array(per_run_max)
    return all_pts, per_run_last, per_run_max


def print_summary(all_pts, per_run_last, per_run_max, attractor):
    print(f"Total points:    {len(all_pts):,}")
    print(f"Total runs:      {len(per_run_last):,}")
    print(f"All-points:   min={all_pts.min():.4f}  max={all_pts.max():.4f}  "
          f"mean={all_pts.mean():.4f}  median={np.median(all_pts):.4f}")
    print(f"Per-run last: min={per_run_last.min():.4f}  max={per_run_last.max():.4f}  "
          f"mean={per_run_last.mean():.4f}  median={np.median(per_run_last):.4f}")
    print(f"Per-run max:  min={per_run_max.min():.4f}  max={per_run_max.max():.4f}  "
          f"mean={per_run_max.mean():.4f}  median={np.median(per_run_max):.4f}")
    tiny = (all_pts < 0.005).sum()
    zero_exact = (all_pts == 0).sum()
    print(f"\nPoints with T < 0.005:    {tiny:,} / {len(all_pts):,} = "
          f"{100 * tiny / len(all_pts):.1f}%")
    print(f"Points with T exactly 0:  {zero_exact:,} / {len(all_pts):,} = "
          f"{100 * zero_exact / len(all_pts):.2f}%")
    if attractor is not None and attractor > 0:
        lo, hi = attractor - 0.0025, attractor + 0.0025
        near = ((all_pts > lo) & (all_pts < hi)).sum()
        near_last = ((per_run_last > lo) & (per_run_last < hi)).sum()
        print(f"\nAll points in [{lo:.4f}, {hi:.4f}]: {near:,} / {len(all_pts):,} = "
              f"{100 * near / len(all_pts):.2f}%")
        print(f"Runs whose LAST transmission is in [{lo:.4f}, {hi:.4f}]: "
              f"{near_last} / {len(per_run_last)} = "
              f"{100 * near_last / len(per_run_last):.1f}%")
    frac_last_ne_max = (per_run_last < per_run_max - 0.01).sum() / len(per_run_last)
    print(f"\nRuns where T_last < T_max by >0.01: {frac_last_ne_max * 100:.1f}%")


def make_figure(all_pts, per_run_last, per_run_max,
                attractor, bins_all, bins_per_run, t_max, full_hist_only=False):
    if t_max is None:
        t_max = float(max(all_pts.max(), per_run_max.max()) * 1.05)

    show_attractor = attractor is not None and attractor > 0

    
    if full_hist_only:
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(1, 1, hspace=0.35, wspace=0.30)
    else:
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)
    

    # ---- Main histogram: all points ----
    ax = fig.add_subplot(gs[0, :])
    bins = np.linspace(0, t_max, bins_all + 1)
    counts, edges, _ = ax.hist(all_pts, bins=bins, color="steelblue", edgecolor="none")
    ax.set_xlabel("Transmission")
    ax.set_ylabel("Count")
    ax.set_title(f"All SA-trajectory transmission values  (n = {len(all_pts):,})")

    if show_attractor:
        ax.axvline(attractor, color="crimson", lw=1.5, ls="--",
                   label=f"T ≈ {attractor:.3f} (reference)")
        ax.axvspan(attractor - 0.0025, attractor + 0.0025,
                   color="crimson", alpha=0.15)
    ax.axvline(np.median(all_pts), color="black", lw=1, ls=":", alpha=0.6,
               label=f"median = {np.median(all_pts):.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    # Log-y overlay on secondary axis
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax2.plot(centers, counts + 1, color="navy", lw=0.8, alpha=0.6)
    ax2.set_ylabel("Count (log, overlay)", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    
    if not full_hist_only:
        # ---- Per-run T_last ----
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(per_run_last, bins=np.linspace(0, t_max, bins_per_run + 1),
                color="darkorange", edgecolor="none")
        # set y axis to log
        ax.set_yscale("log")
        if show_attractor:
            ax.axvline(attractor, color="crimson", lw=1.5, ls="--")
            ax.axvspan(attractor - 0.0025, attractor + 0.0025, color="crimson", alpha=0.15)
        ax.set_xlabel("Last SA-step transmission  (= T_true)")
        ax.set_ylabel("Run count")
        ax.set_title(f"Per-run T_true  (n = {len(per_run_last)})")
        ax.grid(alpha=0.3)

        # ---- Per-run T_max ----
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(per_run_max, bins=np.linspace(0, t_max, bins_per_run + 1),
                color="seagreen", edgecolor="none")
        ax.set_yscale("log")
        if show_attractor:
            ax.axvline(attractor, color="crimson", lw=1.5, ls="--")
            ax.axvspan(attractor - 0.0025, attractor + 0.0025, color="crimson", alpha=0.15)
        ax.set_xlabel("Max SA-step transmission per run")
        ax.set_ylabel("Run count")
        ax.set_title(f"Per-run T_max  (n = {len(per_run_max)})")
        ax.grid(alpha=0.3)

        # ---- Last vs Max scatter ----
        ax = fig.add_subplot(gs[1, 2])
        ax.scatter(per_run_max, per_run_last, s=6, alpha=0.4, color="purple")
        ax.plot([0, t_max], [0, t_max], "k--", alpha=0.5, lw=1)
        if show_attractor:
            ax.axhline(attractor, color="crimson", ls=":", alpha=0.6)
            ax.axvline(attractor, color="crimson", ls=":", alpha=0.6)
        ax.set_xlabel("T_max (per run)")
        ax.set_ylabel("T_last (per run)")
        ax.set_title("Last vs Max  (below diagonal = last worse than max)")
        ax.set_xlim(0, t_max); ax.set_ylim(0, t_max)
        ax.grid(alpha=0.3)

        frac_last_ne_max = (per_run_last < per_run_max - 0.01).sum() / len(per_run_last)
        ax.text(0.05, 0.72 * t_max,
                f"last < max by >0.01:\n{frac_last_ne_max * 100:.1f}% of runs",
                fontsize=9, color="purple",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle(
        f"Transmission distribution  —  {len(per_run_last)} runs, "
        f"{len(all_pts):,} trajectory points",
        fontsize=12,
    )
    return fig


def main():
    args = parse_args()
    files = discover_files(args.data_dir)
    print(f"Found {len(files)} centroid_data.npz file(s):")
    for f in files:
        print(f"  {f}")

    all_pts, per_run_last, per_run_max = load_transmissions(files)
    print()
    print_summary(all_pts, per_run_last, per_run_max, args.attractor)

    fig = make_figure(all_pts, per_run_last, per_run_max,
                      attractor=args.attractor,
                      bins_all=args.bins_all,
                      bins_per_run=args.bins_per_run,
                      t_max=args.t_max,
                      full_hist_only=args.full_hist_only)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
