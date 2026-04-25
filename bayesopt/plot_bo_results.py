"""
plot_bo_results.py — aggregate and visualize run_bo_experiment.py output.

Reads the manifest.csv and the per-run training_set.csv files, produces:
  (1) mean ± band BO trajectories over iterations, by condition
  (2) paired final-T scatter (surrogate_mean vs default), per val run
  (3) Wilcoxon signed-rank test on paired final-T differences

Usage
-----
python plot_bo_results.py --exp_dir runs/bo_experiment
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_training_set(run_dir: Path):
    """Return per-iteration y array for this BO run, or None on failure."""
    p = run_dir / "training_set.csv"
    if not p.exists():
        return None
    ys = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_str = row.get("train_x", "")
            y_str = row.get("train_y", "")
            # Skip the stashed-names row
            if "'HEBT2" in x_str or '"HEBT2' in x_str:
                continue
            try:
                ys.append(float(y_str))
            except Exception:
                continue
    return np.array(ys) if ys else None


def running_max(y: np.ndarray) -> np.ndarray:
    """Return the running max over y — the standard BO convergence curve."""
    out = np.empty_like(y)
    m = -np.inf
    for i, v in enumerate(y):
        if v > m:
            m = v
        out[i] = m
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp_dir", required=True, type=Path,
                   help="Directory containing manifest.csv (output of run_bo_experiment.py).")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Where to write plots. Default: <exp_dir>/plots/")
    p.add_argument("--conditions", nargs="+",
                   default=["default", "surrogate_mean"],
                   help="Condition names in the order to plot.")
    args = p.parse_args()

    exp_dir: Path = args.exp_dir.resolve()
    out_dir = args.out_dir or (exp_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = exp_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"No manifest at {manifest}")

    # ── load manifest ──────────────────────────────────────────
    rows = []
    with open(manifest) as f:
        for r in csv.DictReader(f):
            r["val_run_idx"] = int(r["val_run_idx"])
            r["seed"] = int(r["seed"])
            r["t_peak_observed"] = float(r["t_peak_observed"])
            r["t_final_bois"] = float(r["t_final_bois"]) if r["t_final_bois"] else None
            rows.append(r)
    print(f"[load] {len(rows)} rows from {manifest}")

    # ── read per-run training_set.csv for trajectories ─────────
    trajectories = {}  # (val_run_idx, condition, seed) -> np.ndarray of running max
    for r in rows:
        y = read_training_set(Path(r["run_dir"]))
        if y is None:
            print(f"  [warn] no training_set.csv for {r['run_dir']}")
            continue
        trajectories[(r["val_run_idx"], r["condition"], r["seed"])] = running_max(y)

    # ── figure 1: mean ± band trajectories per condition ───────
    cond_to_color = {"default": "steelblue", "surrogate_mean": "crimson"}
    cond_to_label = {"default": "default prior (ConstantMean)",
                     "surrogate_mean": "surrogate prior"}

    fig, ax = plt.subplots(figsize=(9, 5))
    # Determine common length — trim to min across all runs
    lengths = [len(t) for t in trajectories.values()]
    if not lengths:
        print("No trajectories loaded. Exiting.")
        return
    min_len = min(lengths)
    print(f"[plot ] min trajectory length across runs: {min_len}")

    for cond in args.conditions:
        traj_matrix = np.stack([
            t[:min_len]
            for (ri, c, s), t in trajectories.items()
            if c == cond
        ])
        if traj_matrix.size == 0:
            continue
        iters = np.arange(min_len)
        mu = traj_matrix.mean(axis=0)
        sd = traj_matrix.std(axis=0, ddof=1)
        # lo, hi = np.percentile(traj_matrix, [25, 75], axis=0)
        ax.plot(iters, mu, color=cond_to_color.get(cond, None),
                lw=2, label=f"{cond_to_label.get(cond, cond)} (n={len(traj_matrix)})")
        # ax.fill_between(iters, lo, hi, color=cond_to_color.get(cond, None),
        #                 alpha=0.2, linewidth=0)
        ax.fill_between(iters, mu - sd, mu + sd, color=cond_to_color.get(cond, None),
                        alpha=0.2, linewidth=0)

    ax.set_xlabel("BO iteration (including initial sampling)")
    ax.set_ylabel("Running max transmission")
    ax.set_title("BO trajectories, mean with standard deviation band across val runs")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bo_trajectories.png", dpi=120, bbox_inches="tight")
    print(f"[save] {out_dir / 'bo_trajectories.png'}")

    # ── figure 2: paired final-T scatter ───────────────────────
    # Aggregate by (val_run, seed), pair across conditions.
    t_final = {}
    for r in rows:
        key = (r["val_run_idx"], r["seed"])
        t_final.setdefault(key, {})[r["condition"]] = r["t_final_bois"]

    paired = [(t_final[k].get("default"), t_final[k].get("surrogate_mean"),
               k[0], k[1])
              for k in t_final
              if t_final[k].get("default") is not None
              and t_final[k].get("surrogate_mean") is not None]

    if not paired:
        print("No paired final-T values; skipping scatter.")
        return

    default_final = np.array([p[0] for p in paired])
    surrogate_final = np.array([p[1] for p in paired])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(default_final, surrogate_final, s=30, alpha=0.75, color="purple")
    lim = max(default_final.max(), surrogate_final.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlabel("final T — default prior")
    ax.set_ylabel("final T — surrogate prior")
    ax.set_title(f"Final T after BO, paired per val run × seed  (n={len(paired)})")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend()
    ax.grid(alpha=0.3)

    # Paired Wilcoxon
    try:
        from scipy.stats import wilcoxon
        diffs = surrogate_final - default_final
        diffs_nonzero = diffs[diffs != 0]
        if len(diffs_nonzero) >= 3:
            w_p = wilcoxon(diffs_nonzero).pvalue
            ax.text(0.03, 0.97,
                    f"mean(surr - default) = {diffs.mean():+.4f}\n"
                    f"median = {np.median(diffs):+.4f}\n"
                    f"Wilcoxon p = {w_p:.4f}\n"
                    f"frac surrogate > default = {(diffs>0).sum()/len(diffs)*100:.1f}%",
                    transform=ax.transAxes, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                    fontsize=9)
    except ImportError:
        pass

    fig.tight_layout()
    fig.savefig(out_dir / "final_t_paired.png", dpi=120, bbox_inches="tight")
    print(f"[save] {out_dir / 'final_t_paired.png'}")

    # ── numerical summary printed to stdout ───────────────────
    print("\n─── Summary ───")
    for cond in args.conditions:
        finals = [r["t_final_bois"] for r in rows
                  if r["condition"] == cond and r["t_final_bois"] is not None]
        if finals:
            print(f"{cond:20s}  n={len(finals)}  "
                  f"mean={np.mean(finals):.4f}  median={np.median(finals):.4f}  "
                  f"std={np.std(finals):.4f}")


if __name__ == "__main__":
    main()
