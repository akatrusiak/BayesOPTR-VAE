"""
run_bo_experiment.py — orchestrator for the surrogate-as-mean vs default-mean
BO comparison experiment.

Pipeline
--------
1. Recover the same 144 val runs that seed=0 VAE/surrogate training used,
   via recover_split(720, val_frac=0.2, seed=0).
2. Rank them by observed T_peak, take the top 20.
3. For each val run, write a Misalignments.yaml containing its δ_true
   (in the BOIS format, 27 keys including MISALIGN* and HEBT2:*MISALIGN*).
4. For each (val_run, condition) in the experiment grid, invoke BOIS:
       python main.py hydra.run.dir=<out>/<run_tag> \
           misalignments=<yaml_path> \
           sim.random_seed=<seed> \
           iterations=60 initial_points=20 \
           [mean_function=null  OR  mean_function.surrogate_path=... + fields]
5. Aggregate results into a manifest CSV (one row per BO run):
       val_run_idx, val_run_label, condition, seed, t_peak_true_observed,
       t_final, t_max_over_bo, wall_seconds, run_dir

Re-entrance
-----------
If `<out>/<run_tag>/optimal_input.yaml` already exists, that BO run is
skipped. This lets you restart after interrupts without re-doing completed
work.

Usage
-----
python run_bo_experiment.py \
    --bois_dir /mnt/c/.../bayesoptr_research/bayesopt \
    --surrogate /abs/path/to/surrogate_best.pt \
    --data_dir ../../simulation_data/data/output_good_4 \
    --data_dir ../../simulation_data/data/output_good_5 \
    --data_dir ../../simulation_data/data/output_good_6 \
    --out_dir runs/bo_experiment \
    --n_val_runs 20 \
    --seed 0 \
    --iterations 60 --initial_points 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

# Import pattern matching compare_latent_signal / verify_dz
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vae_project.vae.data import MultiFolderRunDataset  # type: ignore


# ═════════════════════════════════════════════════════════════════════
# Split recovery — identical to compare_latent_signal.recover_split
# ═════════════════════════════════════════════════════════════════════

def recover_split(n_items: int, val_frac: float = 0.2, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n_items)))
    val_idx = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    return train_idx, val_idx


# ═════════════════════════════════════════════════════════════════════
# Val-run selection
# ═════════════════════════════════════════════════════════════════════

def select_top_k_val_runs(ds, val_indices, k: int):
    """
    Rank val runs by observed per-run T_peak, return the top k indices.
    """
    scored = []
    for ri in val_indices:
        rows = ds._run_row_idx[ri]
        t_peak = float(np.max(ds.transmission[rows]))
        scored.append((t_peak, ri))
    scored.sort(reverse=True)
    return [ri for _, ri in scored[:k]], [tp for tp, _ in scored[:k]]


# ═════════════════════════════════════════════════════════════════════
# Misalignments.yaml writer
# ═════════════════════════════════════════════════════════════════════

def write_misalignments_yaml(delta_phys, delta_names, out_path: Path):
    """
    Write a BOIS-format misalignments yaml from a physical-units δ vector.

    Parameters
    ----------
    delta_phys : np.ndarray, shape (d_delta,)
        Physical-units δ values, in `delta_names` order.
    delta_names : list[str]
        Names in the same order as delta_phys.
    out_path : Path
        Where to write. Parent directory must exist.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {n: float(v) for n, v in zip(delta_names, delta_phys)}
    # Preserve insertion order in the emitted yaml so it reads naturally
    with open(out_path, "w") as f:
        for n, v in mapping.items():
            f.write(f"{n}: {v}\n")


# ═════════════════════════════════════════════════════════════════════
# BOIS invocation
# ═════════════════════════════════════════════════════════════════════

def build_bois_command(
    bois_dir: Path,
    run_dir: Path,
    misalignments_yaml: Path,
    seed: int,
    iterations: int,
    initial_points: int,
    mean_cfg: dict | None,
) -> list[str]:
    """
    Build the argv list for invoking BOIS. Returns a list suitable for
    subprocess.run(args=...).

    BOIS expects:
      python main.py hydra.run.dir=<...> misalignments=<...> \
        sim.random_seed=N iterations=... initial_points=...
        [mean_function.surrogate_path=...  mean_function.delta_yaml=...  ...]
    """
    cmd: list[str] = [
        sys.executable,
        str(bois_dir / "main.py"),
        f"hydra.run.dir={run_dir}",
        f"misalignments={misalignments_yaml}",
        f"sim.random_seed={seed}",
        f"iterations={iterations}",
        f"initial_points={initial_points}",
    ]

    if mean_cfg is None:
        cmd.append("mean_function=null")
    else:
        # Hydra dot-syntax. List values need brackets and commas, but
        # safer to pass them via +mean_function.delta_names=... with
        # properly quoted list literal.
        cmd.append(f"mean_function.surrogate_path={mean_cfg['surrogate_path']}")
        cmd.append(f"mean_function.delta_yaml={mean_cfg['delta_yaml']}")
        cmd.append(
            f"mean_function.delta_names={json_list_inline(mean_cfg['delta_names'])}"
        )
        cmd.append(
            f"mean_function.surrogate_input_names="
            f"{json_list_inline(mean_cfg['surrogate_input_names'])}"
        )
        cmd.append(f"mean_function.device={mean_cfg.get('device','cpu')}")

    return cmd


def json_list_inline(items: Sequence[str]) -> str:
    """Render a Python list of strings as a Hydra-friendly inline list."""
    # Hydra accepts JSON-style bracketed lists on the command line.
    return "[" + ",".join(items) + "]"


def run_bois_once(
    cmd: list[str],
    log_path: Path,
    env: dict | None = None,
) -> dict:
    """Run BOIS and capture stdout/stderr to log_path. Return a small result dict."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write("COMMAND:\n" + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    wall = time.time() - t0
    return {"returncode": proc.returncode, "wall_seconds": wall}


# ═════════════════════════════════════════════════════════════════════
# Results collection
# ═════════════════════════════════════════════════════════════════════

def read_optimal_input(run_dir: Path) -> dict | None:
    """Return the optimal_input.yaml contents, or None if missing/malformed."""
    p = run_dir / "optimal_input.yaml"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def read_training_set(run_dir: Path):
    """
    Return (train_x list, train_y list) from BOIS's training_set.csv, or
    (None, None) if unavailable. BOIS's training_set.csv last row is a
    stashed name list — drop it.
    """
    p = run_dir / "training_set.csv"
    if not p.exists():
        return None, None
    train_x, train_y = [], []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_str = row["train_x"]
            y_str = row["train_y"]
            # Skip the name-stash row whose train_x is a list of strings
            if "'" in x_str or '"HEBT2' in x_str:
                continue
            try:
                x_vec = eval(x_str)  # safe: BOIS writes plain python lists
                y_val = float(y_str)
                train_x.append(x_vec)
                train_y.append(y_val)
            except Exception:
                continue
    return train_x, train_y


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bois_dir", required=True, type=Path,
                   help="Path to BOIS repo root (contains main.py).")
    p.add_argument("--surrogate", required=True, type=Path,
                   help="Path to surrogate best.pt (absolute path preferred).")
    p.add_argument("--data_dir", action="append", required=True, dest="data_dirs",
                   help="Repeat for each output_good_* directory, in the same "
                        "order as was used to train the surrogate.")
    p.add_argument("--out_dir", default="runs/bo_experiment", type=Path)

    p.add_argument("--n_val_runs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="Random seeds for sim.random_seed. Default: [0].")
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--initial_points", type=int, default=20)

    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0,
                   help="Seed used to recover the val split. Must match the "
                        "seed the surrogate was trained at.")

    p.add_argument("--dry_run", action="store_true",
                   help="Print the command matrix but don't invoke BOIS.")
    args = p.parse_args()

    # ── load dataset ────────────────────────────────────────────
    print(f"[load] dataset: {args.data_dirs}")
    ds = MultiFolderRunDataset(args.data_dirs, standardize=False)
    print(ds.summary())
    delta_names = list(ds.delta_names)
    surrogate_input_names = list(ds.input_names)

    # ── recover val split and rank ──────────────────────────────
    _, val_idx_all = recover_split(len(ds), val_frac=args.val_frac,
                                    seed=args.split_seed)
    print(f"[split] {len(val_idx_all)} val runs from seed={args.split_seed}")
    selected_val, selected_t_peaks = select_top_k_val_runs(
        ds, val_idx_all, args.n_val_runs,
    )
    print(f"[select] top {args.n_val_runs} val runs by T_peak:")
    for ri, tp in zip(selected_val, selected_t_peaks):
        label = ds.run_labels[ri]
        print(f"  run_idx={ri:4d}  T_peak={tp:.3f}  "
              f"axis={label.axis}  folder={Path(label.folder).name}  "
              f"local_id={label.local_id}")

    # ── build per-val-run misalignments.yaml files ──────────────
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas_dir = out_dir / "misalignments_yamls"
    yaml_paths = {}
    for ri in selected_val:
        dp = deltas_dir / f"run_{ri}.yaml"
        write_misalignments_yaml(ds._run_deltas[ri], delta_names, dp)
        yaml_paths[ri] = dp

    # ── env for BOIS: make `surrogate` importable ───────────────
    # BOIS needs to import the surrogate package when mean_function is set.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH", "")])
    )

    # ── experiment grid ─────────────────────────────────────────
    surrogate_path_abs = args.surrogate.resolve()
    conditions = ["default", "surrogate_mean"]
    manifest_rows = []

    grid = [(ri, cond, s)
            for ri in selected_val
            for cond in conditions
            for s in args.seeds]
    print(f"\n[grid ] {len(grid)} BO runs "
          f"({args.n_val_runs} val runs × {len(conditions)} conditions × "
          f"{len(args.seeds)} seeds)")

    for (ri, cond, seed) in grid:
        run_tag = f"run_{ri}__{cond}__seed{seed}"
        run_dir = (out_dir / "bo_runs" / run_tag).resolve()

        # Skip if already complete
        if (run_dir / "optimal_input.yaml").exists():
            print(f"[skip ] {run_tag} (already complete)")
            opt = read_optimal_input(run_dir)
            manifest_rows.append(manifest_row(
                ri, cond, seed, run_dir, opt,
                t_peak_observed=ds_run_t_peak(ds, ri),
                wall_seconds=None,
            ))
            continue

        # Build command
        mean_cfg = None
        if cond == "surrogate_mean":
            mean_cfg = {
                "surrogate_path": str(surrogate_path_abs),
                "delta_yaml": str(yaml_paths[ri]),
                "delta_names": delta_names,
                "surrogate_input_names": surrogate_input_names,
            }
        cmd = build_bois_command(
            bois_dir=args.bois_dir.resolve(),
            run_dir=run_dir,
            misalignments_yaml=yaml_paths[ri].resolve(),
            seed=seed,
            iterations=args.iterations,
            initial_points=args.initial_points,
            mean_cfg=mean_cfg,
        )

        print(f"\n[run  ] {run_tag}")
        print("        " + " ".join(shlex.quote(c) for c in cmd))

        if args.dry_run:
            continue

        log_path = run_dir / "driver.log"
        result = run_bois_once(cmd, log_path, env=env)
        print(f"        wall={result['wall_seconds']:.1f}s  rc={result['returncode']}")

        opt = read_optimal_input(run_dir)
        if opt is None:
            print("        ✗ no optimal_input.yaml produced — BOIS likely failed, "
                  "check driver.log")

        manifest_rows.append(manifest_row(
            ri, cond, seed, run_dir, opt,
            t_peak_observed=ds_run_t_peak(ds, ri),
            wall_seconds=result["wall_seconds"],
        ))

    # ── write manifest ──────────────────────────────────────────
    if args.dry_run:
        print("\n[dry ] done (no manifest written)")
        return

    manifest_path = out_dir / "manifest.csv"
    fieldnames = ["val_run_idx", "condition", "seed", "t_peak_observed",
                  "t_final_bois", "wall_seconds", "run_dir"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)
    print(f"\n[done ] {len(manifest_rows)} rows → {manifest_path}")


def ds_run_t_peak(ds, ri: int) -> float:
    rows = ds._run_row_idx[ri]
    return float(np.max(ds.transmission[rows]))


def manifest_row(ri, cond, seed, run_dir, opt, t_peak_observed, wall_seconds):
    t_final = None
    if opt is not None:
        # BOIS writes the best-seen transmission under 'objective_best' in the yaml.
        t_final = opt.get("objective_best")
        if t_final is None and "transmission" in opt:
            # Fallback to the explicit 'transmission' field if present
            t_final = opt["transmission"]
    return {
        "val_run_idx": ri,
        "condition": cond,
        "seed": seed,
        "t_peak_observed": t_peak_observed,
        "t_final_bois": t_final,
        "wall_seconds": wall_seconds,
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    main()
