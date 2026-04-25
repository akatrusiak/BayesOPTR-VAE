#!/usr/bin/env python3
"""
Driver for a multi-seed d_z scan over the VAE training script.

Runs train.py for each (d_z, seed) combination, skipping runs whose output
directory already contains a history.json (so you can Ctrl-C and resume).

Example:
    python run_sweep.py \
        --data_dir ../../simulation_data/data/output_good_4/ \
        --train_script train.py \
        --out_root runs/sweep \
        --d_latents 5 8 10 12 \
        --seeds 0 1 2 \
        --d_hidden 64

Everything after --extra is passed through to train.py unchanged, so you can
add flags like --epochs 300 --weight_decay 1e-4 without editing this script.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_script", default="train.py",
                   help="Path to your train.py (default: train.py in cwd)")
    p.add_argument("--data_dir", action="append", required=True,
                   help="Passed through to train.py. Repeatable, same as train.py's --data_dir.")
    p.add_argument("--out_root", default="runs/sweep",
                   help="Parent dir for all sweep runs. Each run goes in "
                        "{out_root}/vae_dz{DZ}_s{SEED}/")
    p.add_argument("--d_latents", type=int, nargs="+", default=[5, 8, 10, 12],
                   help="List of d_z values to sweep.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="List of seed values. Defaults to 3 seeds.")
    p.add_argument("--d_hidden", type=int, default=64,
                   help="Passed through as --d_hidden to train.py.")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to use (default: current).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running them.")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is passed through to train.py. "
                        "Example: --extra --epochs 300 --weight_decay 1e-3")
    return p.parse_args()


def run_one(args, dz: int, seed: int) -> str:
    """Run a single (d_z, seed) combination. Returns 'skipped', 'ok', or 'failed'."""
    out_dir = Path(args.out_root) / f"vae_dz{dz}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already complete
    if (out_dir / "history.json").exists() and (out_dir / "best.pt").exists():
        print(f"[skip] {out_dir} already has history.json + best.pt")
        return "skipped"

    cmd = [args.python, args.train_script]
    for dd in args.data_dir:
        cmd += ["--data_dir", dd]
    cmd += [
        "--d_latent", str(dz),
        "--d_hidden", str(args.d_hidden),
        "--seed", str(seed),
        "--out_dir", str(out_dir),
    ]
    cmd += args.extra

    print(f"\n[run ] dz={dz} seed={seed} -> {out_dir}")
    print("       " + " ".join(cmd))
    if args.dry_run:
        return "skipped"

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAIL] dz={dz} seed={seed} exited with code {result.returncode}")
        return "failed"
    return "ok"


def main():
    args = parse_args()
    summary = {"ok": 0, "skipped": 0, "failed": 0}
    failed = []

    total = len(args.d_latents) * len(args.seeds)
    i = 0
    for dz in args.d_latents:
        for seed in args.seeds:
            i += 1
            print(f"\n{'='*70}\n[{i}/{total}] dz={dz}, seed={seed}\n{'='*70}")
            status = run_one(args, dz, seed)
            summary[status] += 1
            if status == "failed":
                failed.append((dz, seed))

    print(f"\n{'='*70}\nSWEEP COMPLETE: {summary['ok']} ok, "
          f"{summary['skipped']} skipped, {summary['failed']} failed")
    if failed:
        print("  failed runs:")
        for dz, seed in failed:
            print(f"    dz={dz} seed={seed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
