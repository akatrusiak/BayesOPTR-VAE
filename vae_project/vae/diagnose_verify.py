"""
diagnose_verify.py — isolate the verify_dz.py injection bug.

Runs ONE test case end-to-end with maximum visibility:
  1. Loads the dataset, picks one run with a known-nontrivial δ_true.
  2. Dumps element name matching between VAE delta_names and data.dat.
  3. Injects δ_true directly (no VAE, no decoder).
  4. Before and after injection, dumps the data.dat element values so you
     can confirm the writes actually happened.
  5. Runs pyoptr.run, reads back fort.envelope and inputs.
  6. Compares resulting T_hat to T_true.
  7. Repeats step 3-6 with δ=0 (physical zero, fully aligned machine).
  8. Repeats step 3-6 with ALL misalignments doubled (gross perturbation — if the
     pipeline is sensitive to δ at all, this MUST differ from step 7).

Expected outputs if everything works:
  - Case (δ_true): T_hat ≈ T_true
  - Case (δ=0):     some T_hat
  - Case (δ*2):     different T_hat from δ=0

If all three give identical T_hat → injection is not reaching the simulator.

Run from the same directory you'd run verify_dz.py from. CLI is minimal.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml

# Repo root on sys.path (so `data` etc resolve)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .data import MultiFolderRunDataset


def _import_pipeline(pipeline_dir: Path):
    p = str(pipeline_dir.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    import generate_data as _gd
    return _gd


def dump_datadat_misalignments(optr_data, prefix: str, names_to_show):
    """
    Print the current value stored in data.dat for each name of interest.
    Uses the DataDat.data["elements"] list and matches on the trailing
    name embedded in the element's metadata — exactly how update_element
    looks up what to set.
    """
    elements = optr_data.data["elements"]
    # Build a name → (index, element) map using whatever attribute
    # update_element uses to match. Inspect the first element to figure
    # out the field name.
    sample = elements[0]
    print(f"\n[{prefix}] DataDat element schema sample: {sample}")
    # Try common field names
    name_field = None
    for candidate in ["name", "label", "id"]:
        if candidate in sample:
            name_field = candidate
            break
    if name_field is None:
        # Fall back: just print by index and let the user eyeball
        print(f"[{prefix}] WARNING: no name field found in element dict; dumping by index")
        for i, e in enumerate(elements):
            print(f"  [{i:3d}] {e}")
        return
    idx_by_name = {e.get(name_field): (i, e) for i, e in enumerate(elements)}
    print(f"[{prefix}] datadat values for names of interest (field={name_field!r}):")
    for n in names_to_show:
        hit = idx_by_name.get(n)
        if hit is None:
            print(f"  [MISS ] {n!r}  — NOT PRESENT in data.dat")
        else:
            idx, e = hit
            v = e.get("value", "?")
            print(f"  [ok   ] {n!r}  idx={idx}  value={v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", action="append", required=True)
    ap.add_argument("--verify_dir", required=True)
    ap.add_argument("--pipeline_dir", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--axis", default="verify")
    ap.add_argument("--run_idx", type=int, default=None,
                    help="Which run to test. If None, picks the run with "
                         "largest ||delta_true||.")
    ap.add_argument("--scale", type=float, default=2.0,
                    help="Multiplier applied to δ_true for the 'gross perturbation' "
                         "test. If 2 doesn't change T_hat, try 10 or 100.")
    args = ap.parse_args()

    # ---------- setup ----------
    gd = _import_pipeline(Path(args.pipeline_dir))

    # Load config from verify_dir (mirrors verify_dz.py convention)
    config_path = Path(args.verify_dir) / args.config
    if not config_path.exists():
        config_path = Path(args.config)  # fallback to cwd
    print(f"[setup] loading global config from {config_path}")
    with open(config_path) as f:
        global_config = yaml.safe_load(f)
    axis_spec = global_config["axes"][args.axis]

    print(f"[setup] init_axis({args.axis!r}) ...")
    axis_state = gd.init_axis(args.axis, axis_spec, global_config,
                              Path(args.verify_dir).resolve())
    if axis_state is None:
        sys.exit("init_axis returned None — abort.")
    optr_data = axis_state["optr_data"]

    # ---------- dataset ----------
    print(f"[setup] loading dataset from {args.data_dir} ...")
    ds = MultiFolderRunDataset(args.data_dir, standardize=True)
    print(ds.summary())

    # Pick a run
    if args.run_idx is None:
        norms = np.linalg.norm(ds._run_deltas, axis=1)
        run_idx = int(np.argmax(norms))
        print(f"[pick ] no --run_idx given; picking argmax ||δ|| → run_idx={run_idx} "
              f"with ||δ||={norms[run_idx]:.4f}")
    else:
        run_idx = args.run_idx
    label = ds.run_labels[run_idx]
    print(f"[pick ] run_idx={run_idx}  label={label}")

    delta_true = ds._run_deltas[run_idx].astype(np.float64)
    delta_names = list(ds.delta_names.tolist())
    rows = ds._run_row_idx[run_idx]
    t_true = float(ds.transmission[rows][-1])
    print(f"[pick ] T_true = {t_true:.6f}")
    print(f"[pick ] ||δ_true|| = {np.linalg.norm(delta_true):.4f}")
    nonzero = np.abs(delta_true) > 1e-10
    print(f"[pick ] δ_true nonzero in {int(nonzero.sum())}/{len(delta_true)} dims")

    # ---------- name matching ----------
    print("\n" + "=" * 70)
    print("NAME MATCHING CHECK")
    print("=" * 70)
    elements = optr_data.data["elements"]
    sample = elements[0]
    print(f"[datadat] first element in data.dat: {sample}")
    name_field = None
    for candidate in ["name", "label", "id"]:
        if candidate in sample:
            name_field = candidate
            break
    print(f"[datadat] inferred name field: {name_field!r}")

    if name_field is not None:
        datadat_names = [e.get(name_field) for e in elements]
        print(f"[datadat] {len(datadat_names)} elements in data.dat")
        print(f"[datadat] first 8: {datadat_names[:8]}")
        print(f"[datadat] last 8:  {datadat_names[-8:]}")

        in_both = [n for n in delta_names if n in datadat_names]
        only_vae = [n for n in delta_names if n not in datadat_names]
        only_dat = [n for n in datadat_names if n not in delta_names and "MISALIGN" in str(n).upper()]
        print(f"[match ] VAE delta_names in data.dat:     {len(in_both)}/{len(delta_names)}")
        if only_vae:
            print(f"[match ] !! VAE names NOT in data.dat ({len(only_vae)}): {only_vae}")
        if only_dat:
            print(f"[match ] !! data.dat misalignment elements NOT in VAE: {only_dat}")
        if not only_vae and not only_dat:
            print(f"[match ] OK: every VAE name matches a data.dat element")

    # ---------- three-case injection test ----------
    cases = [
        ("delta_true",     delta_true.copy()),
        ("delta_zero",     np.zeros_like(delta_true)),
        (f"delta_scaled_x{args.scale}", delta_true * args.scale),
    ]

    results = {}
    for case_name, delta in cases:
        print("\n" + "=" * 70)
        print(f"CASE: {case_name}   ||δ|| = {np.linalg.norm(delta):.4f}")
        print("=" * 70)

        # Reset to zero first (defensive, in case last run left state)
        for n in delta_names:
            try:
                optr_data.update_element(name=n, value=0.0)
            except ValueError:
                pass

        # Show BEFORE state for a few interesting names
        show_names = delta_names[:4] + delta_names[len(delta_names)//2:len(delta_names)//2+2]
        show_names = list(dict.fromkeys(show_names))  # dedup, preserve order
        dump_datadat_misalignments(optr_data, "BEFORE", show_names)

        # Inject
        mis_dict = OrderedDict(zip(delta_names, delta.tolist()))
        n_updated = 0
        n_missing = 0
        for name, v in mis_dict.items():
            try:
                optr_data.update_element(name=name, value=float(v))
                n_updated += 1
            except ValueError:
                n_missing += 1
        print(f"\n[inject] update_element calls: {n_updated} ok, {n_missing} raised ValueError (silently skipped)")

        dump_datadat_misalignments(optr_data, "AFTER ", show_names)

        # Run pyoptr via the same function verify_dz uses
        print(f"\n[run   ] calling run_sample_on_axis ...")
        t0 = time.time()
        result = gd.run_sample_on_axis(axis_state, mis_dict, global_config)
        dt = time.time() - t0
        print(f"[run   ] wall {dt:.1f}s")

        if "error" in result:
            print(f"[run   ] ERROR: {result['error']}")
            results[case_name] = None
            continue

        fc_t = float(result["fc_transmission"])
        trans_arr = np.asarray(result["transmissions"])
        inputs_arr = np.asarray(result["inputs"])
        n_env = int(result.get("n_envelopes", -1))

        print(f"[run   ] n_envelopes returned: {n_env}")
        print(f"[run   ] transmissions array shape: {trans_arr.shape}  "
              f"min={trans_arr.min():.4f} max={trans_arr.max():.4f} "
              f"last={trans_arr[-1]:.4f}")
        print(f"[run   ] fc_transmission = {fc_t:.6f}")
        print(f"[run   ] inputs shape: {inputs_arr.shape}, last row: {inputs_arr[-1] if inputs_arr.size else 'empty'}")

        results[case_name] = fc_t

    # ---------- summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"T_true        = {t_true:.6f}")
    for k, v in results.items():
        if v is None:
            print(f"T_hat [{k}] = FAILED")
        else:
            print(f"T_hat [{k}] = {v:.6f}   (Δ vs T_true = {v - t_true:+.6f})")

    # Interpretation hints
    vals = [v for v in results.values() if v is not None]
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        print(f"\nSpread across cases: {spread:.6f}")
        if spread < 1e-4:
            print("→ All three injection cases produced ~identical T_hat.")
            print("  This means δ is NOT reaching the simulator. Check:")
            print("  (a) pyoptr.run isn't reading from a cached/compiled data.dat")
            print("  (b) update_element is modifying a different object than pyoptr reads from")
            print("  (c) verification data.dat has MISALIGN* elements wired differently")
            print("      (e.g., opt-maxit tunes over them, wiping the injection)")
        elif abs(results["delta_true"] - t_true) < 0.01 and spread > 0.01:
            print("→ δ_true recovers T_true AND the pipeline is δ-sensitive.")
            print("  Verification pipeline is working correctly.")
        else:
            print("→ Partial signal. δ-sensitive but δ_true doesn't recover T_true.")
            print("  Possible: T_true came from a different setup than verify_dir,")
            print("  or TRANSOPTR optimization is tuning over misalignments themselves.")


if __name__ == "__main__":
    main()
