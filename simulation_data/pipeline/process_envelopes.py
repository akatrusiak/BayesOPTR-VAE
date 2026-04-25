"""
process_envelopes.py — Batch post-process existing fort.envelope files.

This is the script you run AFTER TRANSOPTR has already dumped ~500k 
fort.envelope files across multiple misalignment runs. It:

  1. Scans a directory tree for envelope files organized by misalignment run
  2. Loads and parses all envelopes (parallel I/O)
  3. Computes transmission for each envelope (vectorized)
  4. Loads the corresponding misalignment vectors
  5. Saves everything as a structured dataset for VAE training

Expected directory structure:
    <data_root>/
        run_000/
            misalignments.yaml    (or .npz)
            fort.envelope.00001
            fort.envelope.00002
            ...
        run_001/
            misalignments.yaml
            fort.envelope.00001
            ...
        ...
    
    OR flat structure:
    <data_root>/
        misalignment_000.yaml
        envelopes_000/
            fort.envelope.00001
            ...

Usage:
    python process_envelopes.py \
        --data_root /path/to/transoptr_outputs \
        --config hebt2.yaml \
        --syf sy.f \
        --output vae_training_data.npz \
        --fc_name DRA:FC1 \
        --n_workers 8
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from envelope_parser import parse_single_envelope, COLS_NEEDED, N_HEADER_LINES
from transmission import (
    build_aperture_array_from_syf,
    compute_transmissions_vectorized,
    extract_fc_transmission,
    _erf_transmission_1d,
    _propagated_transmission,
)


def find_envelope_files(run_dir: Path) -> list[Path]:
    """Find all fort.envelope* files in a directory, sorted."""
    patterns = ["fort.envelope.*", "fort.envelope"]
    files = []
    for pat in patterns:
        files.extend(run_dir.glob(pat))
    # Filter out the base file if numbered versions exist
    numbered = [f for f in files if f.suffix and f.suffix[1:].isdigit()]
    if numbered:
        return sorted(numbered, key=lambda f: int(f.suffix[1:]))
    return sorted(files)


def discover_runs(data_root: Path) -> list[dict]:
    """
    Discover misalignment runs in the data directory.
    
    Returns list of dicts with:
        'run_dir': Path to directory containing envelopes
        'misalignment_file': Path to misalignment file (if found)
        'envelope_files': list of Path to envelope files
    """
    runs = []
    
    # Pattern 1: subdirectories named run_NNN
    run_dirs = sorted(data_root.glob("run_*"))
    if run_dirs:
        for rd in run_dirs:
            if not rd.is_dir():
                continue
            envs = find_envelope_files(rd)
            if not envs:
                continue
            mis_file = None
            for name in ["misalignments.yaml", "misalignments.npz", "Misalignments.yaml"]:
                candidate = rd / name
                if candidate.exists():
                    mis_file = candidate
                    break
            runs.append({
                "run_dir": rd,
                "misalignment_file": mis_file,
                "envelope_files": envs,
            })
        return runs
    
    # Pattern 2: flat structure with envelope subdirs
    env_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and "envelope" in d.name.lower()])
    if env_dirs:
        for ed in env_dirs:
            envs = find_envelope_files(ed)
            if not envs:
                continue
            # Look for matching misalignment file
            idx = ed.name.split("_")[-1] if "_" in ed.name else ""
            mis_file = None
            for name in [f"misalignment_{idx}.yaml", f"misalignments_{idx}.yaml"]:
                candidate = data_root / name
                if candidate.exists():
                    mis_file = candidate
                    break
            runs.append({
                "run_dir": ed,
                "misalignment_file": mis_file,
                "envelope_files": envs,
            })
        return runs
    
    # Pattern 3: all envelopes in one directory (single misalignment run)
    envs = find_envelope_files(data_root)
    if envs:
        mis_file = None
        for name in ["misalignments.yaml", "Misalignments.yaml", "misalignments.npz"]:
            candidate = data_root / name
            if candidate.exists():
                mis_file = candidate
                break
        runs.append({
            "run_dir": data_root,
            "misalignment_file": mis_file,
            "envelope_files": envs,
        })
    
    return runs


def load_misalignments(mis_file: Path) -> dict:
    """Load misalignment dict from YAML or NPZ."""
    if mis_file is None:
        return {}
    if mis_file.suffix == ".yaml":
        with open(mis_file) as f:
            return yaml.safe_load(f) or {}
    elif mis_file.suffix == ".npz":
        data = np.load(mis_file, allow_pickle=True)
        return dict(data)
    return {}


def parse_envelope_batch(
    filepaths: list[Path],
    n_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a batch of envelope files, returning pre-allocated arrays.
    
    Returns:
        s:     (N_rows,)
        x_env: (N_files, N_rows)
        y_env: (N_files, N_rows)
        x_cm:  (N_files, N_rows)
        y_cm:  (N_files, N_rows)
    """
    N = len(filepaths)
    
    # Parse first to get dimensions
    first = parse_single_envelope(filepaths[0])
    N_rows = first.shape[0]
    
    s = first[:, 0]
    x_env = np.empty((N, N_rows), dtype=np.float64)
    y_env = np.empty((N, N_rows), dtype=np.float64)
    x_cm = np.empty((N, N_rows), dtype=np.float64)
    y_cm = np.empty((N, N_rows), dtype=np.float64)
    
    x_env[0] = first[:, 1]
    y_env[0] = first[:, 2]
    x_cm[0] = first[:, 3]
    y_cm[0] = first[:, 4]
    
    if N == 1:
        return s, x_env, y_env, x_cm, y_cm
    
    # Parse remaining in parallel
    def _parse_one(args):
        idx, fp = args
        try:
            data = parse_single_envelope(fp)
            return idx, data
        except Exception as e:
            return idx, None
    
    actual_workers = min(n_workers, N - 1)
    args_list = [(i, filepaths[i]) for i in range(1, N)]
    
    if actual_workers <= 1:
        for idx, fp in args_list:
            _, data = _parse_one((idx, fp))
            if data is not None and data.shape[0] == N_rows:
                x_env[idx] = data[:, 1]
                y_env[idx] = data[:, 2]
                x_cm[idx] = data[:, 3]
                y_cm[idx] = data[:, 4]
    else:
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = list(pool.map(_parse_one, args_list))
            for idx, data in futures:
                if data is not None and data.shape[0] == N_rows:
                    x_env[idx] = data[:, 1]
                    y_env[idx] = data[:, 2]
                    x_cm[idx] = data[:, 3]
                    y_cm[idx] = data[:, 4]
    
    return s, x_env, y_env, x_cm, y_cm


def process_single_run(
    run_info: dict,
    slits_array: np.ndarray,
    fc_index: int,
    s_reference: np.ndarray,
    n_workers: int,
) -> dict:
    """
    Process all envelopes from one misalignment run.
    
    Returns:
        dict with 'transmissions' (N_steps,) and 'misalignment_dict'
    """
    envelope_files = run_info["envelope_files"]
    
    # Parse envelopes
    s, x_env, y_env, x_cm, y_cm = parse_envelope_batch(envelope_files, n_workers)
    
    # Verify s-grid matches reference
    if s_reference is not None and not np.allclose(s, s_reference, atol=0.01):
        print(f"  [WARNING] s-grid mismatch in {run_info['run_dir']}")
    
    # Compute transmissions (vectorized over all envelopes in this run)
    transmissions = compute_transmissions_vectorized(
        x_env, y_env, x_cm, y_cm, slits_array, fc_index=fc_index,
    )
    
    # Load misalignments
    mis_dict = load_misalignments(run_info["misalignment_file"])
    
    return {
        "transmissions": transmissions,
        "misalignment_dict": mis_dict,
        "n_envelopes": len(envelope_files),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch process fort.envelope files into VAE training data"
    )
    parser.add_argument("--data_root", required=True, help="Root directory with envelope files")
    parser.add_argument("--config", required=True, help="Beamline YAML config")
    parser.add_argument("--syf", required=True, help="Path to sy.f file")
    parser.add_argument("--output", default="vae_training_data.npz", help="Output file")
    parser.add_argument("--fc_name", default="DRA:FC1", help="FC measurement device name")
    parser.add_argument("--n_workers", type=int, default=4, help="Parallel workers for I/O")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Process envelopes in chunks of this size (memory management)")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    wall_width = config.get("beamline_wall_width", 2.54)
    offset = config.get("offset", 1.0)
    
    # Get FC location from config
    fc_loc = config["elements"]["fc"][args.fc_name]["loc"]
    print(f"FC '{args.fc_name}' at s = {fc_loc} cm")
    
    # Discover runs
    runs = discover_runs(data_root)
    print(f"Found {len(runs)} misalignment run(s)")
    total_envelopes = sum(len(r["envelope_files"]) for r in runs)
    print(f"Total envelope files: {total_envelopes}")
    
    if len(runs) == 0:
        print("No runs found. Check directory structure.")
        sys.exit(1)
    
    # Parse one envelope to get the s-grid
    first_file = runs[0]["envelope_files"][0]
    first_data = parse_single_envelope(first_file)
    s_reference = first_data[:, 0]
    print(f"Beamline grid: {len(s_reference)} points, s = [{s_reference[0]:.1f}, {s_reference[-1]:.1f}] cm")
    
    # Build aperture array from sy.f (parse once, use for all)
    quad_apertures = []
    for name, quad in config.get("elements", {}).get("quad", {}).items():
        if "aperture" in quad:
            quad_apertures.append({"loc": quad["loc"], "aperture": quad["aperture"]})
    
    slits_array = build_aperture_array_from_syf(
        s_reference, args.syf,
        wall_width=wall_width, offset=offset,
        quad_apertures=quad_apertures,
    )
    
    n_apertures = np.sum(slits_array < wall_width)
    print(f"Aperture array: {n_apertures} constraining positions (of {len(slits_array)})")
    
    # Pre-compute FC index
    fc_index = extract_fc_transmission(s_reference, fc_loc, offset)
    print(f"FC index: {fc_index} (s = {s_reference[fc_index]:.3f} cm)")
    
    # Process all runs
    all_transmissions = []
    all_misalignments = []
    
    t_total = time.time()
    for i, run_info in enumerate(runs):
        n_env = len(run_info["envelope_files"])
        t0 = time.time()
        print(f"\nRun {i+1}/{len(runs)}: {run_info['run_dir'].name} ({n_env} envelopes)...", flush=True)
        
        # For very large runs, process in chunks
        if n_env > args.chunk_size:
            chunk_transmissions = []
            for chunk_start in range(0, n_env, args.chunk_size):
                chunk_end = min(chunk_start + args.chunk_size, n_env)
                chunk_files = run_info["envelope_files"][chunk_start:chunk_end]
                
                s, x_env, y_env, x_cm, y_cm = parse_envelope_batch(
                    chunk_files, args.n_workers
                )
                
                t_chunk = compute_transmissions_vectorized(
                    x_env, y_env, x_cm, y_cm, slits_array, fc_index=fc_index,
                )
                chunk_transmissions.append(t_chunk)
                
                print(f"  Chunk [{chunk_start}:{chunk_end}] done, "
                      f"mean T = {t_chunk.mean():.4f}")
                
                # Free memory
                del x_env, y_env, x_cm, y_cm
            
            transmissions = np.concatenate(chunk_transmissions)
        else:
            result = process_single_run(
                run_info, slits_array, fc_index, s_reference, args.n_workers,
            )
            transmissions = result["transmissions"]
        
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s. Transmissions: "
              f"min={transmissions.min():.4f}, mean={transmissions.mean():.4f}, "
              f"max={transmissions.max():.4f}")
        
        all_transmissions.append(transmissions)
        
        mis_dict = load_misalignments(run_info["misalignment_file"])
        if isinstance(mis_dict, dict):
            all_misalignments.append(mis_dict)
        else:
            all_misalignments.append({})
    
    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Total processing time: {total_time:.1f}s for {total_envelopes} envelopes")
    print(f"Throughput: {total_envelopes / total_time:.0f} envelopes/sec")
    
    # Build output dataset
    # Misalignment vectors (same for all envelopes in a run)
    if all_misalignments and isinstance(all_misalignments[0], dict) and all_misalignments[0]:
        mis_names = list(all_misalignments[0].keys())
        mis_matrix = np.array([
            [md.get(name, 0.0) for name in mis_names]
            for md in all_misalignments
        ])
    else:
        mis_names = []
        mis_matrix = np.array([])
    
    # Save
    output_path = Path(args.output)
    np.savez_compressed(
        output_path,
        # Per-run data
        misalignment_vectors=mis_matrix,           # (N_runs, D_delta)
        misalignment_names=np.array(mis_names),    # (D_delta,) string labels
        # Per-envelope data (ragged — different runs may have different numbers of steps)
        transmissions=np.array(all_transmissions, dtype=object),  # list of (N_steps_i,)
        # Metadata
        s_grid=s_reference,
        slits_array=slits_array,
        fc_index=np.array(fc_index),
        fc_loc=np.array(fc_loc),
        wall_width=np.array(wall_width),
    )
    
    print(f"\nSaved to {output_path}")
    print(f"  Misalignment matrix: {mis_matrix.shape}")
    print(f"  Runs: {len(all_transmissions)}")
    print(f"  Total data points: {sum(len(t) for t in all_transmissions)}")


if __name__ == "__main__":
    main()
