"""
envelope_parser.py — Fast batch parsing of TRANSOPTR fort.envelope files.

Parses only the 5 columns needed for transmission calculation:
  col 0:  s          (position along beamline, cm)
  col 1:  x-envelope (2-sigma x beam size, cm)
  col 3:  y-envelope (2-sigma y beam size, cm)
  col 70: FS-x       (x centroid offset, cm)
  col 72: FS-y       (y centroid offset, cm)

Optimized for parsing ~500k files:
  - Uses multiprocessing for I/O-bound file reading
  - Pre-allocates output arrays
  - Skips unused columns during parsing
"""

import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union
import os


# Column indices in fort.envelope (0-indexed, tab-separated)
COLS_NEEDED = [0, 1, 2, 3, 4]
COL_NAMES = ["s", "x-envelope", "y-envelope", "x-centroid", "y-centroid"]
N_HEADER_LINES = 1


def parse_single_envelope(filepath: Union[str, Path], expected_rows: int = None) -> np.ndarray:
    """
    Parse a single fort.envelope file, extracting only the needed columns.
    
    Returns:
        array of shape (N_rows, 5) with columns [s, x_env, y_env, FS_x, FS_y]
    """
    rows = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i < N_HEADER_LINES:
                continue
            # Split on whitespace (tabs + spaces)
            fields = line.split()
            if len(fields) < 5:  # need all 5 columns
                continue
            try:
                row = [float(fields[c]) for c in COLS_NEEDED]
                rows.append(row)
            except (ValueError, IndexError):
                continue
    
    data = np.array(rows, dtype=np.float64)
    
    if expected_rows is not None and data.shape[0] != expected_rows:
        raise ValueError(
            f"Envelope file '{filepath}' has {data.shape[0]} rows, expected {expected_rows}. "
            f"File may be incomplete or truncated."
        )
    
    return data


def _parse_file_wrapper(args):
    """Wrapper for multiprocessing — takes (index, filepath, expected_rows)."""
    idx, filepath, expected_rows = args
    try:
        data = parse_single_envelope(filepath, expected_rows=expected_rows)
        return idx, data
    except Exception as e:
        print(f"  [WARNING] Failed to parse {filepath}: {e}")
        return idx, None


def batch_parse_envelopes(
    filepaths: list[Path],
    n_workers: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse multiple fort.envelope files in parallel.
    
    All files must have the same s-grid (same beamline, same TRANSOPTR config).
    This is guaranteed because they share the same sy.f.
    
    Args:
        filepaths: list of paths to fort.envelope files
        n_workers: number of parallel workers (default: cpu_count)
    
    Returns:
        s:     (N_rows,)      — position along beamline (shared by all files)
        x_env: (N_files, N_rows) — x envelope (2-sigma)
        y_env: (N_files, N_rows) — y envelope (2-sigma)
        x_cm:  (N_files, N_rows) — x centroid
        y_cm:  (N_files, N_rows) — y centroid
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, len(filepaths), 16)
    
    N = len(filepaths)
    
    if N == 0:
        raise ValueError("No envelope files to parse")
    
    # Parse first file to get dimensions
    first = parse_single_envelope(filepaths[0])
    
    if first.ndim == 1 or first.shape[0] == 0:
        raise ValueError(
            f"First envelope file '{filepaths[0]}' is empty or has no valid data rows. "
            f"Returned shape: {first.shape}. Check that pyoptr.run completed successfully."
        )
    
    N_rows = first.shape[0]
    
    # Pre-allocate output arrays
    s = first[:, 0]  # shared s-grid
    x_env = np.empty((N, N_rows), dtype=np.float64)
    y_env = np.empty((N, N_rows), dtype=np.float64)
    x_cm = np.empty((N, N_rows), dtype=np.float64)
    y_cm = np.empty((N, N_rows), dtype=np.float64)
    
    # Fill first file
    x_env[0] = first[:, 1]
    y_env[0] = first[:, 2]
    x_cm[0] = first[:, 3]
    y_cm[0] = first[:, 4]
    
    if N == 1:
        return s, x_env, y_env, x_cm, y_cm
    
    # Parse remaining files in parallel
    args = [(i, fp, N_rows) for i, fp in enumerate(filepaths) if i > 0]
    
    if n_workers <= 1 or N < 10:
        # Sequential for small batches or debugging
        for idx, fp, expected_rows in args:
            data = parse_single_envelope(fp, expected_rows=expected_rows)
            if data is not None and data.shape[0] == N_rows:
                x_env[idx] = data[:, 1]
                y_env[idx] = data[:, 2]
                x_cm[idx] = data[:, 3]
                y_cm[idx] = data[:, 4]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_parse_file_wrapper, (idx, fp, N_rows)): (idx, fp, N_rows) for idx, fp, _ in args}
            for future in as_completed(futures):
                idx, data = future.result()
                if data is not None and data.shape[0] == N_rows:
                    x_env[idx] = data[:, 1]
                    y_env[idx] = data[:, 2]
                    x_cm[idx] = data[:, 3]
                    y_cm[idx] = data[:, 4]
    
    return s, x_env, y_env, x_cm, y_cm


def batch_parse_envelopes_numpy(
    filepaths: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Alternative: use np.loadtxt for parsing. Slower per-file but simpler.
    Good for verification / smaller batches.
    """
    arrays = []
    for fp in filepaths:
        data = np.loadtxt(
            fp, skiprows=N_HEADER_LINES, usecols=COLS_NEEDED, dtype=np.float64
        )
        arrays.append(data)
    
    stacked = np.stack(arrays)  # (N_files, N_rows, 5)
    s = stacked[0, :, 0]
    x_env = stacked[:, :, 1]
    y_env = stacked[:, :, 2]
    x_cm = stacked[:, :, 3]
    y_cm = stacked[:, :, 4]
    
    return s, x_env, y_env, x_cm, y_cm
