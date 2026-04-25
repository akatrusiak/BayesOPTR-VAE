"""
Dataset and collation for VAE training on TRANSOPTR tuning runs.

Design:
    One item = one run (all rows sharing a misalignment_id within its source file).
    The .npz stores flat rows; __getitem__ reassembles sets.
    Variable run lengths are handled by a pad+mask collate_fn.

    Two dataset classes are provided:
      - RunDataset: loads a single .npz file. Kept for backward compatibility
        with tests and simple experiments.
      - MultiFolderRunDataset: loads multiple TRANSOPTR output folders, each
        containing x_centroid_data.npz and y_centroid_data.npz. This is the
        dataset to use for production training.

    Per the design decision for this project, each (folder, axis, local_id)
    triple is treated as a *distinct* run, even when folders share the same
    underlying misalignment vector. The encoder therefore sees two
    axis-specific "marginals" of each misalignment configuration and must
    learn a latent representation that is consistent across them.

Standardization:
    Fit means/stds on the pooled training set at load time. These stats are
    part of the model's contract and must be persisted alongside
    checkpoints for inference on new data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class StandardizationStats:
    """Per-dimension mean and std. Store these with the model."""
    input_mean: np.ndarray       # (D_input,)
    input_std: np.ndarray        # (D_input,)
    y_mean: float
    y_std: float
    delta_mean: np.ndarray       # (D_delta,)
    delta_std: np.ndarray        # (D_delta,)

    def to_dict(self) -> dict:
        return {
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardizationStats":
        return cls(
            input_mean=np.asarray(d["input_mean"]),
            input_std=np.asarray(d["input_std"]),
            y_mean=float(d["y_mean"]),
            y_std=float(d["y_std"]),
            delta_mean=np.asarray(d["delta_mean"]),
            delta_std=np.asarray(d["delta_std"]),
        )


def _safe_std(x: np.ndarray, axis=0, eps: float = 1e-8) -> np.ndarray:
    """Std with a floor so constant dimensions don't divide by zero."""
    s = x.std(axis=axis)
    return np.maximum(s, eps)


class RunDataset(Dataset):
    """
    Dataset over TRANSOPTR tuning runs.

    Each __getitem__ returns (pairs, delta, run_id) for one run, where
    pairs is (N_j, D_input + 1) — all `inputs` columns concatenated with
    transmission. Decision per user: use all inputs (quads + steerers),
    not just steerers.

    Args:
        npz_path: path to an x_centroid_data.npz or y_centroid_data.npz file.
        stats: optional pre-computed standardization stats (e.g. fit on train,
               reused on val). If None, stats are fit on this file.
        standardize: if False, returns raw values (useful for debugging).
    """

    def __init__(
        self,
        npz_path: str | Path,
        stats: Optional[StandardizationStats] = None,
        standardize: bool = True,
    ):
        self.npz_path = Path(npz_path)
        self.standardize = standardize

        with np.load(self.npz_path, allow_pickle=False) as f:
            self.inputs = f["inputs"].astype(np.float64)                      # (N_total, D_input)
            self.transmission = f["transmission"].astype(np.float64)          # (N_total,)
            self.misalignment_id = f["misalignment_id"].astype(np.int64)      # (N_total,)
            self.unique_deltas = f["unique_misalignment_vectors"].astype(np.float64)  # (N_runs, D_delta)
            self.input_names = f["input_variable_names"].astype(str)          # (D_input,)
            self.delta_names = f["misalignment_names"].astype(str)            # (D_delta,)

        self.D_input = self.inputs.shape[1]
        self.D_delta = self.unique_deltas.shape[1]

        # Group row indices by run id. The ids are not guaranteed contiguous
        # across files in general, so we map them explicitly.
        self.run_ids = np.unique(self.misalignment_id)
        self.N_runs = len(self.run_ids)
        self._run_row_idx: list[np.ndarray] = [
            np.where(self.misalignment_id == rid)[0] for rid in self.run_ids
        ]

        # Fit or adopt standardization stats.
        if stats is None:
            self.stats = self._fit_stats()
        else:
            self.stats = stats

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def _fit_stats(self) -> StandardizationStats:
        """Fit standardization on this file's training rows."""
        # For delta, use the deduplicated table so each run contributes equally
        # rather than being weighted by run length.
        delta_mean = self.unique_deltas.mean(axis=0)
        delta_std = _safe_std(self.unique_deltas, axis=0)

        # For inputs, use row-level stats (BOIS explores widely within a run;
        # each row is an independent x draw).
        input_mean = self.inputs.mean(axis=0)
        input_std = _safe_std(self.inputs, axis=0)

        # Transmission is a scalar in [0,1]; standardize to zero-mean unit-std
        # so the encoder sees a comparable scale to standardized inputs.
        y_mean = float(self.transmission.mean())
        y_std = float(max(self.transmission.std(), 1e-8))

        return StandardizationStats(
            input_mean=input_mean,
            input_std=input_std,
            y_mean=y_mean,
            y_std=y_std,
            delta_mean=delta_mean,
            delta_std=delta_std,
        )

    # ------------------------------------------------------------------
    # Standardization helpers (numpy in, numpy out)
    # ------------------------------------------------------------------
    def _std_inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.stats.input_mean) / self.stats.input_std

    def _std_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.stats.y_mean) / self.stats.y_std

    def _std_delta(self, d: np.ndarray) -> np.ndarray:
        return (d - self.stats.delta_mean) / self.stats.delta_std

    def unstandardize_delta(self, d_std: np.ndarray) -> np.ndarray:
        return d_std * self.stats.delta_std + self.stats.delta_mean

    # ------------------------------------------------------------------
    # Torch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.N_runs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row_idx = self._run_row_idx[idx]
        run_id = int(self.run_ids[idx])

        x = self.inputs[row_idx]                 # (N_j, D_input)
        y = self.transmission[row_idx]           # (N_j,)
        delta = self.unique_deltas[idx]          # (D_delta,)

        if self.standardize:
            x = self._std_inputs(x)
            y = self._std_y(y)
            delta = self._std_delta(delta)

        # Concatenate into (N_j, D_input + 1)
        pairs = np.concatenate([x, y[:, None]], axis=1).astype(np.float32)
        delta_t = torch.from_numpy(delta.astype(np.float32))
        pairs_t = torch.from_numpy(pairs)

        return pairs_t, delta_t, run_id


def collate_runs(batch):
    """
    Collate a list of (pairs, delta, run_id) into a padded batch.

    Returns:
        pairs_padded: (B, N_max, D_input + 1) float32
        mask: (B, N_max) bool; True where real data, False on padding.
        deltas: (B, D_delta) float32
        run_ids: (B,) long
    """
    pairs_list, deltas, run_ids = zip(*batch)
    B = len(pairs_list)
    feat_dim = pairs_list[0].shape[1]
    N_max = max(p.shape[0] for p in pairs_list)

    pairs_padded = torch.zeros(B, N_max, feat_dim, dtype=torch.float32)
    mask = torch.zeros(B, N_max, dtype=torch.bool)

    for i, p in enumerate(pairs_list):
        n = p.shape[0]
        pairs_padded[i, :n] = p
        mask[i, :n] = True

    deltas_t = torch.stack(list(deltas), dim=0)
    run_ids_t = torch.tensor(list(run_ids), dtype=torch.long)
    return pairs_padded, mask, deltas_t, run_ids_t


# ======================================================================
# Multi-folder dataset
# ======================================================================
# A TRANSOPTR data-generation run produces one "output" folder. Each folder
# holds exactly two .npz files: x_centroid_data.npz and y_centroid_data.npz.
# Per the project's design decision (Option A), each (folder, axis,
# local_misalignment_id) is a distinct run. Runs from different folders may
# share the same underlying misalignment vector, but we do not merge them —
# the encoder treats them independently, and the identical-delta-label
# across shared-misalignment runs provides an implicit consistency
# regularizer during training.
# ----------------------------------------------------------------------

# Files we expect to find inside each folder. Order defines an integer axis tag
# used in run labels but NOT as a model input feature (see HANDOFF notes).
_AXIS_FILES = [
    ("x", "x_centroid_data.npz"),
    ("y", "y_centroid_data.npz"),
]


@dataclass
class RunLabel:
    """Metadata for one run. Useful for debugging, diagnostics, and traceability."""
    run_id: int            # global, dense index assigned by MultiFolderRunDataset
    folder: str            # folder path (as provided by the user)
    axis: str              # "x" or "y"
    local_id: int          # misalignment_id within its source file


def _discover_source_files(folders: Iterable[str | Path]) -> list[tuple[str, Path]]:
    """
    For each folder, return [(axis, path_to_npz), ...] for every axis file
    that exists. We do not require both x and y files to be present in a
    folder — future data-generation runs may produce one or the other.
    """
    found: list[tuple[str, Path, Path]] = []
    for folder in folders:
        fpath = Path(folder)
        if not fpath.is_dir():
            raise FileNotFoundError(f"Not a directory: {fpath}")
        any_found = False
        for axis, fname in _AXIS_FILES:
            npz = fpath / fname
            if npz.exists():
                found.append((axis, npz, fpath))
                any_found = True
        if not any_found:
            raise FileNotFoundError(
                f"No axis .npz files found in {fpath} "
                f"(expected any of {[fn for _, fn in _AXIS_FILES]})"
            )
    return found


class MultiFolderRunDataset(Dataset):
    """
    Dataset over TRANSOPTR tuning runs sourced from one or more output folders.

    Each __getitem__ returns (pairs, delta, run_id) for one run, where
    pairs is (N_j, D_input + 1) — all `inputs` columns concatenated with
    transmission. Standardization is fit on the pooled data across all
    supplied folders and axes.

    Run identity:
        Each (folder, axis, local_misalignment_id) triple is a distinct run
        with a globally unique run_id in [0, N_runs). The mapping is
        recorded in `self.run_labels` for traceability.

    Schema validation:
        All source files must share input_variable_names, misalignment_names,
        D_input, and D_delta. A mismatch raises ValueError at construction.

    Args:
        folders: iterable of folder paths, each containing one or both of
            x_centroid_data.npz, y_centroid_data.npz.
        stats: optional pre-computed StandardizationStats to reuse (e.g. from
            train → val). If None, stats are fit on the pooled data.
        standardize: if False, returns raw values (useful for debugging).
    """

    def __init__(
        self,
        folders: Iterable[str | Path],
        stats: Optional[StandardizationStats] = None,
        standardize: bool = True,
        max_pairs_per_run: Optional[int] = None,
        subsample_strategy: str = "uniform",
        subsample_seed: int = 0,
    ):
        """
        Args:
            folders: iterable of folder paths, each containing one or both of
                x_centroid_data.npz, y_centroid_data.npz.
            stats: optional pre-computed StandardizationStats to reuse.
            standardize: if False, returns raw values (useful for debugging).
            max_pairs_per_run: optional cap on pairs per run. Runs longer than
                this are subsampled down; shorter runs are returned as-is.
                None = no thinning (default).
            subsample_strategy:
                - "uniform": evenly-spaced indices across the full trajectory
                  (stratified by SA time; includes first and last).
                - "head": keep the first N pairs (SA-time truncation).
                - "random": uniformly random indices with subsample_seed.
            subsample_seed: seed for "random" strategy; unused otherwise.
        """
        folders = list(folders)
        if len(folders) == 0:
            raise ValueError("MultiFolderRunDataset requires at least one folder")
        if subsample_strategy not in {"uniform", "head", "random"}:
            raise ValueError(
                f"subsample_strategy must be 'uniform', 'head', or 'random'; "
                f"got {subsample_strategy!r}"
            )

        self.folders = [str(Path(f)) for f in folders]
        self.standardize = standardize
        self.max_pairs_per_run = max_pairs_per_run
        self.subsample_strategy = subsample_strategy
        self.subsample_seed = subsample_seed

        sources = _discover_source_files(folders)

        # ----- load every source file, validate schema, and build the run index
        self._input_names: Optional[np.ndarray] = None
        self._delta_names: Optional[np.ndarray] = None

        # Flat storage. To keep memory and code simple, we concatenate all rows
        # across all source files and store one vector/index per run.
        all_inputs: list[np.ndarray] = []
        all_transmission: list[np.ndarray] = []
        # For each run (globally), we store:
        #   - the row indices into the concatenated inputs/transmission arrays
        #   - the delta vector (copied from its source file's unique table)
        #   - a RunLabel with provenance
        run_row_idx: list[np.ndarray] = []
        run_deltas: list[np.ndarray] = []
        run_labels: list[RunLabel] = []

        row_offset = 0
        for axis, npz_path, folder_path in sources:
            with np.load(npz_path, allow_pickle=False) as f:
                inputs = f["inputs"].astype(np.float64)
                transmission = f["transmission"].astype(np.float64)
                misalignment_id = f["misalignment_id"].astype(np.int64)
                unique_deltas = f["unique_misalignment_vectors"].astype(np.float64)
                input_names = f["input_variable_names"].astype(str)
                delta_names = f["misalignment_names"].astype(str)

            # Schema checks — any mismatch is a correctness bug, not a recoverable state.
            if self._input_names is None:
                self._input_names = input_names
                self._delta_names = delta_names
            else:
                if not np.array_equal(input_names, self._input_names):
                    raise ValueError(
                        f"input_variable_names mismatch in {npz_path}: "
                        f"{input_names.tolist()} vs {self._input_names.tolist()}"
                    )
                if not np.array_equal(delta_names, self._delta_names):
                    raise ValueError(
                        f"misalignment_names mismatch in {npz_path}: "
                        f"{delta_names.tolist()} vs {self._delta_names.tolist()}"
                    )

            # Group rows by local misalignment id. Sorted local_ids give a
            # stable, deterministic ordering of runs regardless of row order.
            local_ids = np.unique(misalignment_id)
            for local_id in local_ids:
                rows = np.where(misalignment_id == local_id)[0]
                # SA-time ordering: within a run, TRANSOPTR writes rows
                # sequentially as SA proceeds, so `rows` is already in
                # SA-time order. This matters for "head" subsampling.
                rows = np.sort(rows)
                rows_thinned = self._subsample_rows(rows, run_id=len(run_labels))
                # Offset to global concatenated-array coordinates
                run_row_idx.append(rows_thinned + row_offset)
                run_deltas.append(unique_deltas[local_id])
                run_labels.append(RunLabel(
                    run_id=len(run_labels),
                    folder=str(folder_path),
                    axis=axis,
                    local_id=int(local_id),
                ))

            all_inputs.append(inputs)
            all_transmission.append(transmission)
            row_offset += inputs.shape[0]

        self.inputs = np.concatenate(all_inputs, axis=0)
        self.transmission = np.concatenate(all_transmission, axis=0)
        self._run_row_idx = run_row_idx
        self._run_deltas = np.stack(run_deltas, axis=0)      # (N_runs, D_delta)
        self.run_labels = run_labels

        self.D_input = self.inputs.shape[1]
        self.D_delta = self._run_deltas.shape[1]
        self.N_runs = len(self.run_labels)

        # Exposed for any code that still expects these names (parity with RunDataset).
        self.input_names = self._input_names
        self.delta_names = self._delta_names
        self.run_ids = np.arange(self.N_runs, dtype=np.int64)
        # unique_deltas here is one row per run (not deduplicated across shared δ);
        # this matches how RunDataset indexes deltas by run index.
        self.unique_deltas = self._run_deltas

        # ----- standardization
        if stats is None:
            self.stats = self._fit_stats()
        else:
            self.stats = stats

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def _fit_stats(self) -> StandardizationStats:
        """Fit standardization on the pooled training rows."""
        # Deltas: weight each run equally rather than by trajectory length.
        delta_mean = self._run_deltas.mean(axis=0)
        delta_std = _safe_std(self._run_deltas, axis=0)

        # Inputs: row-level stats across the pooled rows.
        input_mean = self.inputs.mean(axis=0)
        input_std = _safe_std(self.inputs, axis=0)

        y_mean = float(self.transmission.mean())
        y_std = float(max(self.transmission.std(), 1e-8))

        return StandardizationStats(
            input_mean=input_mean,
            input_std=input_std,
            y_mean=y_mean,
            y_std=y_std,
            delta_mean=delta_mean,
            delta_std=delta_std,
        )

    # ------------------------------------------------------------------
    # Per-run subsampling
    # ------------------------------------------------------------------
    def _subsample_rows(self, rows: np.ndarray, run_id: int) -> np.ndarray:
        """
        Thin a single run's row indices down to <= max_pairs_per_run.

        `rows` must already be in SA-time order (smallest-to-largest row
        number in the source file). We pass `run_id` so that the 'random'
        strategy can seed deterministically per-run (independent of the
        traversal order of folders/axes).
        """
        N = len(rows)
        cap = self.max_pairs_per_run
        if cap is None or N <= cap:
            return rows

        if self.subsample_strategy == "uniform":
            # Evenly-spaced indices spanning the full trajectory.
            # np.linspace endpoint=True + round → guaranteed unique for cap <= N.
            sel = np.linspace(0, N - 1, num=cap).round().astype(np.int64)
            sel = np.unique(sel)  # linspace+round can rarely dup at tiny N
            return rows[sel]

        if self.subsample_strategy == "head":
            return rows[:cap]

        if self.subsample_strategy == "random":
            rng = np.random.RandomState(self.subsample_seed + run_id)
            sel = rng.choice(N, size=cap, replace=False)
            sel.sort()                  # preserve SA-time order for debuggability
            return rows[sel]

        # Unreachable; __init__ validates.
        raise RuntimeError(f"Bad subsample_strategy: {self.subsample_strategy}")

    # ------------------------------------------------------------------
    # Standardization helpers (parity with RunDataset)
    # ------------------------------------------------------------------
    def _std_inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.stats.input_mean) / self.stats.input_std

    def _std_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.stats.y_mean) / self.stats.y_std

    def _std_delta(self, d: np.ndarray) -> np.ndarray:
        return (d - self.stats.delta_mean) / self.stats.delta_std

    def unstandardize_delta(self, d_std: np.ndarray) -> np.ndarray:
        return d_std * self.stats.delta_std + self.stats.delta_mean

    # ------------------------------------------------------------------
    # Torch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.N_runs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rows = self._run_row_idx[idx]

        x = self.inputs[rows]                    # (N_j, D_input)
        y = self.transmission[rows]              # (N_j,)
        delta = self._run_deltas[idx]            # (D_delta,)

        if self.standardize:
            x = self._std_inputs(x)
            y = self._std_y(y)
            delta = self._std_delta(delta)

        pairs = np.concatenate([x, y[:, None]], axis=1).astype(np.float32)
        pairs_t = torch.from_numpy(pairs)
        delta_t = torch.from_numpy(delta.astype(np.float32))
        run_id = self.run_labels[idx].run_id
        return pairs_t, delta_t, run_id

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Compact, single-string description of the dataset contents."""
        from collections import Counter
        by_folder = Counter(lbl.folder for lbl in self.run_labels)
        by_axis = Counter(lbl.axis for lbl in self.run_labels)
        lengths = [len(r) for r in self._run_row_idx]
        lines = [
            f"MultiFolderRunDataset: N_runs={self.N_runs}, D_input={self.D_input}, "
            f"D_delta={self.D_delta}",
            f"  Total pairs: {self.inputs.shape[0]}",
            f"  Run lengths: min={min(lengths)}, max={max(lengths)}, "
            f"mean={np.mean(lengths):.1f}",
            f"  Axis split: " + ", ".join(f"{k}={v}" for k, v in sorted(by_axis.items())),
            f"  Folders:",
        ]
        for folder in self.folders:
            lines.append(f"    {folder}: {by_folder.get(folder, 0)} runs")
        return "\n".join(lines)
