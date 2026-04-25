"""
Tests for MultiFolderRunDataset.

Verifies that the multi-folder loader:
  - Pools runs from all axis files across all supplied folders
  - Assigns globally unique, dense run_ids
  - Produces pairs/deltas consistent with the raw .npz files
  - Validates schema across files
  - Accepts/reuses externally-supplied standardization stats
  - Works with collate_runs end-to-end

Set VAE_TEST_DATA_ROOT to override the default data location.
Defaults to `<repo>/../data` (i.e. where output_good_4 lives).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from vae.data import (
    MultiFolderRunDataset,
    RunLabel,
    StandardizationStats,
    collate_runs,
)


DATA_ROOT = Path(os.environ.get(
    "VAE_TEST_DATA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
))
FOLDER_GOOD_4 = DATA_ROOT / "output_good_4"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def ds_single():
    """Dataset over just output_good_4."""
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")
    return MultiFolderRunDataset([FOLDER_GOOD_4], standardize=True)


@pytest.fixture(scope="module")
def ds_single_raw():
    """Unstandardized version for round-trip checks."""
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")
    return MultiFolderRunDataset([FOLDER_GOOD_4], standardize=False)


# ----------------------------------------------------------------------
# T_M1 — Run count matches sum of per-file local runs
# ----------------------------------------------------------------------
def test_m1_run_count_matches_sources(ds_single):
    """
    With one folder containing both x and y files, the run count should be
    exactly (n_x_runs + n_y_runs). For output_good_4 that's 120 + 120 = 240.
    """
    nx = len(np.unique(np.load(FOLDER_GOOD_4 / "x_centroid_data.npz")["misalignment_id"]))
    ny = len(np.unique(np.load(FOLDER_GOOD_4 / "y_centroid_data.npz")["misalignment_id"]))
    assert len(ds_single) == nx + ny, (
        f"Expected {nx + ny} runs (= {nx} x-runs + {ny} y-runs), got {len(ds_single)}"
    )


# ----------------------------------------------------------------------
# T_M2 — Run IDs are globally unique and dense [0, N)
# ----------------------------------------------------------------------
def test_m2_run_ids_globally_unique_and_dense(ds_single):
    """
    The entire point of the multi-folder design: local ids collide across
    files, but global run_ids must be unique and form a dense range.
    """
    ids = [ds_single.run_labels[i].run_id for i in range(len(ds_single))]
    assert sorted(ids) == list(range(len(ds_single))), (
        "run_ids are not a dense [0, N) range — did the labeller get out of sync "
        "with the per-run index?"
    )


# ----------------------------------------------------------------------
# T_M3 — Each run's pairs match the raw rows for (axis, local_id)
# ----------------------------------------------------------------------
def test_m3_pairs_match_raw_rows(ds_single_raw):
    """
    Pull a few runs, reconstruct what the raw rows *should* be from the
    source .npz, and check they agree bit-for-bit with what __getitem__
    returned. This catches off-by-one bugs in the row-offset bookkeeping.
    """
    caches: dict[Path, dict] = {}

    def _src(axis: str) -> dict:
        p = FOLDER_GOOD_4 / f"{axis}_centroid_data.npz"
        if p not in caches:
            with np.load(p, allow_pickle=False) as f:
                caches[p] = {
                    "inputs": f["inputs"],
                    "transmission": f["transmission"],
                    "mid": f["misalignment_id"],
                    "udeltas": f["unique_misalignment_vectors"],
                }
        return caches[p]

    rng = np.random.RandomState(0)
    sample = rng.choice(len(ds_single_raw), size=6, replace=False)
    for i in sample:
        pairs, delta, run_id = ds_single_raw[int(i)]
        lbl = ds_single_raw.run_labels[int(i)]
        assert run_id == lbl.run_id

        src = _src(lbl.axis)
        rows = np.where(src["mid"] == lbl.local_id)[0]
        assert len(rows) > 0, f"No raw rows for local_id={lbl.local_id}"

        # Pairs = inputs concatenated with transmission column
        expected_pairs = np.concatenate(
            [src["inputs"][rows], src["transmission"][rows, None]], axis=1
        ).astype(np.float32)
        expected_delta = src["udeltas"][lbl.local_id].astype(np.float32)

        assert pairs.shape == expected_pairs.shape, (
            f"run_id={run_id}: pair shape {pairs.shape} != expected {expected_pairs.shape}"
        )
        assert np.allclose(pairs.numpy(), expected_pairs, atol=1e-6), (
            f"run_id={run_id}: pairs differ from raw source"
        )
        assert np.allclose(delta.numpy(), expected_delta, atol=1e-6), (
            f"run_id={run_id}: delta differs from raw source"
        )


# ----------------------------------------------------------------------
# T_M4 — Standardization round-trip on delta
# ----------------------------------------------------------------------
def test_m4_delta_standardization_roundtrip(ds_single, ds_single_raw):
    """
    unstandardize_delta(standardized_delta) must recover the raw delta
    for the same index.
    """
    for i in [0, 17, len(ds_single) - 1]:
        _, d_std, _ = ds_single[i]
        _, d_raw, _ = ds_single_raw[i]
        recovered = ds_single.unstandardize_delta(d_std.numpy())
        assert np.allclose(recovered, d_raw.numpy(), atol=1e-5), (
            f"Round-trip failed at i={i}"
        )


# ----------------------------------------------------------------------
# T_M5 — Standardized deltas are zero-mean, unit-std across runs
# ----------------------------------------------------------------------
def test_m5_standardized_deltas_are_normalized(ds_single):
    all_std_deltas = np.stack([ds_single[i][1].numpy() for i in range(len(ds_single))])
    mean_per_dim = all_std_deltas.mean(axis=0)
    std_per_dim = all_std_deltas.std(axis=0)
    assert np.abs(mean_per_dim).max() < 1e-5, (
        f"Mean of standardized deltas not near zero: max |mean|={np.abs(mean_per_dim).max():.2e}"
    )
    assert np.abs(std_per_dim - 1).max() < 1e-3, (
        f"Std of standardized deltas not near 1: max |std-1|={np.abs(std_per_dim - 1).max():.2e}"
    )


# ----------------------------------------------------------------------
# T_M6 — External stats are respected (train stats reused on val)
# ----------------------------------------------------------------------
def test_m6_external_stats_respected(ds_single):
    """
    Build a second dataset with the first's stats supplied externally —
    the result must use those exact stats, not refit.
    """
    ds2 = MultiFolderRunDataset(
        [FOLDER_GOOD_4], stats=ds_single.stats, standardize=True
    )
    # Objects may differ by identity but all stat fields should match
    assert np.array_equal(ds2.stats.delta_mean, ds_single.stats.delta_mean)
    assert np.array_equal(ds2.stats.delta_std, ds_single.stats.delta_std)
    assert np.array_equal(ds2.stats.input_mean, ds_single.stats.input_mean)
    assert np.array_equal(ds2.stats.input_std, ds_single.stats.input_std)
    assert ds2.stats.y_mean == ds_single.stats.y_mean
    assert ds2.stats.y_std == ds_single.stats.y_std


# ----------------------------------------------------------------------
# T_M7 — Pooling across multiple folders
# ----------------------------------------------------------------------
def test_m7_multi_folder_pool_count_is_additive():
    """
    Load output_good_4 alone, then load it 'twice' (by symlinking to a copy)
    and verify the run count exactly doubles. We symlink to a temp copy
    rather than pass the same path twice because some future constraint
    might reject duplicates — this keeps the contract strict.
    """
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")

    ds_one = MultiFolderRunDataset([FOLDER_GOOD_4], standardize=False)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_folder = Path(tmp) / "output_good_4_copy"
        tmp_folder.mkdir()
        # Symlink the two .npz files in
        for fn in ["x_centroid_data.npz", "y_centroid_data.npz"]:
            os.symlink(FOLDER_GOOD_4 / fn, tmp_folder / fn)

        ds_two = MultiFolderRunDataset(
            [FOLDER_GOOD_4, tmp_folder], standardize=False
        )
        assert len(ds_two) == 2 * len(ds_one), (
            f"Expected {2 * len(ds_one)} runs, got {len(ds_two)}"
        )

        # Run IDs are still dense and unique
        ids = sorted(lbl.run_id for lbl in ds_two.run_labels)
        assert ids == list(range(len(ds_two)))

        # Each folder contributes exactly len(ds_one) runs
        from collections import Counter
        by_folder = Counter(lbl.folder for lbl in ds_two.run_labels)
        assert set(by_folder.values()) == {len(ds_one)}, (
            f"Per-folder counts not balanced: {dict(by_folder)}"
        )


# ----------------------------------------------------------------------
# T_M8 — Schema mismatch is rejected
# ----------------------------------------------------------------------
def test_m8_schema_mismatch_rejected():
    """
    If two folders disagree on input_variable_names or misalignment_names,
    construction must raise rather than silently pool incompatible data.
    """
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")

    with tempfile.TemporaryDirectory() as tmp:
        bad_folder = Path(tmp) / "output_schema_wrong"
        bad_folder.mkdir()

        # Copy the x file but rewrite input_variable_names to disagree.
        src = FOLDER_GOOD_4 / "x_centroid_data.npz"
        with np.load(src, allow_pickle=False) as f:
            payload = {k: f[k] for k in f.files}
        # Rename one input column
        payload["input_variable_names"] = np.array(
            ["BOGUS_NAME"] + list(payload["input_variable_names"][1:])
        )
        np.savez(bad_folder / "x_centroid_data.npz", **payload)

        with pytest.raises(ValueError, match="input_variable_names mismatch"):
            MultiFolderRunDataset([FOLDER_GOOD_4, bad_folder])


# ----------------------------------------------------------------------
# T_M9 — Empty / invalid folders are rejected
# ----------------------------------------------------------------------
def test_m9_empty_folder_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            MultiFolderRunDataset([empty])


def test_m9b_missing_folder_rejected():
    with pytest.raises(FileNotFoundError):
        MultiFolderRunDataset(["/nonexistent/path/does/not/exist"])


def test_m9c_empty_folder_list_rejected():
    with pytest.raises(ValueError):
        MultiFolderRunDataset([])


# ----------------------------------------------------------------------
# T_M10 — collate_runs interop
# ----------------------------------------------------------------------
def test_m10_collate_interop(ds_single):
    items = [ds_single[i] for i in [0, 1, 2, 3]]
    lengths = [p.shape[0] for p, _, _ in items]
    N_max = max(lengths)
    pairs_padded, mask, deltas, run_ids = collate_runs(items)

    assert pairs_padded.shape == (4, N_max, ds_single.D_input + 1)
    assert mask.shape == (4, N_max)
    assert deltas.shape == (4, ds_single.D_delta)
    for i, n in enumerate(lengths):
        assert mask[i].sum().item() == n


def test_m11_single_axis_folder_works():
    """
    If a future data-generation run produces only one axis file, the loader
    should still succeed — it only rejects folders with *zero* axis files.
    """
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "x_only"
        folder.mkdir()
        os.symlink(
            FOLDER_GOOD_4 / "x_centroid_data.npz",
            folder / "x_centroid_data.npz",
        )
        ds = MultiFolderRunDataset([folder], standardize=False)
        # All runs should be x-axis
        axes = {lbl.axis for lbl in ds.run_labels}
        assert axes == {"x"}, f"Expected only x-axis runs, got {axes}"


# ----------------------------------------------------------------------
# T_M12 — Thinning caps run length
# ----------------------------------------------------------------------
def test_m12_max_pairs_caps_all_runs():
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")
    ds = MultiFolderRunDataset(
        [FOLDER_GOOD_4], standardize=False,
        max_pairs_per_run=50, subsample_strategy="uniform",
    )
    for i in range(len(ds)):
        pairs, _, _ = ds[i]
        assert pairs.shape[0] <= 50, (
            f"Run {i} has {pairs.shape[0]} pairs; cap was 50"
        )


# ----------------------------------------------------------------------
# T_M13 — Short runs (<cap) are untouched
# ----------------------------------------------------------------------
def test_m13_short_runs_untouched():
    """Nothing should be padded/duplicated; short runs pass through."""
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")
    ds_full = MultiFolderRunDataset([FOLDER_GOOD_4], standardize=False)
    # Pick a huge cap larger than any run's length
    cap = int(max(len(r) for r in ds_full._run_row_idx)) + 100
    ds_capped = MultiFolderRunDataset(
        [FOLDER_GOOD_4], standardize=False,
        max_pairs_per_run=cap, subsample_strategy="uniform",
    )
    for i in range(len(ds_full)):
        p_full, _, _ = ds_full[i]
        p_cap, _, _ = ds_capped[i]
        assert p_full.shape == p_cap.shape


# ----------------------------------------------------------------------
# T_M14 — Uniform strategy: indices are monotonic and include endpoints
# ----------------------------------------------------------------------
def test_m14_uniform_strategy_is_stratified():
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")
    ds = MultiFolderRunDataset(
        [FOLDER_GOOD_4], standardize=False,
        max_pairs_per_run=10, subsample_strategy="uniform",
    )
    # Pick a run known to be long (full set)
    full = MultiFolderRunDataset([FOLDER_GOOD_4], standardize=False)
    for i in range(3):
        full_rows = full._run_row_idx[i]
        thin_rows = ds._run_row_idx[i]
        # thin indices must be a subset, in order, and span endpoints
        assert len(thin_rows) == 10
        assert thin_rows[0] == full_rows[0], "uniform should keep the first pair"
        assert thin_rows[-1] == full_rows[-1], "uniform should keep the last pair"
        assert np.all(np.diff(thin_rows) > 0), "thin indices must be strictly increasing"
        # All thin indices appear in the full trajectory
        assert np.all(np.isin(thin_rows, full_rows))


# ----------------------------------------------------------------------
# T_M15 — Head strategy: takes the first N rows
# ----------------------------------------------------------------------
def test_m15_head_strategy():
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")
    full = MultiFolderRunDataset([FOLDER_GOOD_4], standardize=False)
    ds = MultiFolderRunDataset(
        [FOLDER_GOOD_4], standardize=False,
        max_pairs_per_run=15, subsample_strategy="head",
    )
    for i in range(3):
        assert np.array_equal(ds._run_row_idx[i], full._run_row_idx[i][:15])


# ----------------------------------------------------------------------
# T_M16 — Random strategy is deterministic given seed
# ----------------------------------------------------------------------
def test_m16_random_strategy_deterministic():
    if not FOLDER_GOOD_4.is_dir():
        pytest.skip(f"Data folder not available: {FOLDER_GOOD_4}")
    ds1 = MultiFolderRunDataset(
        [FOLDER_GOOD_4], standardize=False,
        max_pairs_per_run=20, subsample_strategy="random", subsample_seed=42,
    )
    ds2 = MultiFolderRunDataset(
        [FOLDER_GOOD_4], standardize=False,
        max_pairs_per_run=20, subsample_strategy="random", subsample_seed=42,
    )
    for i in range(5):
        assert np.array_equal(ds1._run_row_idx[i], ds2._run_row_idx[i])
    # Different seed → different indices
    ds3 = MultiFolderRunDataset(
        [FOLDER_GOOD_4], standardize=False,
        max_pairs_per_run=20, subsample_strategy="random", subsample_seed=7,
    )
    diffs = sum(
        0 if np.array_equal(ds1._run_row_idx[i], ds3._run_row_idx[i]) else 1
        for i in range(10)
    )
    assert diffs > 0, "random with different seeds should not match"
