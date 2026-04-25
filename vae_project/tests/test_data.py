"""
Phase 1 tests: data integrity and collation.

Run with:  cd /home/claude/vae_project && python -m pytest tests/test_data.py -v
"""

import numpy as np
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vae.data import RunDataset, collate_runs


NPZ_PATH = os.environ.get(
    "VAE_TEST_NPZ",
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "data",
        "output_good_4", "x_centroid_data.npz",
    )),
)


@pytest.fixture(scope="module")
def ds():
    return RunDataset(NPZ_PATH, standardize=True)


@pytest.fixture(scope="module")
def ds_raw():
    return RunDataset(NPZ_PATH, standardize=False)


# ----------------------------------------------------------------------
# T1.1 — Run grouping sanity check
# ----------------------------------------------------------------------
def test_t1_1_run_grouping(ds_raw):
    """Every __getitem__ must return rows that share one misalignment vector."""
    raw = np.load(NPZ_PATH, allow_pickle=False)
    mv = raw["misalignment_vectors"]
    ids = raw["misalignment_id"]

    rng = np.random.RandomState(0)
    sample_idxs = rng.choice(len(ds_raw), size=5, replace=False)
    for i in sample_idxs:
        pairs, delta, run_id = ds_raw[i]
        # All flat rows with this run_id should share the same misalignment
        mask = ids == run_id
        rows = mv[mask]
        assert np.allclose(rows, rows[0]), f"Run {run_id}: rows disagree"
        # And should match the unique table row we returned
        assert np.allclose(rows[0], delta.numpy()), (
            f"Run {run_id}: delta from dataset disagrees with flat rows"
        )


# ----------------------------------------------------------------------
# T1.2 — Transmission in [0, 1]
# ----------------------------------------------------------------------
def test_t1_2_transmission_range():
    raw = np.load(NPZ_PATH, allow_pickle=False)
    y = raw["transmission"]
    assert y.min() >= -1e-12, f"Negative transmission found: {y.min()}"
    assert y.max() <= 1.0 + 1e-12, f"Transmission > 1 found: {y.max()}"


# ----------------------------------------------------------------------
# T1.3 — Run-length distribution
# ----------------------------------------------------------------------
def test_t1_3_run_length_distribution(ds_raw):
    lengths = [ds_raw[i][0].shape[0] for i in range(len(ds_raw))]
    assert min(lengths) > 0, "Empty run detected"
    # Warn on degenerate short/long runs rather than fail
    if min(lengths) < 10:
        pytest.fail(f"Very short run ({min(lengths)} envelopes) — investigate")
    print(
        f"\n  Run lengths: min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.1f}, n_runs={len(lengths)}"
    )


# ----------------------------------------------------------------------
# T1.4 — Standardization round-trip
# ----------------------------------------------------------------------
def test_t1_4_standardization_roundtrip(ds):
    # Round-trip delta (the one we explicitly invert for Stage 2)
    _, delta_std, run_id = ds[0]
    delta_recovered = ds.unstandardize_delta(delta_std.numpy())
    delta_true = ds.unique_deltas[0]
    assert np.allclose(delta_recovered, delta_true, atol=1e-5), (
        "Delta standardization round-trip failed"
    )

    # Verify stats are actually being applied: standardized delta should have
    # near-zero mean and unit std across the run dimension.
    all_std_deltas = np.stack([ds[i][1].numpy() for i in range(len(ds))])
    # Per-dim mean should be close to 0, per-dim std close to 1
    mean_per_dim = all_std_deltas.mean(axis=0)
    std_per_dim = all_std_deltas.std(axis=0)
    assert np.abs(mean_per_dim).max() < 1e-5, f"Std delta mean nonzero: {mean_per_dim}"
    assert np.abs(std_per_dim - 1).max() < 1e-3, f"Std delta std != 1: {std_per_dim}"


# ----------------------------------------------------------------------
# T1.5 — Collate function
# ----------------------------------------------------------------------
def test_t1_5_collate(ds):
    # Grab three runs with different lengths
    items = [ds[0], ds[1], ds[2]]
    N_expected = [p.shape[0] for p, _, _ in items]
    N_max = max(N_expected)

    pairs_padded, mask, deltas, run_ids = collate_runs(items)

    assert pairs_padded.shape == (3, N_max, ds.D_input + 1)
    assert mask.shape == (3, N_max)
    assert deltas.shape == (3, ds.D_delta)
    assert run_ids.shape == (3,)

    # Mask sums equal original run lengths
    for i, n in enumerate(N_expected):
        assert mask[i].sum().item() == n, f"Run {i}: mask sum {mask[i].sum()} != {n}"

    # Padded positions are zero
    for i, n in enumerate(N_expected):
        if n < N_max:
            assert torch.all(pairs_padded[i, n:] == 0), (
                f"Run {i} has nonzero padding"
            )


# ----------------------------------------------------------------------
# Extra: deterministic indexing (property required by DataLoader)
# ----------------------------------------------------------------------
def test_deterministic_indexing(ds):
    p1, d1, r1 = ds[0]
    p2, d2, r2 = ds[0]
    assert torch.equal(p1, p2)
    assert torch.equal(d1, d2)
    assert r1 == r2
