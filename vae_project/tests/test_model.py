"""
Phase 4 tests: full VAE ELBO and training dynamics on real data.

T4.1 — loss is finite
T4.2 — single-batch overfitting (can the VAE memorize a small batch?)
T4.3 — KL is non-trivial (no posterior collapse on overfit)
T4.4 — KL is not exploding
T4.5 — functional validation smoke test (train recon is meaningful, not just mean prediction)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from vae.data import RunDataset, collate_runs
from vae.decoder import MisalignmentDecoder
from vae.encoder import DeepSetsEncoder
from vae.model import VAE


NPZ_PATH = os.environ.get(
    "VAE_TEST_NPZ",
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "data",
        "output_good_4", "x_centroid_data.npz",
    )),
)


def build_vae(d_latent=5, d_hidden=64, d_input=9, d_delta=27, beta=1.0, seed=0):
    torch.manual_seed(seed)
    enc = DeepSetsEncoder(d_input=d_input + 1, d_hidden=d_hidden, d_latent=d_latent)
    dec = MisalignmentDecoder(d_latent=d_latent, d_hidden=d_hidden, d_delta=d_delta)
    return VAE(enc, dec, beta=beta)


@pytest.fixture(scope="module")
def ds():
    return RunDataset(NPZ_PATH, standardize=True)


@pytest.fixture(scope="module")
def small_batch(ds):
    """Collate the first 4 runs into a batch for repeated overfitting tests."""
    items = [ds[i] for i in range(4)]
    return collate_runs(items)


# ----------------------------------------------------------------------
# T4.1 — Loss is finite
# ----------------------------------------------------------------------
def test_t4_1_loss_finite(ds, small_batch):
    pairs, mask, deltas, _ = small_batch
    vae = build_vae(d_input=ds.D_input, d_delta=ds.D_delta)
    loss, metrics = vae.elbo_loss(pairs, mask, deltas)
    assert math.isfinite(loss.item()), f"Loss is {loss.item()}"
    assert math.isfinite(metrics.recon)
    assert math.isfinite(metrics.kl)
    assert metrics.recon >= 0, f"Negative MSE: {metrics.recon}"
    assert metrics.kl >= 0, f"Negative KL: {metrics.kl}"


# ----------------------------------------------------------------------
# T4.2 — Single-batch overfitting
# ----------------------------------------------------------------------
def test_t4_2_single_batch_overfit(ds, small_batch):
    """
    Train the full VAE on one small batch for ~2000 steps.
    Reconstruction loss should drop substantially. This is the 'can it
    memorize at all?' test.
    """
    pairs, mask, deltas, _ = small_batch
    vae = build_vae(d_input=ds.D_input, d_delta=ds.D_delta, beta=1.0)
    opt = torch.optim.Adam(vae.parameters(), lr=3e-3)

    vae.train()
    recon_history = []
    kl_history = []
    for step in range(2000):
        opt.zero_grad()
        loss, metrics = vae.elbo_loss(pairs, mask, deltas)
        loss.backward()
        # Clip to avoid rare numerical spikes; 10.0 is very loose.
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
        opt.step()
        recon_history.append(metrics.recon)
        kl_history.append(metrics.kl)

    initial_recon = np.mean(recon_history[:20])
    final_recon = np.mean(recon_history[-20:])

    # Standardized deltas have unit variance per dim, so predicting the mean
    # (zeros) gives recon ≈ D_delta. Beating this by a lot means the model
    # is actually using information from the input set.
    mean_predictor_recon = ds.D_delta  # rough baseline

    assert final_recon < initial_recon * 0.5, (
        f"Recon loss barely moved: {initial_recon:.3f} -> {final_recon:.3f}. "
        f"Model not learning."
    )
    assert final_recon < 0.5 * mean_predictor_recon, (
        f"Final recon {final_recon:.3f} is not meaningfully below mean-predictor "
        f"baseline {mean_predictor_recon:.3f}. Posterior may be collapsed "
        f"(KL tail mean = {np.mean(kl_history[-20:]):.3f})."
    )


# ----------------------------------------------------------------------
# T4.3 — KL is non-trivial (no posterior collapse)
# ----------------------------------------------------------------------
def test_t4_3_kl_nontrivial(ds, small_batch):
    """After overfitting, KL must be > 0 meaningfully. If KL ~ 0, the
    encoder is ignoring its input — posterior collapse."""
    pairs, mask, deltas, _ = small_batch
    vae = build_vae(d_input=ds.D_input, d_delta=ds.D_delta, beta=1.0)
    opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
    vae.train()
    for _ in range(2000):
        opt.zero_grad()
        loss, metrics = vae.elbo_loss(pairs, mask, deltas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
        opt.step()

    # A "used" latent should have per-run KL at least on the order of a few
    # nats — one KL-useful dim is ~0.5+. If total KL < 0.1, collapse.
    assert metrics.kl > 0.1, (
        f"Total KL tiny ({metrics.kl:.3e}) after overfitting — posterior collapse. "
        f"Per-dim KL: {[f'{k:.3e}' for k in metrics.per_dim_kl]}"
    )
    # Count "alive" dims (KL > 0.01)
    alive = sum(1 for k in metrics.per_dim_kl if k > 0.01)
    assert alive >= 1, (
        f"No alive latent dims. Per-dim KL: {metrics.per_dim_kl}"
    )


# ----------------------------------------------------------------------
# T4.4 — KL is not exploding
# ----------------------------------------------------------------------
def test_t4_4_kl_not_exploding(ds, small_batch):
    """KL should be bounded. Rough ceiling: a few nats per dim is typical for
    well-behaved VAEs; 100+ per dim usually means log_var is running away."""
    pairs, mask, deltas, _ = small_batch
    vae = build_vae(d_input=ds.D_input, d_delta=ds.D_delta, beta=1.0)
    opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
    vae.train()
    max_kl = 0.0
    for _ in range(2000):
        opt.zero_grad()
        loss, metrics = vae.elbo_loss(pairs, mask, deltas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
        opt.step()
        max_kl = max(max_kl, metrics.kl)
    # d_z = 5, so ~20-50 range for heavy but healthy use; >100 = runaway.
    assert max_kl < 100.0, f"KL exploded to {max_kl:.2f}"


# ----------------------------------------------------------------------
# T4.5 — Functional validation: encoder actually conditions on the set
# ----------------------------------------------------------------------
def test_t4_5_encoder_discriminates_runs(ds, small_batch):
    """
    On the overfit batch, each run has a different ground-truth delta.
    After training, decoding mu_i (deterministic) should give delta_hat_i
    that is closer to delta_true_i than to delta_true_j (j != i).

    This catches the failure mode where the decoder learns a batch-mean
    output regardless of z — invisible in MSE, visible here.
    """
    pairs, mask, deltas, _ = small_batch
    vae = build_vae(d_input=ds.D_input, d_delta=ds.D_delta, beta=1.0)
    opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
    vae.train()
    for _ in range(2000):
        opt.zero_grad()
        loss, _ = vae.elbo_loss(pairs, mask, deltas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
        opt.step()

    vae.eval()
    with torch.no_grad():
        out = vae.forward(pairs, mask, sample=False)  # use mu, not sample
        delta_hat = out["delta_hat"]  # (B, D_delta)

    # Pairwise distance matrix: predicted vs. every ground truth
    # dist[i, j] = || delta_hat[i] - delta_true[j] ||
    diff = delta_hat.unsqueeze(1) - deltas.unsqueeze(0)   # (B, B, D_delta)
    dist = diff.norm(dim=-1)                              # (B, B)

    # Each row's minimum should be on the diagonal
    nearest = dist.argmin(dim=1)
    correct = (nearest == torch.arange(dist.size(0))).sum().item()

    assert correct == dist.size(0), (
        f"Only {correct}/{dist.size(0)} runs decoded to their own delta. "
        f"Encoder is probably not discriminating between runs. "
        f"Distance matrix:\n{dist.numpy()}"
    )
