"""
Phase 2 tests: DeepSets encoder invariances and gradient flow.

The most important test here is T2.1 (permutation invariance).
If it fails, the entire DeepSets premise is broken.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

from vae.encoder import DeepSetsEncoder


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return DeepSetsEncoder(d_input=10, d_hidden=32, d_latent=5)


@pytest.fixture
def fake_batch():
    """Build a deterministic (B=3, N_max=20, d_input=10) batch with per-run masks."""
    torch.manual_seed(1)
    B, N_max, D = 3, 20, 10
    pairs = torch.randn(B, N_max, D)
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    # Each run has a different true length
    true_lens = [20, 15, 7]
    for i, n in enumerate(true_lens):
        mask[i, :n] = True
    return pairs, mask, true_lens


# ----------------------------------------------------------------------
# T2.1 — Permutation invariance (most important test)
# ----------------------------------------------------------------------
def test_t2_1_permutation_invariance(encoder, fake_batch):
    pairs, mask, _ = fake_batch
    encoder.eval()

    with torch.no_grad():
        mu1, lv1 = encoder(pairs, mask)

        # Shuffle pair dimension *per-run* (different permutations per batch element
        # is even stricter than the same permutation)
        pairs_shuf = pairs.clone()
        mask_shuf = mask.clone()
        for i in range(pairs.size(0)):
            perm = torch.randperm(pairs.size(1))
            pairs_shuf[i] = pairs[i, perm]
            mask_shuf[i] = mask[i, perm]

        mu2, lv2 = encoder(pairs_shuf, mask_shuf)

    assert torch.allclose(mu1, mu2, atol=1e-6), (
        f"Permutation changed mu! max diff {(mu1 - mu2).abs().max()}"
    )
    assert torch.allclose(lv1, lv2, atol=1e-6), (
        f"Permutation changed log_var! max diff {(lv1 - lv2).abs().max()}"
    )


# ----------------------------------------------------------------------
# T2.2 — Variable-length invariance (padding doesn't leak)
# ----------------------------------------------------------------------
def test_t2_2_variable_length_invariance(encoder):
    torch.manual_seed(2)
    D = 10
    encoder.eval()

    # Build a "short" unpadded batch: one run of length 12
    pairs_short = torch.randn(1, 12, D)
    mask_short = torch.ones(1, 12, dtype=torch.bool)

    # Same run padded to length 30 with arbitrary garbage in padded slots
    pairs_long = torch.zeros(1, 30, D)
    pairs_long[0, :12] = pairs_short[0]
    pairs_long[0, 12:] = 999.0  # garbage should be ignored
    mask_long = torch.zeros(1, 30, dtype=torch.bool)
    mask_long[0, :12] = True

    with torch.no_grad():
        mu_s, lv_s = encoder(pairs_short, mask_short)
        mu_l, lv_l = encoder(pairs_long, mask_long)

    assert torch.allclose(mu_s, mu_l, atol=1e-6), (
        f"Padding leaked into mu (max diff {(mu_s - mu_l).abs().max()}). "
        f"Check the mask application in the pooling step."
    )
    assert torch.allclose(lv_s, lv_l, atol=1e-6), (
        f"Padding leaked into log_var (max diff {(lv_s - lv_l).abs().max()})."
    )


# ----------------------------------------------------------------------
# T2.3 — Output shape
# ----------------------------------------------------------------------
def test_t2_3_output_shape(encoder, fake_batch):
    pairs, mask, _ = fake_batch
    mu, log_var = encoder(pairs, mask)
    B = pairs.size(0)
    assert mu.shape == (B, encoder.d_latent)
    assert log_var.shape == (B, encoder.d_latent)


# ----------------------------------------------------------------------
# T2.4 — Gradient flow: every parameter gets a gradient
# ----------------------------------------------------------------------
def test_t2_4_gradient_flow(encoder, fake_batch):
    pairs, mask, _ = fake_batch
    encoder.train()

    mu, log_var = encoder(pairs, mask)
    loss = mu.sum() + log_var.sum()
    loss.backward()

    dead = []
    for name, p in encoder.named_parameters():
        if p.grad is None:
            dead.append(name)
        elif p.grad.abs().sum().item() == 0.0:
            dead.append(f"{name} (zero grad)")
    assert not dead, f"Parameters with no/zero gradient: {dead}"


# ----------------------------------------------------------------------
# Extra: log_var clamping is effective
# ----------------------------------------------------------------------
def test_log_var_clamping():
    """Set a very wide clamp range, then a tight one, and verify."""
    enc = DeepSetsEncoder(
        d_input=10, d_hidden=32, d_latent=5,
        log_var_min=-0.5, log_var_max=0.5,
    )
    # Push big inputs through to provoke large log_var outputs
    pairs = torch.randn(2, 10, 10) * 100
    mask = torch.ones(2, 10, dtype=torch.bool)
    _, lv = enc(pairs, mask)
    assert lv.min() >= -0.5 - 1e-6
    assert lv.max() <= 0.5 + 1e-6
