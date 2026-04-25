"""
Phase 3 tests: decoder shape + overfitting sanity check.

T3.2 is the critical one — it isolates "can the decoder even represent
these misalignments?" from full VAE training dynamics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn.functional as F

from vae.decoder import MisalignmentDecoder


# ----------------------------------------------------------------------
# T3.1 — Output shape
# ----------------------------------------------------------------------
def test_t3_1_output_shape():
    dec = MisalignmentDecoder(d_latent=5, d_hidden=64, d_delta=27)
    z = torch.randn(4, 5)
    out = dec(z)
    assert out.shape == (4, 27)


# ----------------------------------------------------------------------
# T3.2 — Overfit a single (z, delta) pair
# ----------------------------------------------------------------------
def test_t3_2_overfit_single_pair():
    """
    Train the decoder alone to map z=0 -> a fixed target delta*.
    MSE should drop to near zero within a few hundred steps.

    If this fails:
      - hidden width is too small,
      - learning rate is bad,
      - or there's a wiring bug.
    """
    torch.manual_seed(0)
    dec = MisalignmentDecoder(d_latent=5, d_hidden=64, d_delta=27, depth=3)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-2)

    z = torch.zeros(1, 5)
    delta_star = torch.randn(1, 27)

    losses = []
    for _ in range(500):
        opt.zero_grad()
        pred = dec(z)
        loss = F.mse_loss(pred, delta_star)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    final = losses[-1]
    assert final < 1e-4, (
        f"Decoder could not overfit a single (z, delta) pair. "
        f"Final MSE = {final:.3e}. Losses: start={losses[0]:.3e}, "
        f"mid={losses[250]:.3e}, end={final:.3e}"
    )


# ----------------------------------------------------------------------
# Extra: gradient flow
# ----------------------------------------------------------------------
def test_decoder_gradient_flow():
    dec = MisalignmentDecoder(d_latent=5, d_hidden=64, d_delta=27)
    z = torch.randn(3, 5, requires_grad=True)
    out = dec(z)
    out.sum().backward()
    for name, p in dec.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.abs().sum().item() > 0, f"Zero grad for {name}"
