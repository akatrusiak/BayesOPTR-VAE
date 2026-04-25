"""
VAE wrapper: encoder + decoder + ELBO.

Loss returns a dict with the individual terms (recon, kl, total) because
posterior collapse and KL explosion are invisible in the combined loss
but obvious when the components are tracked separately.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import DeepSetsEncoder
from .decoder import MisalignmentDecoder


@dataclass
class ELBOMetrics:
    """All values are per-run (averaged over the batch)."""
    total: float
    recon: float
    kl: float
    mu_norm: float            # mean ||mu|| across batch
    sigma_mean: float         # mean sigma across batch and latent dims
    per_dim_kl: list[float]   # per-latent-dim KL, useful for detecting dead dims


class VAE(nn.Module):
    """
    Standard Gaussian VAE with DeepSets encoder and MLP decoder.

    Reconstruction likelihood: N(delta_hat, sigma_x^2 I) with fixed sigma_x,
    which reduces log p(delta|z) to a constant plus (-1/(2 sigma_x^2)) * MSE.
    We roll the constant and sigma_x into the recon_weight scalar.

    Args:
        encoder: DeepSetsEncoder.
        decoder: MisalignmentDecoder.
        beta: KL weight (beta-VAE). Start at 1.0; anneal if posterior
              collapses.
        recon_weight: scales the MSE term. Default 1.0 (assumes standardized
                      deltas so per-dim variance ~1, making this equivalent
                      to a Gaussian likelihood with sigma=1).
    """

    def __init__(
        self,
        encoder: DeepSetsEncoder,
        decoder: MisalignmentDecoder,
        beta: float = 1.0,
        recon_weight: float = 1.0,
        free_bits: float = 0.0,
    ):
        """
        free_bits: per-latent-dim KL floor (nats). The effective KL term
            for dim j becomes max(free_bits, KL_j). This prevents posterior
            collapse by refusing to reward the optimizer for driving a dim
            all the way to the prior — the KL gradient is zero below the
            floor, so the encoder is free to use that dim without KL cost.
            Typical values: 0.05–0.5. 0 disables (standard ELBO).
        """
        super().__init__()
        assert encoder.d_latent == decoder.d_latent, (
            f"Latent dim mismatch: encoder={encoder.d_latent}, decoder={decoder.d_latent}"
        )
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.recon_weight = recon_weight
        self.free_bits = free_bits

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, pairs: torch.Tensor, mask: torch.Tensor, sample: bool = True
    ) -> dict:
        mu, log_var = self.encoder(pairs, mask)
        z = self.reparameterize(mu, log_var) if sample else mu
        delta_hat = self.decoder(z)
        return {"mu": mu, "log_var": log_var, "z": z, "delta_hat": delta_hat}

    # ------------------------------------------------------------------
    # ELBO
    # ------------------------------------------------------------------
    def elbo_loss(
        self,
        pairs: torch.Tensor,
        mask: torch.Tensor,
        delta_true: torch.Tensor,
    ) -> tuple[torch.Tensor, ELBOMetrics]:
        """
        Returns:
            loss: scalar tensor (to .backward() on).
            metrics: ELBOMetrics with the individual components for logging.
        """
        out = self.forward(pairs, mask, sample=True)
        mu, log_var, delta_hat = out["mu"], out["log_var"], out["delta_hat"]
        B = pairs.size(0)

        # Reconstruction: sum over delta dims, mean over batch.
        recon = F.mse_loss(delta_hat, delta_true, reduction="sum") / B

        # KL(N(mu, diag(sigma^2)) || N(0, I)), standard closed form.
        # KL per dim = 0.5 * (mu^2 + sigma^2 - 1 - log sigma^2)
        per_dim_kl = 0.5 * (mu.pow(2) + log_var.exp() - 1 - log_var)  # (B, d_z)
        per_dim_kl_mean = per_dim_kl.mean(dim=0)  # (d_z,) — useful for dead-dim check
        kl = per_dim_kl.sum(dim=1).mean()  # scalar (true KL, for logging)

        # Free bits: replace each dim's batch-mean KL with max(floor, kl_j).
        # Apply on the batch-averaged per-dim KL so the floor is a "per-dim
        # budget" rather than applied per-sample (Kingma et al., 2016).
        if self.free_bits > 0.0:
            kl_for_loss = torch.clamp(per_dim_kl_mean, min=self.free_bits).sum()
        else:
            kl_for_loss = kl

        loss = self.recon_weight * recon + self.beta * kl_for_loss

        with torch.no_grad():
            sigma = torch.exp(0.5 * log_var)
            metrics = ELBOMetrics(
                total=loss.item(),
                recon=recon.item(),
                kl=kl.item(),
                mu_norm=mu.norm(dim=1).mean().item(),
                sigma_mean=sigma.mean().item(),
                per_dim_kl=per_dim_kl_mean.tolist(),
            )
        return loss, metrics
