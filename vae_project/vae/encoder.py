"""
DeepSets encoder for tuning-run sets.

Given a set of (x_i, y_i) pairs from a single run, the encoder produces
a posterior q(z | set) = N(mu, diag(sigma^2)).

Key properties that Phase 2 tests verify:
  - Permutation invariance: shuffling the set does not change (mu, log_var).
  - Variable-length invariance: padded and unpadded identical runs agree.
  - Masked pooling: padded positions contribute nothing.

log_var is clamped to a reasonable range to prevent numerical blow-ups
(sigma ~ 0 causes KL to explode; sigma huge makes reconstruction unused).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP with GELU activations. Shared across whatever axes you apply it to."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        depth: int = 3,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        layers: list[nn.Module] = []
        d_prev = d_in
        for i in range(depth - 1):
            layers.append(nn.Linear(d_prev, d_hidden))
            layers.append(activation())
            d_prev = d_hidden
        layers.append(nn.Linear(d_prev, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSetsEncoder(nn.Module):
    """
    DeepSets encoder.

    Args:
        d_input: dimensionality of one (x, y) pair, i.e. D_input + 1.
        d_hidden: hidden dim of both psi and rho MLPs.
        d_latent: latent dimensionality d_z.
        psi_depth: depth of per-pair MLP.
        rho_depth: depth of post-pool MLP.
        log_var_min: clamp floor for log_var.
        log_var_max: clamp ceiling for log_var.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_latent: int,
        psi_depth: int = 3,
        rho_depth: int = 2,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # Per-pair embedding, shared weights across all pairs.
        self.psi = MLP(d_input, d_hidden, d_hidden, depth=psi_depth)
        # Post-pool: outputs (mu, log_var) concatenated.
        self.rho = MLP(d_hidden, d_hidden, 2 * d_latent, depth=rho_depth)

    def forward(
        self, pairs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pairs: (B, N_max, d_input) — padded pair batch.
            mask: (B, N_max) bool — True where real data.

        Returns:
            mu: (B, d_latent)
            log_var: (B, d_latent) — clamped
        """
        # Per-pair features
        h = self.psi(pairs)                                  # (B, N_max, d_hidden)

        # Masked mean pool. Zero out padding before summing; divide by true count.
        mask_f = mask.unsqueeze(-1).to(h.dtype)              # (B, N_max, 1)
        h_masked = h * mask_f
        n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)  # (B, 1)
        pooled = h_masked.sum(dim=1) / n_valid               # (B, d_hidden)

        out = self.rho(pooled)                               # (B, 2*d_latent)
        mu, log_var = out.chunk(2, dim=-1)
        log_var = log_var.clamp(self.log_var_min, self.log_var_max)
        return mu, log_var
