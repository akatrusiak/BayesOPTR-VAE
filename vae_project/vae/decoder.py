"""
Misalignment decoder.

MLP that maps z_machine -> reconstructed misalignment vector (standardized).
Deliberately simple: only offsets/tilts, no fringe fields, no calibration offsets.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .encoder import MLP


class MisalignmentDecoder(nn.Module):
    """
    z -> delta_hat (both in standardized space).

    Args:
        d_latent: z dimensionality.
        d_hidden: hidden width.
        d_delta: misalignment vector dimension.
        depth: MLP depth.
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        d_delta: int,
        depth: int = 3,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.d_delta = d_delta
        self.net = MLP(d_latent, d_hidden, d_delta, depth=depth)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
