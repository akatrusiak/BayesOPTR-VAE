"""
surrogate.py — landscape surrogate ĝ(x, δ) → y for Tier 2 functional validation.

Motivation
----------
The VAE's training-time early-stopping target is δ-reconstruction MSE, but
what we actually care about is landscape fidelity: if we decode ẑ into
δ̂ and drop it into TRANSOPTR, does the resulting transmission landscape
match the run's observed landscape? The gold-standard answer is
verify_dz.py (re-run TRANSOPTR for every val run), which is too slow for
per-epoch early stopping.

This module provides a cheap approximation. We train a small MLP on the
(x, δ, y) triples extracted from existing TRANSOPTR runs:

    ĝ : (x ∈ R^d, δ ∈ R^{D_delta}) → y ∈ R

Then at VAE evaluation time we can compute:

    landscape MSE = (1/N) Σᵢ (y_true_i - ĝ(x_i, δ̂))²

without calling TRANSOPTR. This is "Tier 2" in the three-tier validation
plan; Tier 1 is val_recon (inline, every epoch), Tier 3 is verify_dz.py
(offline, post-training).

Caveats — read before using downstream
--------------------------------------
1. The (x_i, y_i) points in the training data are *SA trajectories*, not
   a grid. Each run's x values cluster near the optimum TRANSOPTR found
   for that run's δ. The surrogate is therefore most accurate in the
   x-region the SA explored, and may extrapolate poorly elsewhere.
   For the downstream use (comparing δ̂ candidates on a val run's own
   x-values), this is fine — we're interpolating in a region similar
   training runs explored. Don't use this surrogate to evaluate novel
   x-grids.

2. The input dimensionality is d + D_delta. For the TRIUMF pipeline this
   is 9 + 27 = 36. With ~240 runs × ~9 pairs ≈ 2160 training triples,
   this is only ~60 samples per input dimension. Weight decay and a
   narrow bottleneck matter.

3. We train at the *triple* level (N_total samples = sum of run lengths)
   but split at the *run* level (entire runs into train or val). This
   prevents leaking δ across splits: if triples from the same run ended
   up in both splits, a memorizing model would look great on val because
   it's seen the exact δ before.

Module layout
-------------
- LandscapeSurrogate: the MLP itself. Standardized in, standardized out.
- SurrogateTripleDataset: wraps MultiFolderRunDataset + a run-index list,
  yields (x_std, δ_std, y_std) triples.
- save_surrogate / load_surrogate: checkpoint I/O mirroring the VAE pattern.
- predict: convenience for downstream code that has physical-unit inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from vae_project.vae.data import MultiFolderRunDataset, StandardizationStats


# ═════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════

class LandscapeSurrogate(nn.Module):
    """
    MLP mapping (x, δ) → y. All inputs/outputs are STANDARDIZED.

    Args:
        d_input:  dimensionality of x (steerer settings; 9 for TRIUMF).
        d_delta:  dimensionality of δ (misalignment; 27 for TRIUMF).
        d_hidden: MLP hidden width. Default 128.
        n_hidden_layers: number of hidden layers (≥1). Default 3 hidden
            layers → 4 Linear layers total.
        dropout:  dropout between hidden layers. Default 0.0.

    Forward:
        x:     (..., d_input)  standardized
        delta: (..., d_delta)  standardized   (broadcastable with x)
        →      (...,)          standardized y

    We broadcast x and δ together before concatenating, which lets callers
    pass a single δ against a batch of x values (the common case at
    inference time).
    """
    def __init__(
        self,
        d_input: int,
        d_delta: int,
        d_hidden: int = 128,
        n_hidden_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")
        self.d_input = d_input
        self.d_delta = d_delta
        self.d_hidden = d_hidden
        self.n_hidden_layers = n_hidden_layers

        d_in = d_input + d_delta
        layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(d_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        # Broadcast δ to match x's leading dims, then concatenate.
        # Common case: x is (N, d_input) and δ is (d_delta,) → output (N,).
        # Also supports x=(B, N, d_input), δ=(B, d_delta) → (B, N).
        if delta.dim() == x.dim() - 1:
            delta = delta.unsqueeze(-2).expand(*x.shape[:-1], delta.shape[-1])
        elif delta.shape != x.shape[:-1] + (self.d_delta,):
            # User passed a shape we don't know how to broadcast. Let them fix it.
            raise ValueError(
                f"delta shape {tuple(delta.shape)} not broadcastable with "
                f"x shape {tuple(x.shape)}; expected delta to be "
                f"({'...', self.d_delta}) or ({'...', 'N', self.d_delta})"
            )
        xd = torch.cat([x, delta], dim=-1)
        return self.net(xd).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════
# Triple dataset (run-level split, triple-level training)
# ═════════════════════════════════════════════════════════════════════

class SurrogateTripleDataset(Dataset):
    """
    Flatten a MultiFolderRunDataset into (x, δ, y) triples for surrogate training.

    Only runs whose run_idx is in `run_indices` are included. This lets
    the caller perform a run-level train/val split and instantiate one
    SurrogateTripleDataset per split.

    All returned tensors are STANDARDIZED using the stats on `ds`.

    Attributes:
        n_triples: total number of (x, δ, y) triples.
        run_origin: np.array mapping triple index → run_idx it came from
            (useful for per-run error aggregation at eval time).
    """

    def __init__(
        self,
        ds: MultiFolderRunDataset,
        run_indices: Iterable[int],
    ):
        self.ds = ds
        self.run_indices = np.array(sorted(set(int(i) for i in run_indices)),
                                    dtype=np.int64)

        # Precompute flat triple arrays. Each row is one triple; we keep
        # them in-memory because the total size is ~(~2k triples × 36 dims
        # × 4 bytes) ≈ 300 KB for the TRIUMF data — trivial.
        x_list, d_list, y_list, origin_list = [], [], [], []
        for ri in self.run_indices:
            rows = ds._run_row_idx[ri]
            x_raw = ds.inputs[rows]            # (N_j, d_input), physical
            y_raw = ds.transmission[rows]      # (N_j,),         physical (0-1)
            delta_raw = ds._run_deltas[ri]     # (d_delta,),     physical

            x_std = ds._std_inputs(x_raw).astype(np.float32)
            y_std = ds._std_y(y_raw).astype(np.float32)
            d_std = ds._std_delta(delta_raw).astype(np.float32)

            # Broadcast δ across the run's rows so each triple is self-contained.
            N_j = x_std.shape[0]
            d_std_rep = np.broadcast_to(d_std, (N_j, d_std.shape[0])).copy()
            x_list.append(x_std)
            d_list.append(d_std_rep)
            y_list.append(y_std)
            origin_list.append(np.full(N_j, ri, dtype=np.int64))

        self._x = np.concatenate(x_list, axis=0)          # (N_triples, d_input)
        self._delta = np.concatenate(d_list, axis=0)      # (N_triples, d_delta)
        self._y = np.concatenate(y_list, axis=0)          # (N_triples,)
        self.run_origin = np.concatenate(origin_list, axis=0)
        self.n_triples = self._x.shape[0]
        self.d_input = self._x.shape[1]
        self.d_delta = self._delta.shape[1]

    def __len__(self) -> int:
        return self.n_triples

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._x[idx]),
            torch.from_numpy(self._delta[idx]),
            torch.tensor(self._y[idx], dtype=torch.float32),
        )


# ═════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ═════════════════════════════════════════════════════════════════════

@dataclass
class SurrogateConfig:
    d_input: int
    d_delta: int
    d_hidden: int
    n_hidden_layers: int
    dropout: float
    # Provenance — mirrors VAE config fields so downstream tools can
    # recover the train/val split this surrogate was fit against.
    seed: int
    val_frac: float
    n_runs_total: int
    data_source: list[str]

    def to_dict(self) -> dict:
        return {
            "d_input": self.d_input,
            "d_delta": self.d_delta,
            "d_hidden": self.d_hidden,
            "n_hidden_layers": self.n_hidden_layers,
            "dropout": self.dropout,
            "seed": self.seed,
            "val_frac": self.val_frac,
            "n_runs_total": self.n_runs_total,
            "data_source": self.data_source,
        }


def save_surrogate(
    path: str | Path,
    model: LandscapeSurrogate,
    stats: StandardizationStats,
    config: SurrogateConfig,
    epoch: int,
    val_metrics: dict,
):
    """Save in the same format the VAE uses, so downstream code can be symmetric."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "stats":       stats.to_dict(),
            "config":      config.to_dict(),
            "epoch":       int(epoch),
            "val_metrics": val_metrics,
        },
        Path(path),
    )


def load_surrogate(
    path: str | Path,
) -> tuple[LandscapeSurrogate, StandardizationStats, dict, int, dict]:
    """
    Load a surrogate checkpoint.

    Returns:
        model:        LandscapeSurrogate, weights loaded, eval mode
        stats:        StandardizationStats (the ones it was trained with)
        config:       the saved config dict (not a dataclass — raw dict for
                      flexibility on old checkpoints with missing keys)
        epoch:        training epoch of this checkpoint
        val_metrics:  dict of metrics at save time
    """
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    stats = StandardizationStats.from_dict(ck["stats"])
    model = LandscapeSurrogate(
        d_input=cfg["d_input"],
        d_delta=cfg["d_delta"],
        d_hidden=cfg.get("d_hidden", 128),
        n_hidden_layers=cfg.get("n_hidden_layers", 3),
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, stats, cfg, int(ck.get("epoch", -1)), ck.get("val_metrics", {})


# ═════════════════════════════════════════════════════════════════════
# Inference convenience
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(
    model: LandscapeSurrogate,
    x_phys: np.ndarray,          # (..., d_input)  physical
    delta_phys: np.ndarray,      # (..., d_delta)  physical
    stats: StandardizationStats,
) -> np.ndarray:
    """
    Predict transmission in physical (0-1) units from physical inputs.

    Standardizes inputs with `stats`, runs the model, un-standardizes
    the output. This is the function downstream scripts should call.

    Supports any leading-dim broadcasting that LandscapeSurrogate.forward
    supports. Common usage:

        # Single run: N steerer settings, one δ
        y_hat = predict(model, x_phys=(N, d), delta_phys=(d_delta,), stats=...)
        # → shape (N,), values in [~0, ~1]
    """
    x_std_np = (x_phys - stats.input_mean) / stats.input_std
    d_std_np = (delta_phys - stats.delta_mean) / stats.delta_std
    x_t = torch.from_numpy(x_std_np.astype(np.float32))
    d_t = torch.from_numpy(d_std_np.astype(np.float32))
    y_std_t = model(x_t, d_t)
    y_std_np = y_std_t.cpu().numpy()
    return y_std_np * stats.y_std + stats.y_mean
