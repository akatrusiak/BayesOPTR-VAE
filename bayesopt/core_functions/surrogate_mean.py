"""
surrogate_mean.py — BoTorch Mean module wrapping a trained LandscapeSurrogate.

Lives inside BOIS and is instantiated by bayesopt.py when the Hydra
config sets `mean_function.surrogate_path` (otherwise the GP falls back
to BoTorch's default ConstantMean).

Design notes
------------
* The surrogate is loaded once at BO-loop construction, frozen, and
  queried at each GP refit with a fixed physical-space δ (the machine
  state for the BO trial).
* BOIS's x-ordering (all steerers then all quads, produced by
  tuning_elements=[steerer, quad]) does not match the surrogate's
  training column order (steerer/quad interleaved). A permutation is
  computed at construction time from the two name lists and asserted
  set-equal. A silent mismatch would be catastrophic, so this is a
  hard stop.
* GPyTorch Mean modules must produce shape (...,) — the trailing
  "d_output" dim is implicit. We squeeze.
* BoTorch's SingleTaskGP with Standardize(m=1) outcome transform
  handles standardization around the mean internally. Our module
  returns physical-space predictions and BoTorch does the bookkeeping.

Checkpoint contract
-------------------
Relies on `surrogate.load_surrogate` returning (model, stats, cfg,
epoch, metrics) where `stats` has attributes input_mean, input_std,
y_mean, y_std, delta_mean, delta_std (all numpy arrays or scalars).
This matches surrogate.py from the VAE project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from gpytorch.means import Mean


class SurrogateMean(Mean):
    """
    Wrap a LandscapeSurrogate + fixed δ as a frozen GP mean function.

    Parameters
    ----------
    surrogate_path : str | Path
        Path to a surrogate best.pt produced by train_surrogate.py.
    delta_yaml_path : str | Path
        Path to a Misalignments.yaml file in the BOIS format:
            MISALIGNX1: 0.0
            ...
            HEBT2:Q7:MISALIGNY: 0.025
        All 27 misalignment names the training data used must be keys.
    delta_names : Sequence[str]
        Ordered list of delta-dimension names, in the EXACT order the
        surrogate was trained on. Obtain this from
        `MultiFolderRunDataset(...).delta_names`.
    surrogate_input_names : Sequence[str]
        Ordered list of input-dimension names the surrogate was trained
        on. Obtain from `MultiFolderRunDataset(...).input_names`. For
        this project: XCB2, YCB2, XCB4, YCB4, Q5, Q6, XCB6, YCB6, Q7.
    bois_input_names : Sequence[str]
        Ordered list of input-dimension names BOIS constructs at runtime.
        For tuning_elements=[steerer, quad] this is all steerers then
        all quads (see the last row of training_set.csv for the
        runtime-observed ordering). Used only to compute a permutation
        to the surrogate's order — the surrogate is the source of truth.
    device : str
        Torch device for the forward pass. Default "cpu".
    """

    def __init__(
        self,
        surrogate_path: str | Path,
        delta_yaml_path: str | Path,
        delta_names: Sequence[str],
        surrogate_input_names: Sequence[str],
        bois_input_names: Sequence[str],
        device: str = "cpu",
    ):
        super().__init__()

        # ── load surrogate ──────────────────────────────────────
        # Deferred import so this module can be dropped into BOIS without
        # requiring the surrogate package at import time.
        from surrogate import load_surrogate  # type: ignore

        surrogate_path = Path(surrogate_path)
        if not surrogate_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint not found: {surrogate_path}")

        self.surrogate, self.stats, self.cfg, _, _ = load_surrogate(surrogate_path)
        # BOIS's default dtype is float64; promote the surrogate once so we
        # don't have to juggle dtypes inside forward(). The surrogate was
        # trained in float32 — widening is safe (zero info loss).
        self.surrogate.to(device=device, dtype=torch.float64).eval()
        for p in self.surrogate.parameters():
            p.requires_grad_(False)
        self._device = device

        # ── sanity-check dims ───────────────────────────────────
        if len(delta_names) != self.cfg["d_delta"]:
            raise ValueError(
                f"delta_names has {len(delta_names)} entries but surrogate "
                f"was trained with d_delta={self.cfg['d_delta']}"
            )
        if len(surrogate_input_names) != self.cfg["d_input"]:
            raise ValueError(
                f"surrogate_input_names has {len(surrogate_input_names)} "
                f"entries but surrogate was trained with d_input={self.cfg['d_input']}"
            )

        # ── load and standardize delta ──────────────────────────
        delta_yaml_path = Path(delta_yaml_path)
        if not delta_yaml_path.exists():
            raise FileNotFoundError(f"Delta yaml not found: {delta_yaml_path}")
        with open(delta_yaml_path) as f:
            delta_dict = yaml.safe_load(f) or {}

        missing = [n for n in delta_names if n not in delta_dict]
        if missing:
            raise ValueError(
                f"Delta yaml {delta_yaml_path} is missing {len(missing)} "
                f"keys the surrogate expects: {missing[:5]}..."
            )

        delta_phys = np.array([float(delta_dict[n]) for n in delta_names],
                              dtype=np.float64)
        delta_std_np = (delta_phys - self.stats.delta_mean) / self.stats.delta_std
        self.register_buffer(
            "delta_std",
            torch.tensor(delta_std_np, dtype=torch.float64, device=device),
        )

        # ── build permutation from BOIS x-order to surrogate x-order ──
        # permutation[i] = index in BOIS's x-vector of the i-th surrogate column.
        bois_input_names = list(bois_input_names)
        surrogate_input_names = list(surrogate_input_names)
        if set(bois_input_names) != set(surrogate_input_names):
            extra_in_bois = set(bois_input_names) - set(surrogate_input_names)
            missing_in_bois = set(surrogate_input_names) - set(bois_input_names)
            raise ValueError(
                "Input-name mismatch between BOIS and the surrogate.\n"
                f"  Missing from BOIS:     {sorted(missing_in_bois)}\n"
                f"  Extra in BOIS:         {sorted(extra_in_bois)}\n"
                "Either the BOIS sim config or the surrogate training data "
                "is out of date."
            )
        permutation = [bois_input_names.index(n) for n in surrogate_input_names]
        self.register_buffer(
            "permutation",
            torch.tensor(permutation, dtype=torch.long, device=device),
        )

# ── standardization buffers (input and output) ──────────
        # "surr_*" = stats the SURROGATE was trained with (frozen).
        # "gp_*"   = stats the GP's Standardize uses this iteration
        #            (refreshed per-iteration by set_context()).
        self.register_buffer(
            "input_mean",
            torch.tensor(self.stats.input_mean, dtype=torch.float64, device=device),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(self.stats.input_std, dtype=torch.float64, device=device),
        )
        self.register_buffer(
            "y_mean_surr",
            torch.tensor(float(self.stats.y_mean), dtype=torch.float64, device=device),
        )
        self.register_buffer(
            "y_std_surr",
            torch.tensor(float(self.stats.y_std), dtype=torch.float64, device=device),
        )

        # ── GP-context buffers (filled per-iteration by getModel) ───────
        # SingleTaskGP wraps this mean module inside Normalize (on inputs)
        # and Standardize (on outputs). The mean module therefore:
        #   - receives x in [0,1]^d  →  needs bounds to un-normalize
        #   - must return y in the GP's standardized y-space
        #     →  needs the current Standardize.means / .stdvs
        # The y-stats change every iteration (train_Y grows), so we update
        # these buffers from getModel._standard after each GP construction.
        d = len(bois_input_names)
        self.register_buffer("bounds_lo", torch.zeros(d, dtype=torch.float64, device=device))
        self.register_buffer("bounds_hi", torch.ones(d, dtype=torch.float64, device=device))
        self.register_buffer("y_mean_gp", torch.zeros((), dtype=torch.float64, device=device))
        self.register_buffer("y_std_gp",  torch.ones((),  dtype=torch.float64, device=device))

    def set_context(
        self,
        bounds: torch.Tensor,
        y_mean_gp: torch.Tensor,
        y_std_gp: torch.Tensor,
    ) -> None:
        """
        Push per-iteration GP context into the mean module.

        Must be called by getModel._standard() after SingleTaskGP is
        constructed (so Standardize has computed its stats) and before
        fit_gpytorch_mll is called (so those stats are visible during
        MLL evaluation).
        """
        self.bounds_lo.data.copy_(bounds[0].to(self.bounds_lo.dtype))
        self.bounds_hi.data.copy_(bounds[1].to(self.bounds_hi.dtype))
        self.y_mean_gp.data.copy_(y_mean_gp.detach().squeeze().to(self.y_mean_gp.dtype))
        self.y_std_gp.data.copy_(y_std_gp.detach().squeeze().to(self.y_std_gp.dtype))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Evaluate surrogate(x, δ_fixed) for each x in the batch.

        Space conversions (critical — getting any of these wrong silently
        poisons the MLL fit, see git history for the 30-iter crash bug):

            x arrives here in [0,1]^d (after BoTorch Normalize)
              → un-normalize with `bounds` →   x_phys    (BOIS order)
              → permute                   →   x_surr    (surrogate order)
              → standardize, surrogate-stats → x_std
              → surrogate                 →   y_surr_std
              → un-standardize (surrogate y-stats) → y_phys   (≈0…1)
              → standardize (GP y-stats)  →   y_gp_std
              ← returned (this is the space the kernel is fitting)

        Shape contract (GPyTorch Mean): output has no trailing dim.

        Parameters
        ----------
        x_norm : (..., d_input) in [0,1]
            GPyTorch may pass 2D (n, d) during MLL fitting and 3D
            (batch, n, d) during acquisition-function optimization.
        """
        # Surrogate and all buffers are float64 (see __init__); BOIS runs
        # in float64; no casting is needed here. One flatten, one reshape.
        leading_shape = x_norm.shape[:-1]
        x_flat = x_norm.reshape(-1, x_norm.shape[-1])

        # Undo BoTorch's Normalize: x_phys = lo + x_norm * (hi - lo).
        x_phys = self.bounds_lo + x_flat * (self.bounds_hi - self.bounds_lo)

        # Reorder BOIS columns → surrogate columns.
        x_surr = x_phys.index_select(-1, self.permutation)

        # Standardize input with surrogate's training stats.
        x_std = (x_surr - self.input_mean) / self.input_std

        # Run surrogate. NOTE: NO torch.no_grad here. The acquisition-
        # function optimizer needs ∂μ/∂x, and the prior mean contributes
        # to μ. Parameters are already frozen via requires_grad_(False)
        # in __init__, so the surrogate's weights won't accumulate grads
        # — only the input gradient flows.
        y_surr_std = self.surrogate(x_std, self.delta_std)

        # Un-standardize to physical transmission (≈ 0…1).
        y_phys = y_surr_std * self.y_std_surr + self.y_mean_surr

        # Re-standardize into the GP's output space so residuals fed to
        # the kernel are on ~unit scale. y_mean_gp/y_std_gp come from
        # set_context(), called per iteration.
        y_gp_std = (y_phys - self.y_mean_gp) / self.y_std_gp

        return y_gp_std.reshape(leading_shape)


# ──────────────────────────────────────────────────────────────
# Convenience constructor that BOIS can call with a small config
# ──────────────────────────────────────────────────────────────

def build_surrogate_mean_from_config(
    mean_cfg,
    bois_input_names: Sequence[str],
) -> "SurrogateMean | None":
    """
    Instantiate SurrogateMean from a Hydra config subtree.

    Expected config shape (yaml):
        mean_function:
            surrogate_path: /path/to/best.pt
            delta_yaml: /path/to/Misalignments.yaml
            delta_names: [MISALIGNX1, MISALIGNY1, ..., HEBT2:Q7:MISALIGNY]
            surrogate_input_names: [HEBT2:XCB2, ..., HEBT2:Q7:CUR]
            device: cpu       # optional

    Returns None if mean_cfg is None or mean_cfg.surrogate_path is null.
    """
    if mean_cfg is None:
        return None
    if getattr(mean_cfg, "surrogate_path", None) in (None, "", "null"):
        return None

    return SurrogateMean(
        surrogate_path=mean_cfg.surrogate_path,
        delta_yaml_path=mean_cfg.delta_yaml,
        delta_names=list(mean_cfg.delta_names),
        surrogate_input_names=list(mean_cfg.surrogate_input_names),
        bois_input_names=list(bois_input_names),
        device=getattr(mean_cfg, "device", "cpu"),
    )
