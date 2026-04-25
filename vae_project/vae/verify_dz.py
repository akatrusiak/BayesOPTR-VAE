"""
verify_dz.py — functional validation for the latent-dimensionality sweep.

What this script does
---------------------
Given:
  - One or more trained VAE checkpoints (each corresponding to a different d_z)
  - A pooled dataset of TRANSOPTR tuning runs (same format as training)
  - A TRANSOPTR verification directory (config + data.dat + sy.f, set up to
    produce a single fort.envelope at the tuned optimum)

we verify each checkpoint by, for every run in the verification set:

    1. TRUTH PATH
       Take the true misalignment vector delta_true for this run.
       The dataset already contains the tuned transmission T_true — it is
       the *last* transmission value along the SA trajectory (TRANSOPTR's
       internal optimum for delta_true). No re-simulation needed for truth;
       the training data already tells us the optimum TRANSOPTR reaches.

    2. RECONSTRUCTION PATH
       a. Encode the run (full set of pairs, as at training time) to get
          mu_phi.  z_hat = mu_phi (deterministic; we want the best point
          estimate, not a sample).
       b. Decode z_hat -> delta_hat (standardized) -> unstandardize.
       c. Inject delta_hat into the verification TRANSOPTR directory,
          call pyoptr.run, let TRANSOPTR tune to optimum.
       d. Read the *single* fort.envelope that falls out (the verification
          data.dat is configured to dump only the final optimum) and
          compute T_hat from it. Also extract the final tuned parameters.

    3. Compare T_hat vs T_true. Report per-run and aggregate.
       Also record ||delta_hat - delta_true|| per-dim, which is a weaker
       signal (per the proposal's warning) but still informative.

       Aggregation is split into val-only (the headline), train-only
       (memorization check), and all-runs. The val set for each checkpoint
       is recovered from its recorded (seed, val_frac) — see
       recover_split() below.

Why "last point only" for the reconstruction side
-------------------------------------------------
The user-side note: we only care about the *converged* transmission that
TRANSOPTR finds, not the whole SA trajectory. Running full SA for every
reconstruction would be expensive, and the verification question is
"does delta_hat let TRANSOPTR reach the same optimum as delta_true?",
which is a one-number-per-run comparison. The verification TRANSOPTR dir
is configured to write only one fort.envelope (the final one). This works
through run_sample_on_axis's existing fallback branch which already looks
for a single archive_dir/fort.envelope when the per-iteration directory
doesn't exist.

Why the encoder still sees the full set
---------------------------------------
The encoder is permutation- and length-invariant by design and was trained
on full SA trajectories. Feeding it a truncated set at inference time
would be out of distribution. So: full set in for the encode step, single
envelope out of the decode+re-simulate step.

CLI
---
  python -m verification.verify_dz \\
      --checkpoint runs/vae_dz3/best.pt \\
      --checkpoint runs/vae_dz5/best.pt \\
      --checkpoint runs/vae_dz8/best.pt \\
      --checkpoint runs/vae_dz12/best.pt \\
      --data_dir data/output_good_4 \\
      --data_dir data/output_good_5 \\
      --verify_dir path/to/transoptr_verification \\
      --pipeline_dir path/to/pipeline \\
      --config path/to/transoptr_verification/config.yaml \\
      --axis verify \\
      --out_dir runs/verification \\
      [--max_runs N] [--seed 0]
      [--seed_override S] [--val_frac_override F]

Outputs
-------
  out_dir/verify_dz{N}.json          per-checkpoint, per-run metrics
  out_dir/summary.json               aggregate across all checkpoints
  (Plots are left to a separate script so that re-plotting doesn't
   require re-running TRANSOPTR.)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# -------- VAE package (same tree as this file) --------
# verification/ and vae/ are siblings. Running as a module
# (`python -m verification.verify_dz ...`) handles this for us, but the
# explicit path insert helps if someone runs the file directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .data import MultiFolderRunDataset, RunLabel, StandardizationStats, collate_runs
from .decoder import MisalignmentDecoder
from .encoder import DeepSetsEncoder
from .model import VAE


# ═════════════════════════════════════════════════════════════════════
# Pipeline bridge
# ═════════════════════════════════════════════════════════════════════
#
# We reuse init_axis / run_sample_on_axis from the simulation-generation
# pipeline rather than re-implementing them. The pipeline repo lives
# *outside* this project by design, so we add it to sys.path at runtime.
# No pipeline files are modified.
#
# What we borrow:
#   init_axis(axis_name, axis_spec, global_config, root_dir) -> dict
#       Compiles optr (if needed), loads data.dat, runs the baseline
#       s-grid, builds aperture/FC indices, returns an axis_state dict.
#   run_sample_on_axis(axis_state, mis_dict, global_config) -> dict
#       Injects mis_dict into data.dat, calls pyoptr.run, parses whatever
#       envelope files land in archive_dir (single or multi). Returns
#       transmissions, inputs, n_envelopes, fc_index. We want
#       result["transmissions"][-1] (the FC transmission at the final
#       tuned point) and the last row of result["inputs"] (the tuned
#       parameters).
# ---------------------------------------------------------------------

def _import_pipeline(pipeline_dir: Path):
    """Make pipeline imports available. Called once, per-run."""
    p = str(pipeline_dir.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    # These imports fail in the VAE dev environment (no pyoptr), which is
    # fine — the script is expected to run where the pipeline does.
    import generate_data as _gd  # noqa: F401  (re-exported via return)
    return _gd


# ═════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ═════════════════════════════════════════════════════════════════════

def load_vae_checkpoint(ckpt_path: Path) -> tuple[VAE, StandardizationStats, dict]:
    """
    Rebuild a VAE from a saved checkpoint.

    The checkpoint format is the one saved by vae/train.py::_save_checkpoint:
      {
        "model_state": state_dict,
        "stats":       StandardizationStats.to_dict(),
        "config":      {d_input, d_delta, d_latent, d_hidden, beta,
                        data_source, n_runs, seed, val_frac},
        "epoch":       int,
        "val_metrics": {...},
      }

    Returns:
        vae:   VAE (in eval mode, weights loaded)
        stats: StandardizationStats   — NOT the same as the dataset's stats;
                                         we should use THESE at inference
                                         time to match what the model saw
                                         during training.
        cfg:   the checkpoint's config dict
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    stats = StandardizationStats.from_dict(ckpt["stats"])

    encoder = DeepSetsEncoder(
        d_input=cfg["d_input"] + 1,      # +1 for transmission column in the pair
        d_hidden=cfg.get("d_hidden", 128),
        d_latent=cfg["d_latent"],
    )
    decoder = MisalignmentDecoder(
        d_latent=cfg["d_latent"],
        d_hidden=cfg.get("d_hidden", 128),
        d_delta=cfg["d_delta"],
    )
    # NOTE: d_hidden is not currently saved in the checkpoint config
    # (only d_input / d_delta / d_latent / beta are). The .get(..., 128)
    # fallback above matches train.py's default; if a non-default hidden
    # size is ever used we should persist it. TODO: add d_hidden to
    # _save_checkpoint in train.py.
    vae = VAE(encoder, decoder, beta=cfg.get("beta", 1.0))
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()
    return vae, stats, cfg


# ═════════════════════════════════════════════════════════════════════
# Train/val split recovery
# ═════════════════════════════════════════════════════════════════════

def recover_split(n_items: int, val_frac: float = 0.2, seed: int = 0):
    """
    Reconstruct the exact train/val indices used by train_val_split in train.py.

    Must mirror train_val_split's logic exactly: legacy RandomState (NOT
    default_rng), shuffle of np.arange(n_items), sorted() on both halves.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n_items)))
    val_idx = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    return train_idx, val_idx


def recover_splits_for_checkpoint(
    ckpt_cfg: dict,
    n_ds_items: int,
    seed_override: Optional[int] = None,
    val_frac_override: Optional[float] = None,
) -> set:
    """
    Given a checkpoint's config dict, return the set of val indices the
    checkpoint was trained against. Overrides let the caller pass seed /
    val_frac explicitly when older checkpoints didn't record them.
    """
    seed = seed_override if seed_override is not None else ckpt_cfg.get("seed")
    val_frac = (
        val_frac_override if val_frac_override is not None
        else ckpt_cfg.get("val_frac", 0.2)
    )
    if seed is None:
        raise ValueError(
            "Checkpoint config has no 'seed' field and no --seed_override was "
            "provided. Either re-train with the updated train.py (which saves "
            "the seed) or pass --seed_override on the CLI."
        )

    n_runs = ckpt_cfg.get("n_runs", n_ds_items)
    if n_runs != n_ds_items:
        raise ValueError(
            f"Dataset size mismatch: checkpoint saw n_runs={n_runs}, current "
            f"dataset has {n_ds_items}. Cannot recover split safely. "
            f"Check that --data_dir args match what the checkpoint was "
            f"trained on."
        )

    _, val_idx = recover_split(n_ds_items, val_frac=val_frac, seed=seed)
    return set(val_idx)


# ═════════════════════════════════════════════════════════════════════
# Encode + decode one run
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def reconstruct_delta(
    vae: VAE,
    stats: StandardizationStats,
    pairs_std: torch.Tensor,    # (N_j, D_input + 1), standardized
    mask: torch.Tensor,         # (N_j,) bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode a run -> mu -> decode -> unstandardize -> delta_hat (physical units).

    Returns:
        delta_hat: (D_delta,) float64, PHYSICAL units (unstandardized)
        mu:        (d_latent,)
        log_var:   (d_latent,)
    """
    # Encoder expects a batch; wrap to (1, N_j, D_input+1) and (1, N_j).
    batch_pairs = pairs_std.unsqueeze(0)
    batch_mask = mask.unsqueeze(0)

    mu, log_var = vae.encoder(batch_pairs, batch_mask)
    # Use mu as the point estimate. Sampling would add noise that we don't
    # want for evaluation — the sample/decode roundtrip is a training-time
    # device for spreading mass over the posterior, not a prediction step.
    delta_hat_std = vae.decoder(mu).squeeze(0).cpu().numpy()  # (D_delta,)

    # Move back to physical units using the CHECKPOINT stats (not the
    # current dataset's), because those are what the decoder was trained
    # to predict into.
    delta_hat = delta_hat_std * stats.delta_std + stats.delta_mean
    
    #Debug -- bypass decoder and set to true value to check the rest of the pipeline
    return delta_hat, mu.squeeze(0).cpu().numpy(), log_var.squeeze(0).cpu().numpy()


def build_mis_dict(
    delta_values: np.ndarray,
    delta_names: np.ndarray,
) -> "OrderedDict[str, float]":
    """
    Build the OrderedDict that run_sample_on_axis expects.

    The names come from the dataset's misalignment_names array (the same
    names TRANSOPTR expects in data.dat). run_sample_on_axis already
    silently skips names not present in the current data.dat via its
    try/except ValueError on update_element — so if the verification
    data.dat has a subset of element names, that's fine.
    """
    d = OrderedDict()
    for name, v in zip(delta_names.tolist(), delta_values.tolist()):
        d[str(name)] = float(v)
    return d


# ═════════════════════════════════════════════════════════════════════
# Per-run verification
# ═════════════════════════════════════════════════════════════════════

def verify_one_run(
    ds: MultiFolderRunDataset,
    run_idx: int,
    vae: VAE,
    ckpt_stats: StandardizationStats,
    axis_state: dict,
    global_config: dict,
    gd_module,
    is_val: bool,
    delta_source: str,
) -> dict:
    """
    Run the full verification pipeline on one dataset run.

    Steps (numbered to match the module docstring):
        [1] Read T_true from the training-set row for this run's final
            trajectory point.
        [2a,b] Encode (full set) -> mu -> decode -> delta_hat.
        [2c,d] Inject delta_hat into verification TRANSOPTR, let it tune,
               read back T_hat and the tuned parameters.
        [3] Compute metrics.

    Returns a dict with per-run metrics, serializable to JSON. The
    `was_val` flag lets the aggregator split train vs val after the fact.
    """
    # ----- dataset row -----
    # Note: ds[run_idx] returns STANDARDIZED pairs/delta (with the
    # dataset's stats). For the encoder we want pairs standardized with
    # THE CHECKPOINT'S stats, which may differ from the current dataset's
    # if we're verifying on a superset of what we trained on. We therefore
    # grab raw rows and re-standardize manually here.
    rows = ds._run_row_idx[run_idx]
    x_raw = ds.inputs[rows]                    # (N_j, D_input), physical units
    y_raw = ds.transmission[rows]              # (N_j,)           physical units
    delta_true = ds._run_deltas[run_idx]       # (D_delta,),      physical units
    run_label: RunLabel = ds.run_labels[run_idx]

    # Re-standardize with checkpoint stats (defensive — same model contract
    # as at training time).
    x_std = (x_raw - ckpt_stats.input_mean) / ckpt_stats.input_std
    y_std = (y_raw - ckpt_stats.y_mean) / ckpt_stats.y_std
    pairs_std = np.concatenate([x_std, y_std[:, None]], axis=1).astype(np.float32)
    pairs_t = torch.from_numpy(pairs_std)
    mask_t = torch.ones(pairs_t.shape[0], dtype=torch.bool)

    # ----- [1] T_true from training data -----
    # The SA trajectory is in row order (per the MultiFolderRunDataset
    # docstring in data.py), and the final row is the optimum TRANSOPTR
    # found for this delta_true. So the "true" landscape optimum that we
    # want to compare against is simply y_raw[-1]; no re-simulation
    # needed.
    # NOTE: `rows` was already sorted in SA-time order during dataset
    # construction, so y_raw[-1] corresponds to the last SA step.
    t_true = float(y_raw[-1])

    # ----- [2a,b] encode -> decode -> delta_hat -----
    delta_decoded, mu, log_var = reconstruct_delta(vae, ckpt_stats, pairs_t, mask_t)
    
    if delta_source == "decoded":
        delta_hat = delta_decoded
    elif delta_source == "true":
        delta_hat = delta_true.copy()
    elif delta_source in ("zero_std", "mean"):
        # δ_std = 0 → δ_phys = δ_mean
        delta_hat = ckpt_stats.delta_mean.astype(np.float64).copy()
    elif delta_source == "zero_phys":
        delta_hat = np.zeros_like(delta_true)
    else:
        raise ValueError(f"Unknown delta_source: {delta_source}")

    # ----- [2c,d] re-simulate with delta_hat -----
    # The verification data.dat is configured to emit ONE fort.envelope at
    # TRANSOPTR's final tuned point. run_sample_on_axis already handles
    # this via its single-envelope fallback branch, so we can call it
    # directly. transmissions will be a length-1 array (or a scalar,
    # depending on what compute_transmissions_vectorized returns for a
    # single envelope — run_sample_on_axis normalizes that).
    delta_names = ds.delta_names
    mis_dict = build_mis_dict(delta_hat, delta_names)

    t0 = time.time()
    result = gd_module.run_sample_on_axis(axis_state, mis_dict, global_config)
    dt = time.time() - t0

    if "error" in result:
        return {
            "run_idx": run_idx,
            "was_val": is_val,
            "run_label": {
                "run_id": run_label.run_id,
                "folder": run_label.folder,
                "axis": run_label.axis,
                "local_id": run_label.local_id,
            },
            "error": result["error"],
            "wall_seconds": dt,
        }

    # run_sample_on_axis already extracts the last-iteration transmission
    # as `fc_transmission`. That matches what we want — the "optimum"
    # TRANSOPTR reached with delta_hat injected.
    t_hat = float(result["fc_transmission"])

    # Tuned parameters: last row of result["inputs"]. The pipeline reads
    # these out of parameters.log, which in the verification config will
    # be a single row (one SA step). If the user later needs
    # output["fort.console"]["parameters"] specifically, that would be a
    # second hook on run_sample_on_axis; for now, parameters.log is the
    # authoritative source the existing pipeline uses.
    inputs_arr = np.asarray(result["inputs"])
    if inputs_arr.ndim == 1:
        params_hat = inputs_arr
    else:
        params_hat = inputs_arr[-1]

    # ----- [3] metrics -----
    delta_err = delta_hat - delta_true                     # (D_delta,)
    # Normalize per-dim error by the delta-std so "big" means big relative
    # to the prior spread. This mirrors how the reconstruction loss was
    # computed in standardized space at training time.
    delta_err_std = delta_err / ckpt_stats.delta_std
    delta_mse = float(np.mean(delta_err ** 2))
    delta_mse_std = float(np.mean(delta_err_std ** 2))
    delta_l2 = float(np.linalg.norm(delta_err))

    return {
        "run_idx": run_idx,
        "was_val": is_val,
        "run_label": {
            "run_id": run_label.run_id,
            "folder": run_label.folder,
            "axis": run_label.axis,
            "local_id": run_label.local_id,
        },
        # Transmissions — the proposal's preferred metric.
        "t_true": t_true,
        "t_hat": t_hat,
        "t_err": t_hat - t_true,
        # Delta metrics — cheap proxy (weaker signal per proposal).
        "delta_mse_physical": delta_mse,
        "delta_mse_standardized": delta_mse_std,
        "delta_l2_physical": delta_l2,
        # Latent diagnostics — helpful for debugging posterior collapse
        # at inference time.
        "mu": mu.tolist(),
        "log_var": log_var.tolist(),
        "params_hat": params_hat.tolist(),
        "wall_seconds": dt,
    }


# ═════════════════════════════════════════════════════════════════════
# Sweep entrypoint
# ═════════════════════════════════════════════════════════════════════

def _aggregate(subset: list, label: str) -> dict:
    """
    Aggregate a list of per-run result dicts (only `ok` entries, no errors)
    into a summary dict. Returns an empty-but-labeled dict when subset is
    empty, so downstream JSON is uniform across runs.
    """
    if not subset:
        return {"label": label, "n_ok": 0}
    t_err = np.array([r["t_hat"] - r["t_true"] for r in subset])
    return {
        "label": label,
        "n_ok": len(subset),
        "t_err_mean": float(t_err.mean()),
        "t_err_std": float(t_err.std()),
        "t_err_abs_mean": float(np.abs(t_err).mean()),
        "t_err_abs_median": float(np.median(np.abs(t_err))),
        "delta_mse_std_mean": float(
            np.mean([r["delta_mse_standardized"] for r in subset])
        ),
        "delta_mse_std_median": float(
            np.median([r["delta_mse_standardized"] for r in subset])
        ),
    }


def run_sweep(
    checkpoints: list[Path],
    data_dirs: list[str | Path],
    verify_dir: Path,
    pipeline_dir: Path,   # default "../../simulation_data/pipeline"
    config_path: Path,
    axis_name: str,
    out_dir: Path,
    max_runs: Optional[int] = None,
    seed: int = 0,
    seed_override: Optional[int] = None,
    val_frac_override: Optional[float] = None,
    delta_source: str = "decoded",
    val_only: bool = False,
) -> dict:
    """
    Full verification sweep across checkpoints.

    One TRANSOPTR initialization is shared across all checkpoints: the
    verification directory is the same regardless of d_z, so
    compile + baseline-run only needs to happen once.

    For each checkpoint we recover its training-time val split from the
    recorded (seed, val_frac) in its config, then verify across all runs
    and aggregate separately into val-only / train-only / all. Val is
    the headline number.

    `seed` here controls only the run-subsetting RNG used when
    --max_runs caps the sweep; it is *not* the VAE training seed.
    `seed_override` / `val_frac_override` are for old checkpoints whose
    config dict doesn't record these fields.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- pipeline setup -----
    gd = _import_pipeline(pipeline_dir)

    with open(config_path) as f:
        global_config = yaml.safe_load(f)

    axes = global_config.get("axes", {})
    if axis_name not in axes:
        raise KeyError(
            f"Axis '{axis_name}' not in {config_path}. "
            f"Available: {list(axes.keys())}"
        )
    axis_spec = axes[axis_name]

    # init_axis compiles optr, runs a baseline, builds aperture array, etc.
    # This is the expensive one-time setup — roughly a few seconds on the
    # TRIUMF boxes. Shared across all checkpoints below.
    # root_dir is the directory paths in the YAML are resolved relative to;
    # mirrors generate_data.main()'s `root_dir = Path.cwd()`. We pin it to
    # verify_dir so the YAML can use paths relative to the verification
    # directory.
    root_dir = verify_dir.resolve()
    print(f"[init_axis] axis={axis_name} root_dir={root_dir}")
    axis_state = gd.init_axis(axis_name, axis_spec, global_config, root_dir)
    if axis_state is None:
        raise RuntimeError("init_axis returned None — check the pipeline logs")

    # ----- dataset setup -----
    ds = MultiFolderRunDataset(data_dirs, standardize=True)
    print(ds.summary())

    # Pick which runs to verify. For now: all runs (optionally capped).
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(ds))
    if max_runs is not None and max_runs < len(all_indices):
        all_indices = rng.choice(all_indices, size=max_runs, replace=False)
        all_indices.sort()
    print(f"[sweep] verifying {len(all_indices)} runs across "
          f"{len(checkpoints)} checkpoints")

    # ----- per-checkpoint loop -----
    summary = {
        "checkpoints": [],
        "data_dirs": [str(d) for d in data_dirs],
        "verify_dir": str(verify_dir),
        "config_path": str(config_path),
        "axis": axis_name,
        "n_runs": int(len(all_indices)),
        "seed": seed,
    }

    for ckpt_path in checkpoints:
        ckpt_path = Path(ckpt_path)
        print(f"\n[ckpt] {ckpt_path}")
        vae, ckpt_stats, ckpt_cfg = load_vae_checkpoint(ckpt_path)
        d_z = ckpt_cfg["d_latent"]

        # Recover this checkpoint's val split so we can report val-only
        # metrics. Overrides let old checkpoints (pre-seed-logging) work.
        val_idx_set = recover_splits_for_checkpoint(
            ckpt_cfg, len(ds),
            seed_override=seed_override,
            val_frac_override=val_frac_override,
        )
        print(f"  [split] recovered {len(val_idx_set)} val runs "
              f"(out of {len(ds)}, seed={ckpt_cfg.get('seed', seed_override)}, "
              f"val_frac={ckpt_cfg.get('val_frac', val_frac_override or 0.2)})")

        if val_only:
            run_indices = sorted(val_idx_set)
            print(f"  [val_only] restricting to {len(run_indices)} val runs")
        else:
            run_indices = all_indices

        results = []
        t_start = time.time()
        for k, idx in enumerate(run_indices):
            try:
                r = verify_one_run(
                    ds, int(idx), vae, ckpt_stats,
                    axis_state, global_config, gd,
                    is_val=(int(idx) in val_idx_set),
                    delta_source=delta_source,
                )
            except Exception as e:
                # Keep going — one bad run shouldn't tank the whole sweep.
                r = {
                    "run_idx": int(idx),
                    "was_val": (int(idx) in val_idx_set),
                    "error": f"verify_one_run: {e}",
                }
            results.append(r)
            if "error" in r:
                print(f"  [{k+1:4d}/{len(all_indices)}] run={int(idx)} "
                      f"{'VAL' if r['was_val'] else 'TRN'} FAIL {r['error']}")
            else:
                print(
                    f"  [{k+1:4d}/{len(all_indices)}] run={int(idx)} "
                    f"{'VAL' if r['was_val'] else 'TRN'} "
                    f"T_true={r['t_true']:.4f} T_hat={r['t_hat']:.4f} "
                    f"dT={r['t_err']:+.4f} "
                    f"dMSE_std={r['delta_mse_standardized']:.3f} "
                    f"({r['wall_seconds']:.1f}s)"
                )

        dt_total = time.time() - t_start

        # ----- aggregate: val / train / all -----
        ok = [r for r in results if "error" not in r]
        ok_val   = [r for r in ok if r.get("was_val")]
        ok_train = [r for r in ok if not r.get("was_val")]
        agg = {
            "val":   _aggregate(ok_val,   "val"),
            "train": _aggregate(ok_train, "train"),
            "all":   _aggregate(ok,       "all"),
            "n_failed": len(results) - len(ok),
            "wall_seconds": dt_total,
        }

        # Print the headline numbers.
        if ok_val:
            print(f"  [agg ] VAL   n={len(ok_val):3d}  "
                  f"|T_err|_mean = {agg['val']['t_err_abs_mean']:.4f}  "
                  f"|T_err|_median = {agg['val']['t_err_abs_median']:.4f}  "
                  f"T_err_std = {agg['val']['t_err_std']:.4f}")
        if ok_train:
            print(f"  [agg ] TRAIN n={len(ok_train):3d}  "
                  f"|T_err|_mean = {agg['train']['t_err_abs_mean']:.4f}  "
                  f"|T_err|_median = {agg['train']['t_err_abs_median']:.4f}")
        if agg["n_failed"]:
            print(f"  [agg ] failed: {agg['n_failed']}")

        out_path = out_dir / f"verify_dz{d_z}__{delta_source}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "checkpoint": str(ckpt_path),
                    "d_latent": d_z,
                    "config": ckpt_cfg,
                    "delta_source": delta_source,
                    "aggregate": agg,
                    "per_run": results,
                },
                f, indent=2,
            )
        print(f"  [save] {out_path}")

        summary["checkpoints"].append({
            "path": str(ckpt_path),
            "d_latent": d_z,
            "aggregate": agg,
            "per_run_path": str(out_path),
        })
        summary["delta_source"] = delta_source

    with open(out_dir / f"summary_{delta_source}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] summary at {out_dir / f'summary_{delta_source}.json'}")
    return summary


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Latent-dimensionality verification via TRANSOPTR re-simulation.",
    )
    p.add_argument(
        "--checkpoint", action="append", required=True,
        help="Path to a trained VAE checkpoint. Repeat for multiple d_z values.",
    )
    p.add_argument(
        "--data_dir", action="append", required=True,
        help="Folder containing x_centroid_data.npz and/or y_centroid_data.npz. "
             "Repeatable; pooled into one MultiFolderRunDataset.",
    )
    p.add_argument(
        "--verify_dir", default="../../simulation_data/transoptr_verification",
        help="TRANSOPTR verification directory (contains data.dat, sy.f, archive/, "
             "and is configured to emit exactly one fort.envelope per run).",
    )
    p.add_argument(
        "--pipeline_dir", default="../../simulation_data/pipeline",
        help="Path to the simulation-generation pipeline folder "
             "(the one with generate_data.py, envelope_parser.py, etc.).",
    )
    p.add_argument(
        "--config", default="config.yaml",  # inside verify_dir by default
        help="YAML config for the verification run. Same schema as the "
             "pipeline's config.yaml — must contain `axes:` with the "
             "--axis key, `input_variables:`, `optr_src_dir:`, etc.",
    )
    p.add_argument(
        "--axis", default="verify",
        help="Which axis entry inside the --config file to use. Default: 'verify'.",
    )
    p.add_argument(
        "--out_dir", default="runs/verification",
        help="Where to write per-checkpoint JSON and summary.json.",
    )
    p.add_argument(
        "--max_runs", type=int, default=None,
        help="Cap the number of runs verified per checkpoint. Default: all.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed used only for the --max_runs subsampling. NOT the "
             "VAE training seed (that's read from each checkpoint's config).",
    )
    p.add_argument(
        "--seed_override", type=int, default=None,
        help="Override the training seed used for split recovery. Use this "
             "only when the checkpoint's config dict lacks a 'seed' field "
             "(i.e., pre-split-recovery checkpoints). Note: this applies to "
             "ALL checkpoints in the sweep, so don't mix checkpoints with "
             "different training seeds when using it.",
    )
    p.add_argument(
        "--val_frac_override", type=float, default=None,
        help="Override val_frac for split recovery. Default: 0.2 (or whatever "
             "is recorded in the checkpoint config).",
    )

    p.add_argument(
    "--delta_source", default="decoded",
    choices=["decoded", "true", "zero_std", "zero_phys", "mean"],
    help="What δ to inject into TRANSOPTR for each run:\n"
         "  decoded  = VAE: decode(μ_φ(run))  [default, the real experiment]\n"
         "  true     = ground-truth δ_true for that run  [pipeline sanity check]\n"
         "  zero_std = δ = 0 in standardized space = δ_mean in physical space\n"
         "             [= what decode(0) would give, the 'average machine']\n"
         "  zero_phys = δ = 0 in physical space [perfectly-aligned machine]\n"
         "  mean     = δ_mean (same as zero_std; different name for clarity)",
)
    p.add_argument("--val_only", action="store_true",
               help="Restrict verification to each checkpoint's val runs. "
                    "Cannot be combined with --max_runs.")

    args = p.parse_args()
    run_sweep(
        checkpoints=[Path(c) for c in args.checkpoint],
        data_dirs=[Path(d) for d in args.data_dir],
        verify_dir=Path(args.verify_dir),
        pipeline_dir=Path(args.pipeline_dir),
        config_path=Path(args.config),
        axis_name=args.axis,
        out_dir=Path(args.out_dir),
        max_runs=args.max_runs,
        seed=args.seed,
        seed_override=args.seed_override,
        val_frac_override=args.val_frac_override,
        delta_source=args.delta_source,
        val_only=args.val_only,
    )


if __name__ == "__main__":
    main()