# Verification — d_z sweep via TRANSOPTR re-simulation

## What you need

1. **Trained VAE checkpoints**, one per `d_z` you want to compare. Train
   them with `vae.train` in the usual way, just varying `--d_latent`:

   ```
   python -m vae.train --data_dir data/output_good_4 --d_latent 3  --out_dir runs/vae_dz3  --epochs 100 --beta_warmup_epochs 10 --free_bits 0.3
   python -m vae.train --data_dir data/output_good_4 --d_latent 5  --out_dir runs/vae_dz5  --epochs 100 --beta_warmup_epochs 10 --free_bits 0.3
   python -m vae.train --data_dir data/output_good_4 --d_latent 8  --out_dir runs/vae_dz8  --epochs 100 --beta_warmup_epochs 10 --free_bits 0.3
   python -m vae.train --data_dir data/output_good_4 --d_latent 12 --out_dir runs/vae_dz12 --epochs 100 --beta_warmup_epochs 10 --free_bits 0.3
   ```

2. **A verification TRANSOPTR directory.** Same layout as any axis
   directory in the simulation-generation pipeline — `sy.f`, `data.dat`,
   and an `archive/` subfolder — except the `data.dat` must be configured
   so that a single `pyoptr.run` call produces exactly **one**
   `fort.envelope` (the tuned optimum), written to `archive/fort.envelope`.
   The pipeline's `run_sample_on_axis` already has a fallback branch that
   picks up this single-file case, so no pipeline changes are needed.

3. **A verification YAML config.** Same schema as the
   simulation-generation `config.yaml`. Minimal version:

   ```yaml
   optr_src_dir: /path/to/TRANSOPTR/src     # or set $OPTRDIR
   input_variables:                          # 9 names, same order as training
     - HEBT2:Q3:CUR
     - HEBT2:Q4:CUR
     # ...
   maxit: 1000
   n_workers: 1                              # only 1 envelope here
   axes:
     verify:
       dir: .                                # path to the verification dir (rel. to --verify_dir)
       config: axis.yaml                     # path to the axis-specific config
   ```

   The `axes.verify.dir` and `axes.verify.config` paths are resolved
   relative to `--verify_dir`, matching how the pipeline's main loop
   resolves them relative to its `root_dir`.

## Running

```
python -m verification.verify_dz \
    --checkpoint runs/vae_dz3/best.pt \
    --checkpoint runs/vae_dz5/best.pt \
    --checkpoint runs/vae_dz8/best.pt \
    --checkpoint runs/vae_dz12/best.pt \
    --data_dir data/output_good_4 \
    --data_dir data/output_good_5 \
    --verify_dir path/to/transoptr_verify \
    --pipeline_dir path/to/simulation_data \
    --config path/to/transoptr_verify/config.yaml \
    --axis verify \
    --out_dir runs/verification
```

Optional flags:

- `--max_runs N` — cap the number of runs verified per checkpoint
  (useful while iterating).
- `--seed N` — only matters when `--max_runs` subsamples.

## Outputs

- `runs/verification/verify_dz3.json`, `…dz5.json`, etc. — per-run
  results for each checkpoint, including `t_true`, `t_hat`, δ-space
  errors, and the latent `mu`/`log_var`.
- `runs/verification/summary.json` — aggregate metrics per checkpoint,
  handy for picking the smallest `d_z` whose `t_err` is within noise.

## What we actually test

For each dataset run `j` with true misalignment `δ_true`:

1. `T_true` is the transmission at the *last* SA trajectory step in the
   training data — TRANSOPTR's own best-effort optimum for `δ_true`.
   (No re-simulation needed for the truth side; we already have it.)

2. Encode the **full** run (all `(x, y)` pairs) through the trained
   encoder to get `μ`. Decode `μ` to get `δ̂` in physical units.

3. Inject `δ̂` into the verification TRANSOPTR directory, call `pyoptr.run`,
   and read the single resulting `fort.envelope`. `T̂` is the transmission
   at that tuned optimum.

4. Record `T̂ − T_true` (primary metric) and `‖δ̂ − δ_true‖` (secondary —
   the proposal notes that Euclidean distance in δ-space is a weak signal
   because landscape-equivalent δ's are mapped to the same `z` by design).

## Picking `d_z`

Per the proposal: smallest `d_z` whose landscape fidelity is comparable
to the larger ones wins. The headline number is
`summary.checkpoints[k].aggregate.t_err_abs_mean` (or `t_err_abs_median`
if outliers dominate). Plot these against `d_z` and pick the elbow.
