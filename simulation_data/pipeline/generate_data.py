"""
generate_data.py — VAE training data generation pipeline.

Runs from: simulation_data/

Key design: the OUTER loop is over misalignment samples, the INNER loop
is over axes. This ensures the same δ is applied to both x_centroid and
y_centroid, which is required for the VAE to learn a shared latent space.

Per-step tuning inputs are read from parameters.log, written by the
modified TRANSOPTR alongside the fort.envelope files.

Usage:
    python generate_data.py
    python generate_data.py --config my_config.yaml
    python generate_data.py --axis x_centroid
"""

import argparse
import os
import sys
import time
import subprocess
import numpy as np
import yaml
from pathlib import Path
from collections import OrderedDict
from contextlib import redirect_stdout

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import pyoptr
from pyoptr import DataDat
from pyoptr.postprocess import get_column

from envelope_parser import batch_parse_envelopes, parse_single_envelope
from transmission import (
    build_aperture_array_from_config,
    compute_transmissions_vectorized,
    extract_fc_transmission,
)
from misalignments import sample_misalignments


# ═════════════════════════════════════════════════════════════════
# Compilation
# ═════════════════════════════════════════════════════════════════

def compile_optr(optr_src_dir: str, work_dir: Path):
    """Build the optr executable in work_dir (run once per axis)."""
    compile_script = Path(__file__).parent / "compile_optr.sh"
    if compile_script.exists():
        result = subprocess.run(
            ["bash", str(compile_script), optr_src_dir, str(work_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            sys.exit(1)
        print(result.stdout.strip())
    else:
        fflags = "-std=legacy -finit-local-zero -O3"
        src_dir = Path(optr_src_dir) / "src"
        subprocess.run(f"cd {src_dir} && rm -f *.o && gfortran {fflags} -c *.f",
                       shell=True, check=True)
        subprocess.run(f"cd {work_dir} && gfortran {fflags} -c sy.f",
                       shell=True, check=True)
        subprocess.run(
            f"cd {work_dir} && gfortran {fflags} -o optr sy.o {src_dir}/*.o",
            shell=True, check=True,
        )
        (work_dir / "sy.o").unlink(missing_ok=True)
        if not (work_dir / "optr").exists():
            sys.exit("Error: optr executable not created")


def clean_fort_files(work_dir: Path):
    """Remove TRANSOPTR output files, preserving optr, data.dat, sy.f."""
    for pattern in ["fort.*", "default.gnu", "envelope.gnu"]:
        for f in work_dir.glob(pattern):
            f.unlink(missing_ok=True)


# ═════════════════════════════════════════════════════════════════
# Diagnostic plotting
# ═════════════════════════════════════════════════════════════════

def plot_progress(axis_name: str, results: list, archive_dir: Path, output_dir: Path):
    """
    Create a diagnostic plot showing FC transmission over transoptr iterations for the most recent run.
    
    Args:
        axis_name: name of the axis
        results: list of result dicts from run_sample_on_axis (some may be None if failed)
        archive_dir: directory containing envelope and parameters.log
        output_dir: where to save the plot
    """
    # Only plot for the most recent (last) result
    if not results or results[-1] is None:
        return
    
    result = results[-1]
    transmissions = result["transmissions"]
    if not isinstance(transmissions, np.ndarray) or len(transmissions) == 0:
        return
    
    iterations = np.arange(1, len(transmissions) + 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Plot FC transmission over transoptr iterations
    ax.plot(iterations, transmissions, 'b-o', label='FC Transmission', linewidth=2, markersize=6)
    ax.set_xlabel('Transoptr Iteration', fontsize=12)
    ax.set_ylabel('FC Transmission', color='b', fontsize=12)
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3)
    
    # Read chi values from parameters.log for the current run
    log_path = archive_dir / "envelope" / "parameters.log"
    if log_path.exists():
        try:
            data = np.loadtxt(log_path, skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            # Column 1 is sqrt(chi), trim to match number of transmissions
            chi_values = data[:, 1][:len(transmissions)]
            if len(chi_values) == len(iterations):
                ax2 = ax.twinx()
                ax2.plot(iterations, chi_values, 'r-^', label='√(χ)', linewidth=2, markersize=6)
                ax2.set_ylabel('√(χ)', color='r', fontsize=12)
                ax2.tick_params(axis='y', labelcolor='r')
        except Exception as e:
            pass  # Skip chi plotting if there's an issue
    
    # Title
    plt.title(f'{axis_name} — Transoptr Progress (Misalignment {len(results)})', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    # Save
    plot_path = output_dir / f"{axis_name}_progress.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved {plot_path}")


# ═════════════════════════════════════════════════════════════════
# parameters.log parsing
# ═════════════════════════════════════════════════════════════════

def build_input_index_map(optr_data: DataDat, input_variable_names: list[str]) -> list[int]:
    """
    Build a mapping from input_variable PV names to their element index
    in the DataDat. This index corresponds to the column in parameters.log
    (offset by 2, since col 0 = step, col 1 = n/a, col 2+ = elements).

    Returns:
        list of element indices (into DataDat.data["elements"])
    """
    indices = []
    elements = optr_data.data["elements"]
    name_to_idx = {el.get("name", ""): i for i, el in enumerate(elements)}

    for pv_name in input_variable_names:
        if pv_name in name_to_idx:
            indices.append(name_to_idx[pv_name])
        else:
            print(f"  [WARNING] Input variable '{pv_name}' not found in data.dat elements")
            indices.append(-1)
    return indices


def parse_parameters_log(
    log_path: Path,
    input_element_indices: list[int],
) -> np.ndarray:
    """
    Parse parameters.log to extract the input variables for each step.

    parameters.log format:
        col 0: step index (or s)
        col 1: n/a
        col 2 ... 2+N-1: all N data.dat element values, in data.dat order

    Args:
        log_path: path to parameters.log
        input_element_indices: which element indices to extract (from build_input_index_map)

    Returns:
        inputs: (n_steps, n_input_vars) array
    """
    if not log_path.exists():
        return None

    data = np.loadtxt(log_path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_steps = data.shape[0]
    n_input_vars = len(input_element_indices)
    inputs = np.empty((n_steps, n_input_vars))

    for j, elem_idx in enumerate(input_element_indices):
        if elem_idx < 0:
            inputs[:, j] = 0.0
        else:
            col = elem_idx + 2  # offset: col0=step, col1=n/a, col2+=elements
            if col < data.shape[1]:
                inputs[:, j] = data[:, col]
            else:
                inputs[:, j] = 0.0

    return inputs


# ═════════════════════════════════════════════════════════════════
# Per-axis initialization (run once)
# ═════════════════════════════════════════════════════════════════

def init_axis(axis_name: str, axis_spec: dict, global_config: dict, root_dir: Path) -> dict:
    """
    Initialize one axis: compile, load DataDat, compute aperture array.
    Returns a dict with everything needed for run_sample_on_axis().
    """
    axis_dir = root_dir / axis_spec["dir"]
    if not axis_dir.exists():
        print(f"  [ERROR] Directory not found: {axis_dir}")
        return None

    axis_cfg_path = root_dir / axis_spec["config"]
    with open(axis_cfg_path) as f:
        axis_cfg = yaml.safe_load(f)

    # Compile
    optr_src_dir = global_config.get("optr_src_dir") or os.environ.get("OPTRDIR")
    if not optr_src_dir:
        raise KeyError(
            "optr_src_dir not found. Set it in config.yaml or export $OPTRDIR."
        )
    optr_src_dir = Path(optr_src_dir).expanduser()

    if not (axis_dir / "optr").exists():
        compile_optr(optr_src_dir, axis_dir)
    else:
        print(f"  {axis_name}: optr exists")

    # DataDat
    optr_data = DataDat(file=str(axis_dir / "data.dat"))

    # Input variable index map
    input_vars = global_config.get("input_variables", [])
    input_indices = build_input_index_map(optr_data, input_vars)

    # Baseline run for s-grid.
    # We only need the s-grid here; the baseline is not itself training data,
    # so we squash opt-maxit to 1 to avoid wasting time on TRANSOPTR's
    # fitter before we even have a misalignment injected. The full maxit
    # (from the top-level config) is restored before returning, so sampling
    # uses the intended number of optimization iterations.
    sampling_maxit = int(global_config.get("maxit", 1000))
    try:
        optr_data["opt-maxit"] = 1
    except Exception as e:
        print(f"  [WARNING] {axis_name}: could not set opt-maxit=1 for baseline ({e})")

    print(f"  {axis_name}: baseline run for s-grid (opt-maxit=1)...")
    archive_dir = axis_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    clean_fort_files(archive_dir)
    with redirect_stdout(open(os.devnull, 'w')):
        baseline = pyoptr.run(str(axis_dir), input_class=optr_data, archive_dir=str(archive_dir))
    s_ref, _, _, _, _ = get_column(baseline, 's', 'x-envelope', 'y-envelope', 'x-centroid', 'y-centroid')
    s_reference = np.array(s_ref)

    # Restore opt-maxit for the sampling runs
    try:
        optr_data["opt-maxit"] = sampling_maxit
        print(f"  {axis_name}: opt-maxit restored to {sampling_maxit} for sampling")
    except Exception as e:
        print(f"  [WARNING] {axis_name}: could not restore opt-maxit to {sampling_maxit} ({e})")

    # Aperture array
    wall_width = axis_cfg.get("beamline_wall_width", 2.54)
    offset = axis_cfg.get("offset", 1.0)
    fc_name = axis_cfg.get("measurement_device", "DRA:FC1")
    fc_loc = axis_cfg["fc"][fc_name]["loc"]
    fc_index = extract_fc_transmission(s_reference, fc_loc, offset)

    quad_apertures = [
        {"loc": qa["loc"], "aperture": qa["aperture"]}
        for qa in axis_cfg.get("quad_apertures", {}).values()
    ]
    slits_array = build_aperture_array_from_config(s_reference, axis_cfg)

    print(f"  {axis_name}: s-grid {len(s_reference)} pts, "
          f"FC at idx {fc_index} (s={s_reference[fc_index]:.1f}), "
          f"{np.sum(slits_array < wall_width)} apertures")

    clean_fort_files(archive_dir)

    return {
        "axis_name": axis_name,
        "axis_dir": axis_dir,
        "archive_dir": archive_dir,
        "axis_cfg": axis_cfg,
        "optr_data": optr_data,
        "input_indices": input_indices,
        "s_reference": s_reference,
        "slits_array": slits_array,
        "fc_index": fc_index,
    }


# ═════════════════════════════════════════════════════════════════
# Run one sample on one axis
# ═════════════════════════════════════════════════════════════════

def run_sample_on_axis(
    axis_state: dict,
    mis_dict: OrderedDict,
    global_config: dict,
) -> dict:
    """
    Apply a misalignment to one axis, run TRANSOPTR, parse results.

    The mis_dict is already sampled — this function just injects, runs,
    and collects.
    """
    optr_data = axis_state["optr_data"]
    optr_dir = axis_state["axis_dir"]
    archive_dir = axis_state["archive_dir"]
    slits_array = axis_state["slits_array"]
    fc_index = axis_state["fc_index"]
    input_indices = axis_state["input_indices"]

    # Inject misalignments
    for name, value in mis_dict.items():
        try:
            optr_data.update_element(name=name, value=value)
        except ValueError:
            pass  # element not in this data.dat — expected for some names

    # Clean and run
    clean_fort_files(archive_dir)
    try:
        with redirect_stdout(open(os.devnull, 'w')):
            optr_out = pyoptr.run(str(optr_dir), input_class=optr_data, archive_dir=str(archive_dir))
    except Exception as e:
        _reset_misalignments(optr_data, mis_dict)
        return {"error": str(e)}

    # Collect envelopes
    # Multiple envelopes in archive_dir/envelope/fort.envelope.*
    envelope_files = sorted((archive_dir / "envelope").glob("fort.envelope.*"))
    if not envelope_files:
        # Single envelope in archive_dir/fort.envelope
        single = archive_dir / "fort.envelope"
        if single.exists():
            envelope_files = [single]
        else:
            print(f"  [WARNING] No envelope files found in {archive_dir}/envelope/")
            # Diagnostic: check what files exist in archive_dir
            print(f"  [DEBUG] Looking for envelopes in {archive_dir}")
            print(f"  [DEBUG] Files in archive_dir: {list(archive_dir.glob('*'))}")
            if (archive_dir / "envelope").exists():
                print(f"  [DEBUG] Files in archive_dir/envelope: {list((archive_dir / 'envelope').glob('*'))}")
            _reset_misalignments(optr_data, mis_dict)
            return {"error": "no envelopes"}



    # Parse envelopes
    try:
        s, x_env, y_env, x_cm, y_cm = batch_parse_envelopes(
            envelope_files,
            n_workers=min(global_config.get("n_workers", 4), len(envelope_files)),
        )
        
        # Rebuild slits_array to match the actual s-grid from envelope files
        # (in case envelope files have different resolution than baseline).
        # Uses the YAML config directly — mirrors Beamline._generate_slits_array.
        slits_array = build_aperture_array_from_config(s, axis_state["axis_cfg"])
        
        # Recalculate fc_index for the new s-grid
        fc_loc = axis_state["axis_cfg"]["fc"][axis_state["axis_cfg"].get("measurement_device", "DRA:FC1")]["loc"]
        fc_index = extract_fc_transmission(s, fc_loc, axis_state["axis_cfg"].get("offset", 1.0))
        
    except Exception as e:
        _reset_misalignments(optr_data, mis_dict)
        return {"error": f"parse: {e}"}

    # Compute transmissions
    transmissions = compute_transmissions_vectorized(
        x_env, y_env, x_cm, y_cm, slits_array, fc_index=fc_index,
    )

    # Parse per-step inputs. Preferred source is parameters.log (one row per
    # SA step, used for multi-envelope training runs). If it's absent — e.g.
    # the verification data.dat is configured for a single tuned envelope and
    # doesn't emit the log — fall back to fort.console['parameters'] from the
    # pyoptr return value. That gives us one row (the final tuned point),
    # which we broadcast to match len(transmissions). Last resort: the
    # current DataDat state.
    log_path = archive_dir / "envelope" / "parameters.log"
    inputs = parse_parameters_log(log_path, input_indices)
    if inputs is None:
        # Fallback A: fort.console.parameters from the pyoptr return value.
        # The list is in DataDat-element order (same ordering as parameters.log
        # columns), so we index it with input_indices directly. An index of -1
        # means the input variable wasn't found in data.dat — fill with 0.0.
        console = (optr_out or {}).get("fort.console", {})
        console_params = console.get("parameters", None)
        if console_params is not None:
            console_params = np.asarray(console_params, dtype=np.float64)
            current = np.array([
                console_params[idx] if (idx >= 0 and idx < len(console_params)) else 0.0
                for idx in input_indices
            ])
            inputs = np.tile(current, (len(transmissions), 1))
        else:
            # Fallback B: current DataDat state. This is the pre-existing
            # behavior and is a last resort — it reflects the pre-run values,
            # not the tuned ones, so it's only correct if TRANSOPTR did no
            # optimization (e.g. opt-maxit=1 baseline).
            current = []
            for idx in input_indices:
                if idx >= 0:
                    current.append(optr_data.data["elements"][idx]["value"])
                else:
                    current.append(0.0)
            inputs = np.tile(current, (len(transmissions), 1))


    # Reset misalignments
    _reset_misalignments(optr_data, mis_dict)

    # Clean
    # clean_fort_files(archive_dir)

    # Extract FC transmission (what gets measured at the Faraday Cup)
    # Use the transmission from the final (best) transoptr iteration
    if isinstance(transmissions, np.ndarray) and len(transmissions) > 0:
        fc_transmission = transmissions[-1]
    else:
        fc_transmission = transmissions  # scalar fallback
    
    return {
        "transmissions": transmissions,     # Full beamline profile (for diagnostics)
        "fc_transmission": fc_transmission, # Single measured value at FC
        "inputs": inputs,                   # (n_steps, n_input_vars)
        "n_envelopes": len(envelope_files),
        "fc_index": fc_index,               # Index of FC in s-grid
    }


def _reset_misalignments(optr_data: DataDat, mis_dict: OrderedDict):
    """Set all misalignment elements back to zero."""
    for name in mis_dict:
        try:
            optr_data.update_element(name=name, value=0.0)
        except ValueError:
            pass


# ═════════════════════════════════════════════════════════════════
# Main loop
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate VAE training data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--axis", default=None, help="Run one axis only")
    args = parser.parse_args()

    root_dir = Path.cwd()
    with open(root_dir / args.config) as f:
        global_config = yaml.safe_load(f)

    rng = np.random.default_rng(global_config.get("seed", 42))
    output_dir = root_dir / global_config.get("output_dir", "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = global_config.get("n_misalignment_samples", 100)
    input_var_names = global_config.get("input_variables", [])

    # ── Select axes ───────────────────────────────────────────────
    all_axes = global_config.get("axes", {})
    if args.axis:
        if args.axis not in all_axes:
            sys.exit(f"Axis '{args.axis}' not in config. Available: {list(all_axes.keys())}")
        all_axes = {args.axis: all_axes[args.axis]}

    # ── Initialize all axes ───────────────────────────────────────
    print("Initializing axes...")
    axis_states = {}
    for axis_name, axis_spec in all_axes.items():
        state = init_axis(axis_name, axis_spec, global_config, root_dir)
        if state is not None:
            axis_states[axis_name] = state

    if not axis_states:
        sys.exit("No axes initialized successfully")

    # ── Pre-allocate result storage ───────────────────────────────
    # Keyed by axis name
    all_results = {name: [] for name in axis_states}
    all_mis_vectors = []
    all_mis_dicts = []

    # ── Main loop: outer=misalignment, inner=axis ─────────────────
    print(f"\nStarting {n_samples} misalignment samples across {list(axis_states.keys())}...")
    t_total = time.time()
    try:
        for i in range(n_samples):
            t0 = time.time()

            # Sample misalignment ONCE, shared across axes
            mis_dict, mis_vector = sample_misalignments(global_config, rng)
            all_mis_vectors.append(mis_vector)
            all_mis_dicts.append(dict(mis_dict))

            status_parts = []
            for axis_name, state in axis_states.items():
                result = run_sample_on_axis(state, mis_dict, global_config)

                if "error" in result:
                    status_parts.append(f"{axis_name}: FAIL ({result['error']})")
                    all_results[axis_name].append(None)
                else:
                    t_fc = result["fc_transmission"]
                    n_env = result["n_envelopes"]
                    status_parts.append(f"{axis_name}: {n_env}env T_FC={t_fc:.4f}")
                    all_results[axis_name].append(result)

            dt = time.time() - t0
            print(f"  [{i+1:4d}/{n_samples}] {' | '.join(status_parts)}  ({dt:.1f}s)")
            
            # Update progress plots for each axis
            for axis_name, state in axis_states.items():
                try:
                    plot_progress(axis_name, all_results[axis_name], state["archive_dir"], output_dir)
                except Exception as e:
                    print(f"  [WARNING] Could not update plot for {axis_name}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user, stopping early...")
    finally:
        total_time = time.time() - t_total
        print(f"\nCompleted in {total_time:.0f}s")

        # ── Save ──────────────────────────────────────────────────────
        # Shared misalignment data
        mis_matrix = np.array(all_mis_vectors)
        mis_names = list(all_mis_dicts[0].keys()) if all_mis_dicts else []

        for axis_name, results in all_results.items():
            # Filter out failed samples (keep indices aligned with mis_matrix)
            valid_indices = [i for i, r in enumerate(results) if r is not None]
            valid_results = [results[i] for i in valid_indices]

            if not valid_results:
                print(f"  {axis_name}: no valid results, skipping save")
                continue

            # ── Flatten per-envelope data across all misalignment runs ──
            # Each envelope file produces one row:
            #   (misalignment_id, misalignment_vector, input_params, transmission)
            # Shape sanity: for each run, len(transmissions) should equal inputs.shape[0].
            # If they mismatch (e.g. parameters.log truncated), we trim to the shorter.
            flat_mis_ids     = []   # (N_total,)   int, index into valid_mis
            flat_mis_vectors = []   # (N_total, D_delta)
            flat_inputs      = []   # (N_total, D_input)
            flat_transmission = []  # (N_total,)

            for run_idx, (orig_i, r) in enumerate(zip(valid_indices, valid_results)):
                T = np.asarray(r["transmissions"]).ravel()
                X = np.asarray(r["inputs"])
                if X.ndim == 1:
                    X = X.reshape(1, -1)

                n = min(len(T), X.shape[0])
                if n == 0:
                    continue
                if len(T) != X.shape[0]:
                    print(f"  [WARNING] {axis_name} run {orig_i}: "
                        f"transmissions ({len(T)}) != inputs ({X.shape[0]}), trimming to {n}")

                mis_vec = mis_matrix[orig_i]  # (D_delta,)

                flat_mis_ids.append(np.full(n, run_idx, dtype=np.int32))
                flat_mis_vectors.append(np.broadcast_to(mis_vec, (n, mis_vec.shape[0])).copy())
                flat_inputs.append(X[:n])
                flat_transmission.append(T[:n])

            if not flat_transmission:
                print(f"  {axis_name}: no envelopes after flattening, skipping save")
                continue

            flat_mis_ids      = np.concatenate(flat_mis_ids)
            flat_mis_vectors  = np.concatenate(flat_mis_vectors, axis=0)
            flat_inputs       = np.concatenate(flat_inputs, axis=0)
            flat_transmission = np.concatenate(flat_transmission)

            # Also keep the (N_runs, D_delta) table of unique misalignments
            valid_mis = mis_matrix[valid_indices]

            npz_path = output_dir / f"{axis_name}_data.npz"
            np.savez_compressed(
                npz_path,
                # Per-envelope (flat) — primary VAE training data
                misalignment_id=flat_mis_ids,             # (N_total,) run index
                misalignment_vectors=flat_mis_vectors,    # (N_total, D_delta) — repeated per run
                inputs=flat_inputs,                       # (N_total, D_input)
                transmission=flat_transmission,           # (N_total,) FC transmission per envelope
                # Per-misalignment table (for inspection / grouping)
                unique_misalignment_vectors=valid_mis,    # (N_runs, D_delta)
                misalignment_names=np.array(mis_names),
                input_variable_names=np.array(input_var_names),
            )
            print(f"  Saved {npz_path}: {len(valid_results)} runs, "
                f"{flat_transmission.shape[0]} envelopes total")

        # Save all misalignments (for both axes, including failed)
        yaml_path = output_dir / "misalignments.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(all_mis_dicts, f, default_flow_style=False)
        print(f"  Saved {yaml_path}: {len(all_mis_dicts)} entries")

        print(f"\nDone. Output in {output_dir}/")


if __name__ == "__main__":
    main()
