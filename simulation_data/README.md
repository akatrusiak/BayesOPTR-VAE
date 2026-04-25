# VAE Training Data Pipeline

## Directory Layout

```
simulation_data/                    ← run everything from here
├── config.yaml                     ← global: seed, n_samples, input_variables, misalignment bounds
├── x_centroid/
│   ├── config.yaml                 ← axis-specific: beamline geometry, FC, apertures
│   ├── data.dat
│   ├── sy.f
│   └── optr
├── y_centroid/
│   ├── config.yaml
│   ├── data.dat
│   ├── sy.f
│   └── optr
├── transoptr_verification/         ← used for functional validation of VAE
│   ├── config.yaml
│   ├── data.dat
│   ├── sy.f
│   └── optr
├── output/
│   ├── x_centroid_data.npz
│   ├── y_centroid_data.npz
│   └── misalignments.yaml          ← shared across axes
└── pipeline/
    ├── generate_data.py
    ├── process_envelopes.py
    ├── envelope_parser.py
    ├── transmission.py
    ├── misalignments.py
    └── compile_optr.sh
```

## Quick Start

```bash
cd simulation_data
# 1. Edit config.yaml (set optr_src_dir, misalignment bounds, input_variables)
# 2. Edit x_centroid/config.yaml and y_centroid/config.yaml (FC locations, apertures)
# 3. Run:
python pipeline/generate_data.py
```

## How It Works

### Per misalignment sample:

1. **Sample δ** — draw misalignment values from config bounds (source offsets, quad shifts, dipole shifts + tilts)
2. **Inject into DataDat** — `pyoptr.DataDat.update_element(name=..., value=...)` modifies the in-memory data.dat
3. **Run TRANSOPTR** — `pyoptr.run(optr_dir, input_class=optr_data)` executes `./optr`, which writes `fort.envelope.*` files (one per internal optimization step)
4. **Parse envelopes** — parallel file I/O extracts s, x-envelope, y-envelope, FS-x, FS-y
5. **Compute transmission** — vectorized `scipy.special.erf` over all envelopes at once
6. **Store** — {misalignment_vector, transmissions} per sample
7. **Clean** — remove `fort.envelope.*`, keep `optr` executable, reset misalignments to zero

### Key interfaces:

| Component | Interface | Source |
|---|---|---|
| data.dat read/write | `pyoptr.DataDat` | pyoptr library |
| Simulation execution | `pyoptr.run()` | pyoptr library |
| Envelope parsing | `envelope_parser.parse_single_envelope()` | this pipeline |
| Transmission calc | `transmission.compute_transmissions_vectorized()` | this pipeline |
| Misalignment sampling | `misalignments.sample_misalignments()` | this pipeline |

## Key Design: Tied Misalignments

The outer loop iterates over **misalignment samples**. For each sample δ, both x_centroid and y_centroid are run with the **same** misalignment vector. This is essential because:

- The VAE encoder must learn that x and y observations from the same run share one machine state
- The decoder reconstructs a single δ that explains both planes
- Cross-plane correlations (quads focus x / defocus y) require paired training data

## Data Flow

```
For each sample i = 1..M:
    δ = sample_misalignments(global_config, rng)      ← ONCE
    
    For each axis ∈ {x_centroid, y_centroid}:
        DataDat.update_element(name, value) for all δ  ← inject
        pyoptr.run(optr_dir, input_class=optr_data)    ← TRANSOPTR writes:
                                                           fort.envelope.00001, .00002, ...
                                                           parameters.log
        parse fort.envelope.* → (s, x_env, y_env, FS-x, FS-y)
        parse parameters.log → per-step input variables
        compute transmission (vectorized erf)
        reset δ to zero, clean fort files
```

## parameters.log Format

Written by the modified TRANSOPTR alongside each `fort.envelope.*` file:

```
col 0:    step index (or s)
col 1:    n/a
col 2+i:  value of DataDat.data["elements"][i] at that step
```

Row ordering matches the envelope file ordering. The pipeline extracts only the columns corresponding to `input_variables` from `config.yaml`, identified by matching PV names to element indices via `DataDat.get_element(name=...)`.

## Config Reference

### Global (`config.yaml`)

| Key | Description |
|---|---|
| `seed` | Random seed |
| `n_misalignment_samples` | Number of δ draws |
| `optr_src_dir` | TRANSOPTR source path |
| `axes` | Maps axis names → {dir, config} |
| `input_variables` | PV names for the VAE's x (matched to data.dat by name) |
| `src_mis` | Source misalignment bounds |
| `quad_misalignments` | Per-quad bounds, keyed by physical name (no :CUR) |
| `dipole_misalignments` | Per-dipole positional + angular bounds |

### Per-axis (`x_centroid/config.yaml`)

| Key | Description |
|---|---|
| `beamline_wall_width` | Pipe half-width (cm) |
| `offset` | Beamline offset for s-index lookup |
| `measurement_device` | FC name for transmission readout |
| `fc` | FC locations along beamline |
| `quad_apertures` | Quad locations + aperture half-widths |

## Output Format (`x_centroid_data.npz`)

| Key | Shape | Description |
|---|---|---|
| `misalignment_vectors` | `(M, D_δ)` | Ground-truth δ (shared with y_centroid) |
| `misalignment_names` | `(D_δ,)` | Element name per dimension |
| `transmissions` | ragged `(N_i,)` | Per-step transmission |
| `inputs` | ragged `(N_i, d)` | Per-step input variables from parameters.log |
| `input_variable_names` | `(d,)` | PV names for each input column |

## Element Naming

Misalignment element names must match data.dat **exactly**:

| Config key | Produces | data.dat name |
|---|---|---|
| `src_mis.x1` | `MISALIGNX1` | `MISALIGNX1` |
| `quad_misalignments.HEBT2:Q3` | `HEBT2:Q3:MISALIGNX` | `HEBT2:Q3:MISALIGNX` |
| `dipole_misalignments.HEBT2:MB1` | `HEBT2:MB1:MBX` | `HEBT2:MB1:MBX` |

Note: quad current PVs use `:CUR` (e.g. `HEBT2:Q3:CUR`), but misalignment elements don't. The config uses the physical element base name for misalignments.
