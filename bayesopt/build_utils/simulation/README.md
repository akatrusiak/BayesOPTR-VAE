# Simulation Build Pipeline

Builds TRANSOPTR simulation files and Bayesian optimization configs from a tune config XML.

## Quick Start

```bash
python run_build.py --name mebt-hebt --tune_config /path/to/tune_config.xml
```

This reads the tune XML and runs three steps:
1. Trims the beamline path XML from `$ACCDIR` to the start/end elements
2. Builds TRANSOPTR files (`sy.f`, `data.dat`, compiled executable) with misalignment-wrapped element handlers
3. Generates a YAML config for Bayesian optimization

## File Overview

| File | Purpose |
|---|---|
| `run_build.py` | Orchestrator — parses tune XML, runs all three steps |
| `build_xml.py` | Step 1 — loads path from `$ACCDIR`, trims to start/end, returns lxml tree |
| `build_transoptr.py` | Step 2 — calls `xml2optr.translate()` with misalignment wrappers, compiles executable |
| `build_sim_config.py` | Step 3 — parses trimmed path + `data.dat` to produce BO YAML config |

## Tune Config XML

The tune config is the single input file. It contains everything the pipeline needs:

- **Path name**: `<root path="ios-mws-hebt1-prague">` — which accelerator path in `$ACCDIR`
- **Start/end elements**: `<optr start="MEBT:RPM5" end="HEBT:FC5"/>` — section boundaries
- **Beam parameters**: `<optr>` tag (beam dimensions, correlations) and `<tune>` tag (mass, charge, energy)
- **Quad setpoints**: `<set pv="..." value="..."/>` tags

The end element is assumed to be the measurement device (Faraday cup).

## CLI Options

```
python run_build.py --name NAME --tune_config PATH [options]

Required:
  --name              Name for this config (used for directories and filenames)
  --tune_config       Path to tune_config.xml

Optional:
  --device_types      Element types to tune (default: steerer)
  --measurement_device  Override measurement device (default: end element)
  --write_xml         Write the trimmed path XML to disk for inspection
  --skip_transoptr    Skip TRANSOPTR build (reuse existing files)
  --skip_config       Skip YAML config generation
```

## Output Locations

Relative to this directory:

- `../../simulation/transoptr/<name>/` — TRANSOPTR files (`sy.f`, `data.dat`, executable)
- `../../config/sim/<name>.yaml` — BO YAML config

## Dependencies

**System**: `gfortran`, `$ACCDIR` and `$OPTRDIR` environment variables

**Python**: `lxml`, `pyyaml`, `tabulate`, `setuptools`, `gitpython`, `accpy`, `xml2optr`, `pyoptr`

## How It Works

- **Step 1** loads the path directly from `$ACCDIR` and trims it in memory — no pre-built XML file needed.

- **Step 2** calls `xml2optr.translate()` with a `setup` hook that injects source misalignment handling. The element map wraps quads with `misalign_shift`, benders with `misalign_bend`, and steerers with `ideal_steerers`. Misalignment bounds are defined as module-level constants in `build_transoptr.py` for future randomization per-run.

- **Step 3** merges parsing logic from the previous sim and beam config scripts — PV-based keys, `seq/element` iteration, plus simulation-specific data (apertures, misalignment bounds, slit widths). The trimmed XML tree is passed in memory from step 1.
