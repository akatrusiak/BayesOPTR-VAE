"""
misalignments.py — Sample random misalignments from the global config.

Misalignment bounds live in the TOP-LEVEL config.yaml (shared across axes)
because the same physical δ applies to both x_centroid and y_centroid.

Naming convention: element names match data.dat EXACTLY.
  - Source: MISALIGNX1, MISALIGNY1, MISALIGNX2, MISALIGNY2
  - Quads:  HEBT2:Q3:MISALIGNX, HEBT2:Q3:MISALIGNY  (no :CUR)
  - Dipole: HEBT2:MB1:MISALIGNX/Y/Z, HEBT2:MB1:MBX/MBY
"""

import numpy as np
from collections import OrderedDict


def sample_misalignments(
    global_config: dict,
    rng: np.random.Generator,
) -> tuple[OrderedDict, np.ndarray]:
    """
    Sample a random misalignment vector from the bounds in the global config.

    Args:
        global_config: the top-level config dict (contains src_mis,
                       quad_misalignments, dipole_misalignments)
        rng: numpy random generator

    Returns:
        mis_dict:   OrderedDict of {data.dat element name → value}
        mis_vector: 1-D array of all values in deterministic order
    """
    mis_dict = OrderedDict()

    # ── Source misalignments ──────────────────────────────────────
    for name, bounds in global_config.get("src_mis", {}).items():
        tag = "MISALIGN" + name.upper()  # x1 → MISALIGNX1
        lo, hi = bounds["min"], bounds["max"]
        sign = rng.choice([-1, 1])
        mis_dict[tag] = float(sign * rng.uniform(lo, hi))

    # ── Quad misalignments ────────────────────────────────────────
    # Config keys are the physical element base name (e.g. HEBT2:Q3)
    # We append :MISALIGNX and :MISALIGNY to match data.dat
    for base_name, bounds in global_config.get("quad_misalignments", {}).items():
        lo, hi = bounds["min"], bounds["max"]
        for suffix in [":MISALIGNX", ":MISALIGNY"]:
            sign = rng.choice([-1, 1])
            mis_dict[base_name + suffix] = float(sign * rng.uniform(lo, hi))

    # ── Dipole misalignments ──────────────────────────────────────
    for base_name, bounds in global_config.get("dipole_misalignments", {}).items():
        lo_pos = bounds.get("pos_mis_min", 0.0)
        hi_pos = bounds.get("pos_mis_max", 0.0)
        for suffix in [":MISALIGNX", ":MISALIGNY", ":MISALIGNZ"]:
            sign = rng.choice([-1, 1])
            mis_dict[base_name + suffix] = float(sign * rng.uniform(lo_pos, hi_pos))

        lo_ang = bounds.get("ang_mis_min", 0.0)
        hi_ang = bounds.get("ang_mis_max", 0.0)
        for suffix in [":MBX", ":MBY"]:
            sign = rng.choice([-1, 1])
            mis_dict[base_name + suffix] = float(sign * rng.uniform(lo_ang, hi_ang))

    mis_vector = np.array(list(mis_dict.values()))
    return mis_dict, mis_vector


def get_misalignment_names(global_config: dict) -> list[str]:
    """Return ordered list of misalignment parameter names (for VAE decoder labels)."""
    rng = np.random.default_rng(0)
    mis_dict, _ = sample_misalignments(global_config, rng)
    return list(mis_dict.keys())


def get_misalignment_dim(global_config: dict) -> int:
    """Return D_delta."""
    return len(get_misalignment_names(global_config))
