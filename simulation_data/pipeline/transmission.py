"""
transmission.py — Vectorized transmission calculation.

Reimplements the physics from Beamline._get_erf_losses() and Beamline.measure_fc()
as pure NumPy operations that operate on entire batches of envelopes simultaneously.

The transmission model:
  At each position s along the beamline, the beam has a Gaussian profile in x and y.
  The fraction of beam passing through an aperture of half-width w is:
    T(s) = 0.5 * [erf((w - mu) / (sqrt(2) * sigma)) - erf((-w - mu) / (sqrt(2) * sigma))]
  where mu = centroid offset, sigma = envelope / 2 (1-sigma beam size).
  
  Propagated transmission = cumulative product of per-element transmissions.
  FC reading = product of x and y propagated transmissions at the FC position.

Vectorization strategy:
  Instead of looping over files and then over positions, we compute erf() on
  arrays of shape (N_files, N_positions) in one call. scipy.special.erf is
  implemented in C and handles broadcasting natively, so this is ~1000x faster
  than the Python loop in Beamline.py for large batches.
"""

import numpy as np
from scipy.special import erf


# Minimum sigma to avoid division by zero (same as Beamline.py)
SIGMA_FLOOR = 1e-5


def build_aperture_array(
    s: np.ndarray,
    aperture_info: dict,
) -> np.ndarray:
    """
    Build the array of aperture half-widths along the beamline.
    
    This is computed ONCE per beamline config (independent of misalignments
    or steering inputs — the apertures are fixed hardware).
    
    Args:
        s: (N_rows,) position array from any envelope file
        aperture_info: dict from generate_data.build_aperture_info_from_config()
                       with keys: wall_width, offset, quad_apertures, slit_apertures
    
    Returns:
        slits_array: (N_rows,) half-width of the tightest aperture at each s position
    """
    wall_width = aperture_info["wall_width"]
    offset = aperture_info["offset"]
    
    slits_array = np.full_like(s, wall_width)
    
    # Quad apertures
    for quad in aperture_info.get("quad_apertures", []):
        loc = quad["loc"]
        aperture = quad["aperture"]
        if aperture < wall_width:
            idx = np.argmin(np.abs(s - loc + offset))
            slits_array[idx] = min(slits_array[idx], aperture)
    
    # Slit apertures (from sy.f)
    for slit in aperture_info.get("slit_apertures", []):
        loc = slit["loc"]
        aperture = slit["aperture"]
        if aperture < wall_width:
            idx = np.argmin(np.abs(s - loc + offset))
            slits_array[idx] = min(slits_array[idx], aperture)
    
    return slits_array


def build_aperture_array_from_config(
    s: np.ndarray,
    axis_cfg: dict,
) -> np.ndarray:
    """
    Build the aperture half-width array from the axis YAML config.
    
    This mirrors Beamline._generate_slits_array() from the reference
    implementation EXACTLY, including:
      - wall_width baseline everywhere
      - slits: point-apply at idx = argmin(|s - loc + offset|), only if
        aperture < wall_width
      - quads: same point-apply rule, keyed by quad loc (center)
      - RF cavities: apply over [low_idx, upper_idx] using plate_length,
        only if aperture < wall_width
    
    The YAML layout expected (see y_config.yaml):
        beamline_wall_width: 2.54
        offset: 1
        quad_apertures:
          HEBT2:Q3: {loc: 259.124, aperture: 2.6}
          ...
        slit:                               # optional
          Dragon-Gas-Slit-8: {loc: 720.557, aperture: 0.8}
          ...
        rf:                                 # optional
          BCAV1: {loc: 100.0, plate_length: 30.0, aperture: 1.0}
    
    Args:
        s: (N_rows,) position array from an envelope file (TRANSOPTR's s-grid)
        axis_cfg: dict loaded from the axis YAML config
    
    Returns:
        slits_array: (N_rows,) aperture half-widths (cm)
    
    Notes:
        - Apertures >= wall_width are skipped (they don't constrain).
        - If multiple apertures map to the same s-index (e.g. a slit inside
          a quad), we take the minimum via `min(slits_array[idx], aperture)`.
        - This replaces the buggy `build_aperture_array_from_syf` for the
          common case where the YAML carries all aperture info.
    """
    wall_width = axis_cfg.get("beamline_wall_width", 2.54)
    offset = axis_cfg.get("offset", 1.0)
    
    slits_array = np.full_like(s, wall_width, dtype=np.float64)
    
    # ── Quad apertures ───────────────────────────────────────────
    # Same convention as reference: argmin(|s - loc + offset|)
    for name, quad in axis_cfg.get("quad_apertures", {}).items():
        loc = quad.get("loc")
        aperture = quad.get("aperture")
        if loc is None or aperture is None:
            continue
        if aperture < wall_width:
            idx = np.argmin(np.abs(s - loc + offset))
            slits_array[idx] = min(slits_array[idx], aperture)
    
    # ── Slits ────────────────────────────────────────────────────
    # Reference takes a single `aperture` per slit. If the YAML instead
    # has separate x_aperture/y_aperture, we collapse to min (symmetric).
    for name, slit in axis_cfg.get("slit", {}).items():
        loc = slit.get("loc")
        if loc is None:
            continue
        if "aperture" in slit:
            aperture = slit["aperture"]
        else:
            # Fall back to min of x/y half-widths if provided separately
            x_ap = slit.get("x_aperture", wall_width)
            y_ap = slit.get("y_aperture", wall_width)
            aperture = min(x_ap, y_ap)
        if aperture < wall_width:
            idx = np.argmin(np.abs(s - loc + offset))
            slits_array[idx] = min(slits_array[idx], aperture)
    
    # ── RF cavity apertures ──────────────────────────────────────
    # Reference applies RF aperture across [low_idx, upper_idx] using
    # plate_length. Empty if no `rf` section in config.
    for name, rf in axis_cfg.get("rf", {}).items():
        loc = rf.get("loc")
        aperture = rf.get("aperture")
        plate_length = rf.get("plate_length", 0.0)
        if loc is None or aperture is None:
            continue
        if aperture < wall_width:
            lower_loc = loc - plate_length / 2.0
            upper_loc = loc + plate_length / 2.0
            low_idx = int(np.argmin(np.abs(s - lower_loc + offset)))
            up_idx = int(np.argmin(np.abs(s - upper_loc + offset)))
            if up_idx < low_idx:
                low_idx, up_idx = up_idx, low_idx
            slits_array[low_idx:up_idx + 1] = np.minimum(
                slits_array[low_idx:up_idx + 1], aperture
            )
    
    return slits_array


def build_aperture_array_from_syf(
    s: np.ndarray,
    syf_path: str,
    wall_width: float = 2.54,
    offset: float = 1.0,
    quad_apertures: list[dict] = None,
) -> np.ndarray:
    """
    Build aperture array by parsing slit calls directly from sy.f.
    
    This is the most reliable method since sy.f is the ground truth
    for the beamline geometry that TRANSOPTR actually uses.
    
    Args:
        s: position array from an envelope file
        syf_path: path to sy.f file
        wall_width: default pipe half-width (cm)
        offset: beamline offset parameter
        quad_apertures: list of dicts with 'loc' and 'aperture' from YAML
    """
    import re
    
    slits_array = np.full_like(s, wall_width)
    
    # Parse slit calls from sy.f
    # Format: call slit(x_half, y_half, wo, 'name')
    # We accumulate drift lengths to compute absolute positions
    slit_pattern = re.compile(
        r"call\s+slit\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,", re.IGNORECASE
    )
    drift_pattern = re.compile(
        r"call\s+drift\(\s*([\d.Ee+-]+)\s*,", re.IGNORECASE
    )
    # Also need bend, mquad lengths to track position — but actually
    # TRANSOPTR already gives us s in the envelope output, and the slits
    # appear at known s values. We need the cumulative s of each slit.
    
    # Simpler approach: parse sy.f to get slit half-widths and compute
    # their s-positions by accumulating drift/element lengths.
    # This requires tracking all element lengths which is complex.
    
    # Practical approach: use the s values from the envelope file.
    # TRANSOPTR marks slits as discontinuities in the envelope.
    # We can identify them by looking for sudden changes in the envelope
    # that correspond to slit apertures.
    
    # Best approach: hardcode from sy.f analysis or parse from config.
    # For HEBT2, the slits are the Dragon gas cell slits.
    # Let's parse them from sy.f properly:
    
    current_s = 0.0
    slits_found = []
    
    with open(syf_path, 'r') as f:
        for line in f:
            line_stripped = line.strip().lower()
            
            # Track position via drift calls
            drift_match = drift_pattern.search(line_stripped)
            if drift_match:
                drift_len = float(drift_match.group(1))
                current_s += drift_len
            
            # Track bends: call bend(radius, angle, ...)
            bend_match = re.search(
                r"call\s+bend\(\s*([\d.]+)\s*,\s*([\d./]+)", line_stripped
            )
            if bend_match:
                radius = float(bend_match.group(1))
                angle_expr = bend_match.group(2)
                # Handle expressions like "22.5/2"
                if '/' in angle_expr:
                    parts = angle_expr.split('/')
                    angle = float(parts[0]) / float(parts[1])
                else:
                    angle = float(angle_expr)
                arc_length = radius * angle * np.pi / 180.0
                current_s += arc_length
            
            # Track quads: call mquad(k, aperture, length, ...)
            mquad_match = re.search(
                r"call\s+mquad\([^,]+,\s*([\d.]+)\s*,\s*([\d.]+)",
                line_stripped
            )
            if mquad_match:
                quad_length = float(mquad_match.group(2))
                current_s += quad_length
            
            # Detect slits
            slit_match = slit_pattern.search(line_stripped)
            if slit_match:
                x_half = float(slit_match.group(1))
                y_half = float(slit_match.group(2))
                # Use the smaller of x and y (symmetric for now)
                half_width = min(x_half, y_half)
                slits_found.append({"loc": current_s, "aperture": half_width})
    
    # Apply slits to array
    for slit in slits_found:
        loc = slit["loc"]
        aperture = slit["aperture"]
        if aperture < wall_width:
            idx = np.argmin(np.abs(s - loc + offset))
            slits_array[idx] = min(slits_array[idx], aperture)
    
    # Apply quad apertures from YAML config
    if quad_apertures:
        for quad in quad_apertures:
            loc = quad["loc"]
            aperture = quad["aperture"]
            if aperture < wall_width:
                idx = np.argmin(np.abs(s - loc + offset))
                slits_array[idx] = min(slits_array[idx], aperture)
    
    return slits_array


def _erf_transmission_1d(
    envelope: np.ndarray,
    centroid: np.ndarray,
    slits: np.ndarray,
) -> np.ndarray:
    """
    Compute per-position transmission fraction for one transverse dimension.
    
    Fully vectorized: works on single envelopes (N_rows,) or batches (N_files, N_rows).
    
    Args:
        envelope: beam 2-sigma size. Shape (..., N_rows)
        centroid: beam centroid offset. Shape (..., N_rows)
        slits:    aperture half-widths. Shape (N_rows,) — broadcast over batch dims
    
    Returns:
        per_position_transmission: Shape (..., N_rows), values in [0, 1]
    """
    # sigma = 1-sigma = envelope / 2
    sigma = envelope / 2.0
    
    # Floor sigma to avoid division by zero
    sigma = np.maximum(sigma, SIGMA_FLOOR)
    
    sqrt2_sigma = np.sqrt(2.0) * sigma
    
    # T = 0.5 * [erf((w - mu) / (sqrt2 * sigma)) - erf((-w - mu) / (sqrt2 * sigma))]
    #   = 0.5 * [erf((w - mu) / (sqrt2 * sigma)) + erf((w + mu) / (sqrt2 * sigma))]
    transmission = 0.5 * (
        erf((slits - centroid) / sqrt2_sigma)
        + erf((slits + centroid) / sqrt2_sigma)
    )
    
    # Clamp to [0, 1] for numerical safety
    return np.clip(transmission, 0.0, 1.0)


def _propagated_transmission(per_position: np.ndarray) -> np.ndarray:
    """
    Compute cumulative (propagated) transmission along the beamline.
    
    Args:
        per_position: shape (..., N_rows), per-position transmission in [0, 1]
    
    Returns:
        propagated: shape (..., N_rows), cumulative product along last axis
    """
    return np.cumprod(per_position, axis=-1)


def compute_transmissions_vectorized(
    x_env: np.ndarray,
    y_env: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    slits_array: np.ndarray,
    fc_index: int = None,
    s: np.ndarray = None,
    fc_loc: float = None,
    offset: float = 1.0,
) -> np.ndarray:
    """
    Compute FC transmission for a batch of envelope files.
    
    This is the main entry point. It replaces the entire chain of:
      Beamline._get_erf_losses() → Beamline._calculate_transmission_integral() → Beamline.measure_fc()
    
    Args:
        x_env: (N_files, N_rows) x envelope (2-sigma)
        y_env: (N_files, N_rows) y envelope (2-sigma)
        x_cm:  (N_files, N_rows) x centroid
        y_cm:  (N_files, N_rows) y centroid
        slits_array: (N_rows,) aperture half-widths
        fc_index: pre-computed index into s array for the FC position.
                  If None, computed from s and fc_loc.
        s: (N_rows,) position array (needed if fc_index not provided)
        fc_loc: FC location in cm (needed if fc_index not provided)
        offset: beamline offset parameter
    
    Returns:
        transmissions: (N_files,) scalar transmission at the FC for each envelope
    """
    # Compute per-position transmissions for x and y
    # These are all shape (N_files, N_rows) — broadcasting handles the (N_rows,) slits
    tx = _erf_transmission_1d(x_env, x_cm, slits_array)
    ty = _erf_transmission_1d(y_env, y_cm, slits_array)
    
    # Propagated (cumulative) transmissions
    tx_prop = _propagated_transmission(tx)
    ty_prop = _propagated_transmission(ty)
    
    # FC measurement = product of x and y propagated transmissions at FC position
    if fc_index is None:
        if s is None or fc_loc is None:
            raise ValueError("Must provide either fc_index or both s and fc_loc")
        fc_index = np.argmin(np.abs(s - fc_loc + offset))
    
    # Extract scalar transmission at FC for each file
    transmissions = tx_prop[:, fc_index] * ty_prop[:, fc_index]
    
    return transmissions


def extract_fc_transmission(
    s: np.ndarray,
    fc_loc: float,
    offset: float = 1.0,
) -> int:
    """
    Get the index into the s array corresponding to the FC position.
    Pre-compute this once and reuse for all files.
    """
    return int(np.argmin(np.abs(s - fc_loc + offset)))
