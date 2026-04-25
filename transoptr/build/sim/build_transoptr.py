"""
Module 2: Build TRANSOPTR simulation files.

Uses xml2optr's translate() function directly with setup hooks
to inject source misalignment handling. Wraps element handlers with
misalignment operators for quads, benders, and steerers.
"""

import os

from lxml import etree

from xml2optr.main import get_type_func_mapping, translate
from xml2optr.utilities import fuzzy_find_path, fuzzy_find_tune
from xml2optr.operators import ideal_steerers, misalign_shift, misalign_bend

from pyoptr import DataDat
from pyoptr.build import build


# ---------------------------------------------------------------------------
# Misalignment bounds (hardcoded defaults, later can be randomized per-run)
# ---------------------------------------------------------------------------

MISALIGN_QUAD_BOUNDS = (1.0, 1.0)  # +/- X and Y in cm
MISALIGN_BENDER_BOUNDS = {         # named for clarity with misalign_bend signature
    'x_range': 1.0,
    'y_range': 1.0,
    'z_range': 1.0,
    'strength_range': 1.0,
    'roll_range': 1.0,
}

STEERER_RANGE_X = 2.0   # mrad
STEERER_RANGE_Y = 1.0   # mrad

SRC_MISALIGNMENTS = [
    {'name': 'MISALIGNX1', 'min': -1.0, 'max': 1.0, 'unit': 'cm',  'var': 'SGX1', 'optimize': 0, 'value': 0.0},
    {'name': 'MISALIGNY1', 'min': -1.0, 'max': 1.0, 'unit': 'cm',  'var': 'SGY1', 'optimize': 0, 'value': 0.0},
    {'name': 'MISALIGNX2', 'min': -1.0, 'max': 1.0, 'unit': 'rad', 'var': 'SGX2', 'optimize': 0, 'value': 0.0},
    {'name': 'MISALIGNY2', 'min': -1.0, 'max': 1.0, 'unit': 'rad', 'var': 'SGY2', 'optimize': 0, 'value': 0.0},
]

# Element types that receive each kind of misalignment wrapper
MISALIGN_ELEMENTS = ['eq', 'mq']
BENDER_ELEMENTS = ['eb', 'yeb', 'mb', 'ymb']
STEERER_ELEMENTS = ['ecb', 'mcb']


# ---------------------------------------------------------------------------
# Build element map with misalignment wrappers
# ---------------------------------------------------------------------------

def build_element_map(steerer_range_x=STEERER_RANGE_X,
                      steerer_range_y=STEERER_RANGE_Y,
                      quad_mis_bounds=MISALIGN_QUAD_BOUNDS,
                      bender_mis_bounds=MISALIGN_BENDER_BOUNDS):
    """
    Build the element_map dict that maps XML element types to their
    handler functions, with misalignment and steerer wrappers applied.

    Args:
        steerer_range_x: steerer range in X (mrad)
        steerer_range_y: steerer range in Y (mrad)
        quad_mis_bounds: (x, y) misalignment bounds for quads in cm
        bender_mis_bounds: dict with x_range, y_range, z_range,
                           strength_range, roll_range for benders

    Returns:
        dict mapping element type strings to handler functions
    """
    default_map = get_type_func_mapping()
    steerers_map = ideal_steerers(steerer_range_x, steerer_range_y)

    element_map = {}
    for etype, efunc in default_map.items():
        if etype in MISALIGN_ELEMENTS:
            element_map[etype] = misalign_shift(
                efunc, quad_mis_bounds[0], quad_mis_bounds[1]
            )
        elif etype in BENDER_ELEMENTS:
            element_map[etype] = misalign_bend(efunc, **bender_mis_bounds)
        elif etype in STEERER_ELEMENTS:
            element_map[etype] = steerers_map[etype]
        else:
            element_map[etype] = efunc

    return element_map


# ---------------------------------------------------------------------------
# Setup function: inject source misalignment genshift into sy.f
# ---------------------------------------------------------------------------

def source_misalignment_setup(syf, datadat, counters, output_dir):
    """
    Setup hook for translate(): prepends the source misalignment
    genshift call and COMMON block variables.

    This is passed as the `setup` argument to xml2optr.translate(),
    which calls it before the element iteration begins.
    """
    # Add the genshift call that will appear before the first drift
    syf.extend([
        'call genshift(SGX1,SGX2,SGY1,SGY2)',
    ])

    # Add the source misalignment variables to the elements list.
    # translate() builds COMMON/BLOC1 from datadat['elements'] vars,
    # so these will be included in the Fortran COMMON block.
    for entry in reversed(SRC_MISALIGNMENTS):
        datadat['elements'].insert(0, entry.copy())
        datadat['num-element'] += 1

    return syf, datadat, counters


# ---------------------------------------------------------------------------
# Inject source misalignments into data.dat after translation
# ---------------------------------------------------------------------------

def inject_source_misalignments(datadat_path, src_mis=None):
    """
    Prepend source misalignment variables to an existing data.dat file.

    Note: if using the setup hook with translate(), the misalignment
    variables are already in the DataDat elements list and will be
    written to data.dat. This function is only needed if you're
    working with a data.dat that was generated without the setup hook.

    Args:
        datadat_path: path to the data.dat file to modify in place
        src_mis: list of misalignment dicts, or None for defaults
    """
    if src_mis is None:
        src_mis = SRC_MISALIGNMENTS

    datadat = DataDat(file=datadat_path)

    for entry in reversed(src_mis):
        datadat.data['elements'].insert(0, entry.copy())

    datadat.data['num-element'] += len(src_mis)

    with open(datadat_path, 'w') as f:
        f.write(datadat.to_string())

    print("Source misalignments injected into data.dat")


# ---------------------------------------------------------------------------
# build_transoptr: full pipeline from tune config to compiled executable
# ---------------------------------------------------------------------------

def build_transoptr(tune_config_path, output_dir):
    """
    Build TRANSOPTR simulation files: sy.f, data.dat, and compiled executable.

    The tune config XML provides everything needed:
    - path name (root[@path])
    - start/end elements (optr[@start], optr[@end])
    - beam parameters (optr and tune tags)
    - PV setpoints (set tags)

    Uses xml2optr.translate() directly with:
    - element_map: default handlers wrapped with misalignment operators
    - setup: source_misalignment_setup to inject genshift call

    Args:
        tune_config_path: path to the tune_config.xml file
        output_dir: directory to write output files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse tune config to get the path name
    tree = etree.parse(tune_config_path)
    root = tree.getroot()
    tune_path = root.get("path")

    # Use xml2optr's fuzzy finders to load path and tune
    path_info = fuzzy_find_path(tune_path)
    tune_info = fuzzy_find_tune(given_tune=tune_config_path, path_name=tune_path)

    # Build element map with misalignment wrappers
    element_map = build_element_map()

    # Use xml2optr's translate() directly with our setup hook
    translate(
        path_info['root'],
        tune_info['root'],
        output_dir,
        path_info['source'],
        tune_info['source'],
        element_map=element_map,
        setup=source_misalignment_setup,
        force=True,
    )

    # Compile TRANSOPTR executable
    # OPTRDIR should be set in the system environment already.
    # Only raise a warning if it's missing — pyoptr.build will
    # give a clear error if it's not set.
    if os.environ.get('OPTRDIR') is None:
        print("WARNING: $OPTRDIR is not set. pyoptr.build() will fail "
              "unless TRANSOPTR is installed and OPTRDIR is configured.")
    build(output_dir, optimize=True)

    print(f"TRANSOPTR build complete in {output_dir}")
