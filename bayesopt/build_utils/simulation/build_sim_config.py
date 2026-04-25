"""
Module 3: Build the Bayesian optimization YAML config file.

Parses the trimmed path XML using the beam-config style (seq/element
iteration, PV-based keys) and adds simulation-specific data
(misalignment bounds, apertures, source misalignments).
Reads data.dat for starting values.
"""

import os
import yaml
from lxml import etree

from pyoptr import DataDat


# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------

DEFAULT_BEAMLINE_WALL_WIDTH = 2.54  # radius in cm

DEFAULT_MIS_QUAD = (0.0, 0.05)       # position misalignment bounds (cm)
DEFAULT_MIS_SLIT = (0.0, 1.0)
DEFAULT_MIS_DIPOLE_POS = (0.0, 0.05)
DEFAULT_MIS_DIPOLE_ANG = (0.0, 0.01)

DEFAULT_SOURCE_MIS = {
    'src_mis': {
        'x1': {'min': 0.0, 'max': 0.1},
        'y1': {'min': 0.0, 'max': 0.1},
        'x2': {'min': 0.0, 'max': 0.01},
        'y2': {'min': 0.0, 'max': 0.01},
    }
}

DEFAULT_SLIT_WIDTHS = {
    'x_lower': 0.5,
    'x_upper': 0.5,
    'y_lower': 1.0,
    'y_upper': 1.0,
}

DEFAULT_STEER_BOUNDS = (-0.002, 0.002)  # rad


# ---------------------------------------------------------------------------
# Element type classification (shared with extract_beam_config pattern)
# ---------------------------------------------------------------------------

def classify_element_type(raw_type):
    """Map raw XML element types to normalized category names."""
    mapping = {
        'ecb': 'steerer', 'mcb': 'steerer',
        'col': 'slit', 'slit': 'slit',
        'mq': 'quad', 'eq': 'quad',
        'mb': 'dipole',
        'fc': 'fc',
    }
    return mapping.get(raw_type, raw_type)


# ---------------------------------------------------------------------------
# Unit parsing
# ---------------------------------------------------------------------------

def parse_value_with_unit(attr_string, target_unit='cm'):
    """
    Parse a value*unit string (e.g. '123.4*mm') and convert to cm.
    """
    parts = attr_string.split('*')
    value = float(parts[0])
    if len(parts) > 1:
        unit = parts[1]
        if unit == 'mm' and target_unit == 'cm':
            value /= 10.0
    return value


# ---------------------------------------------------------------------------
# Core XML parsing — beam-config style with simulation extras
# ---------------------------------------------------------------------------

def parse_elements(trimmed_tree, datadat_path, apertures=None,
                   beamline_wall_width=DEFAULT_BEAMLINE_WALL_WIDTH):
    """
    Parse a trimmed path XML tree and extract element data.

    Uses the seq/element two-level iteration pattern and PV-based keys
    from extract_beam_config.py, combined with simulation-specific data
    (apertures, plate_length) from extract_config.py.

    Args:
        trimmed_tree: lxml Element — the trimmed path (flat <seq> or nested)
        datadat_path: path to data.dat for starting values
        apertures: dict mapping element IDs to custom aperture values
        beamline_wall_width: default aperture radius in cm

    Returns:
        dict of element data keyed by PV name (or element id)
    """
    if apertures is None:
        apertures = {}

    # Accept either an lxml Element directly or a file path
    if isinstance(trimmed_tree, str):
        tree = etree.parse(trimmed_tree)
        root = tree.getroot()
    else:
        root = trimmed_tree

    datadat = DataDat(file=datadat_path)

    data = {}

    # Handle both flat <seq> root and nested structures
    if root.tag == 'seq':
        sequences = [root]
    else:
        sequences = root.findall('seq')

    # Find first element for starting position
    first_elem = None
    for seq in sequences:
        elems = seq.findall('element')
        if elems:
            first_elem = elems[0]
            break

    if first_elem is None:
        raise ValueError("No elements found in path XML.")

    starting_length = parse_value_with_unit(first_elem.attrib['s'])

    # Parse all elements
    end_of_last_seq = 0
    prev_loc = 0

    for seq in sequences:
        for elem in seq.findall('element'):
            elem_id = elem.attrib['id']
            raw_type = elem.attrib['type']
            ele_type = classify_element_type(raw_type)

            # Position tracking
            loc = parse_value_with_unit(elem.attrib['s'])
            if loc < prev_loc:
                end_of_last_seq += prev_loc
            prev_loc = loc
            loc = loc + end_of_last_seq - starting_length

            # Length
            length = parse_value_with_unit(elem.attrib['l'])

            # Aperture from optr tag
            aper = beamline_wall_width
            optr_tag = elem.find('optr')
            if optr_tag is not None:
                if 'aperR' in optr_tag.attrib:
                    aper = parse_value_with_unit(optr_tag.attrib['aperR'])
                elif 'aperrx' in optr_tag.attrib:
                    # For slits, aperrx is half-width in mm, treat as radius
                    aper = parse_value_with_unit(optr_tag.attrib['aperrx']) / 10.0

            # Does not fit HEBT2:Q1/2 naming convention, fix manually for now
            key = elem_id
            if ele_type == 'quad' and ':CUR' not in elem_id:
                key = elem_id + ':CUR'

            # Always record the element
            data[key] = {
                'type': ele_type,
                'loc': loc,
                'aperture': aper,
                'plate_length': length,
            }

            # Parse EPICS setpoints
            for epics in elem.findall('epics'):
                for setpoint in epics.findall('setpoint'):
                    pv_name = setpoint.attrib.get('pv', elem_id)
                    data[key]['pv'] = pv_name

                    if 'min' not in setpoint.attrib:
                        continue

                    # Determine bounds
                    if ele_type == 'steerer':
                        pv_min = DEFAULT_STEER_BOUNDS[0]
                        pv_max = DEFAULT_STEER_BOUNDS[1]
                    else:
                        pv_min = float(setpoint.attrib['min'])
                        pv_max = float(setpoint.attrib['max'])

                    if ele_type in ('steerer', 'dipole'):
                        pv_min = -0.02
                        pv_max = 0.02

                    # Starting value from data.dat, fall back to midpoint
                    try:
                        lookup_key = data[key].get('pv', elem_id)
                        if ele_type == 'quad' and ':CUR' not in elem_id:
                            lookup_key = elem_id + ':CUR'
                        start_val = datadat.get_element(name=lookup_key)['value']
                    except Exception:
                        start_val = (pv_max + pv_min) / 2.0

                    # Check for zero crossing
                    set_zero_crossing = (pv_min * pv_max < 0)

                    # Aperture override by element ID
                    elem_aper = apertures.get(elem_id, aper)
                    
                    data[key].update({
                        'lower_bound': pv_min,
                        'upper_bound': pv_max,
                        'starting_mean': start_val,
                        'starting_var': (pv_max - pv_min) / 10.0,
                        'aperture': elem_aper,
                        'plate_length': length,
                        'max_incremental_change': pv_max - pv_min,
                        'set_zero_crossing': set_zero_crossing,
                    })

    return data


# ---------------------------------------------------------------------------
# Group elements by type with misalignment metadata
# ---------------------------------------------------------------------------

def group_elements(data, mis_quad=DEFAULT_MIS_QUAD, mis_slit=DEFAULT_MIS_SLIT,
                   mis_dipole_pos=DEFAULT_MIS_DIPOLE_POS,
                   mis_dipole_ang=DEFAULT_MIS_DIPOLE_ANG,
                   slit_widths=DEFAULT_SLIT_WIDTHS):
    """
    Group parsed element data by type and attach misalignment bounds.

    Args:
        data: dict of element data from parse_elements()
        mis_quad: (min, max) quad misalignment bounds
        mis_slit: (min, max) slit misalignment bounds
        mis_dipole_pos: (min, max) dipole position misalignment bounds
        mis_dipole_ang: (min, max) dipole angular misalignment bounds
        slit_widths: dict with x_lower, x_upper, y_lower, y_upper

    Returns:
        dict of {element_type: {element_id: {properties}}}
    """
    grouped = {}

    for key, props in data.items():
        etype = props['type']

        if etype not in grouped:
            grouped[etype] = {}

        grouped[etype][key] = props

        # Attach misalignment metadata per type
        if etype == 'quad':
            props['misalignments_min'] = mis_quad[0]
            props['misalignments_max'] = mis_quad[1]
        elif etype == 'dipole':
            props['pos_mis_min'] = mis_dipole_pos[0]
            props['pos_mis_max'] = mis_dipole_pos[1]
            props['ang_mis_min'] = mis_dipole_ang[0]
            props['ang_mis_max'] = mis_dipole_ang[1]
        elif etype == 'slit':
            props['misalignments_min'] = mis_slit[0]
            props['misalignments_max'] = mis_slit[1]
            props['x_lower'] = slit_widths['x_lower']
            props['x_upper'] = slit_widths['x_upper']
            props['y_lower'] = slit_widths['y_lower']
            props['y_upper'] = slit_widths['y_upper']

    return dict(sorted(grouped.items()))


# ---------------------------------------------------------------------------
# Write the YAML config file
# ---------------------------------------------------------------------------

def build_sim_config(name, trimmed_tree, datadat_path, output_yaml_path,
                     tuning_elements, measurement_device, apertures=None,
                     beamline_wall_width=DEFAULT_BEAMLINE_WALL_WIDTH,
                     acc_path=None, offset=1, random_seed='Null'):
    """
    Build and write a simulation YAML config for Bayesian optimization.

    Args:
        name: configuration name
        trimmed_tree: lxml Element of the trimmed path, or a file path string
        datadat_path: path to data.dat
        output_yaml_path: path to write the output YAML
        tuning_elements: list of element types to tune (e.g. ['steerer'])
        measurement_device: name of the measurement device (FC)
        apertures: dict of element-specific aperture overrides
        beamline_wall_width: default aperture in cm
        acc_path: accelerator path string (e.g. 'ios-mws-hebt1-prague')
        offset: offset parameter
        random_seed: random seed value
    """
    optr_dir = os.path.dirname(os.path.abspath(datadat_path))
    misalignment_elements = ['quad', 'src_mis']

    # Parse and group elements
    data = parse_elements(trimmed_tree, datadat_path, apertures, beamline_wall_width)
    grouped = group_elements(data)
    element_data = {'elements': grouped}

    with open(output_yaml_path, 'w') as f:
        print(f'name:  {name}', file=f)
        print(f'type:  beamline', file=f)
        print(f'random_seed:  {random_seed}', file=f)
        print(f'\nacc_path:  {acc_path}', file=f)
        print(f'\ndatadat_path: {optr_dir}/data.dat', file=f)
        print(f'optr_dir: {optr_dir}', file=f)
        print(f'\nbeamline_wall_width: {beamline_wall_width}', file=f)
        print(f'offset: {offset}', file=f)
        print(f'tuning_elements: {str(tuning_elements)}', file=f)
        print(f'misalignment_elements: {str(misalignment_elements)}', file=f)
        print(f'measurement_device: {measurement_device}', file=f)

        yaml.dump(DEFAULT_SOURCE_MIS, f, default_flow_style=False, sort_keys=False)
        yaml.dump(element_data, f, default_flow_style=False, sort_keys=False)

    print(f'Simulation config YAML written to {output_yaml_path}')
