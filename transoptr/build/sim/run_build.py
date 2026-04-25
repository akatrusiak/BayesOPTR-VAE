#!/usr/bin/env python3
"""
Orchestrator: runs the simulation build pipeline.

Everything the pipeline needs is derived from the tune config XML:
  - Accelerator path name:  root[@path]
  - Start element:          optr[@start]
  - End element (= FC):     optr[@end]
  - Beam parameters:        optr and tune tags
  - Quad/element setpoints: set tags

Steps:
  1. Build trimmed path XML from $ACCDIR
  2. Build TRANSOPTR files (sy.f, data.dat, executable)
  3. Build Bayesian optimization YAML config

Usage:
    python run_build.py --name mebt-hebt --tune_config /path/to/tune_config.xml

    # Override defaults:
    python run_build.py --name mebt-hebt --tune_config tune_config.xml \\
        --device_types steerer quad \\
        --measurement_device HEBT:FC5
"""

import os
import argparse
from lxml import etree

from build_xml import build_trimmed_xml
from build_transoptr import build_transoptr
from build_sim_config import build_sim_config


def parse_tune_config(tune_config_path):
    """
    Extract key parameters from the tune config XML.

    Returns:
        dict with keys: path, start, end, measurement_device
    """
    tree = etree.parse(tune_config_path)
    root = tree.getroot()

    path_name = root.get('path')
    if path_name is None:
        raise ValueError("Tune config XML missing 'path' attribute on root element.")

    optr = root.find('optr')
    if optr is None:
        raise ValueError("Tune config XML missing <optr> tag.")

    start_id = optr.get('start')
    end_id = optr.get('end')

    if start_id is None or end_id is None:
        raise ValueError("Tune config <optr> tag missing 'start' or 'end' attribute.")

    return {
        'path': path_name,
        'start': start_id,
        'end': end_id,
        'measurement_device': end_id,  # FC at the end of the section
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build simulation pipeline: trimmed XML -> TRANSOPTR -> BO config'
    )

    # Required
    parser.add_argument('--name', required=True,
                        help='Name for this configuration (directory and file naming)')
    parser.add_argument('--tune_config', required=True,
                        help='Path to tune_config.xml (contains path, start/end, beam params)')

    # Optional overrides (defaults derived from tune XML)
    parser.add_argument('--device_types', nargs='+', default=None,
                        help="Element types to tune (default: ['steerer'])")
    parser.add_argument('--measurement_device', default=None,
                        help='Measurement device name (default: end element from tune XML)')

    # Step control
    parser.add_argument('--skip_transoptr', action='store_true',
                        help='Skip step 2 (use existing TRANSOPTR files)')
    parser.add_argument('--skip_config', action='store_true',
                        help='Skip step 3 (only build XML and TRANSOPTR)')
    parser.add_argument('--write_xml', action='store_true',
                        help='Write the trimmed path XML to disk (for inspection)')

    args = parser.parse_args()

    # ── Parse tune config ──────────────────────────────────────────────
    tune_params = parse_tune_config(args.tune_config)

    # Apply defaults and overrides
    device_types = args.device_types if args.device_types else ['steerer']
    measurement_device = args.measurement_device if args.measurement_device else tune_params['measurement_device']

    # Set up directory paths relative to where this script lives
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')  # adjust as needed       
    transoptr_dir = os.path.join(project_root, 'simulation', 'transoptr', args.name)
    config_dir = os.path.join(project_root, 'config', 'sim')

    os.makedirs(transoptr_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    datadat_path = os.path.join(transoptr_dir, 'data.dat')
    output_yaml = os.path.join(config_dir, f'{args.name}.yaml')

    # Optionally write trimmed XML path
    xml_output_path = None
    if args.write_xml:
        xml_output_path = os.path.join(transoptr_dir, f'{args.name}_path.xml')

    # ── Step 1: Build trimmed XML ──────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 1: Building trimmed path XML')
    print('=' * 60)

    trimmed_tree = build_trimmed_xml(
        path_name=tune_params['path'],
        start_id=tune_params['start'],
        end_id=tune_params['end'],
        output_path=xml_output_path,
    )

    print(f"  Path:  {tune_params['path']}")
    print(f"  Start: {tune_params['start']}")
    print(f"  End:   {tune_params['end']}")
    num_elements = len(trimmed_tree.findall('element'))
    print(f"  Elements in trimmed path: {num_elements}")

    # ── Step 2: Build TRANSOPTR files ──────────────────────────────────
    if not args.skip_transoptr:
        print('\n' + '=' * 60)
        print('Step 2: Building TRANSOPTR files')
        print('=' * 60)

        build_transoptr(
            tune_config_path=args.tune_config,
            output_dir=transoptr_dir,
        )
    else:
        print('\n[Skipping Step 2: TRANSOPTR build]')
        if not os.path.isfile(datadat_path):
            print(f"  WARNING: {datadat_path} does not exist. Step 3 will fail.")

    # ── Step 3: Build BO YAML config ──────────────────────────────────
    if not args.skip_config:
        print('\n' + '=' * 60)
        print('Step 3: Building simulation YAML config')
        print('=' * 60)

        # Default apertures — customize per beamline as needed
        apertures = {
            'ISAC1:DTL1': 0.69977,
            'ISAC1:DTL2': 0.499999,
            'ISAC1:DTL3': 0.8001,
            'ISAC1:DTL4': 0.8001,
            'ISAC1:DTL5': 0.8001,
            'ISAC1:MEBT': 1.0,
        }

        build_sim_config(
            name=args.name,
            trimmed_tree=trimmed_tree,
            datadat_path=datadat_path,
            output_yaml_path=output_yaml,
            tuning_elements=device_types,
            measurement_device=measurement_device,
            apertures=apertures,
            acc_path=tune_params['path'],
        )
    else:
        print('\n[Skipping Step 3: YAML config build]')

    # ── Summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Pipeline complete!')
    print(f'  Tune config: {args.tune_config}')
    print(f'  Path:        {tune_params["path"]}')
    print(f'  Section:     {tune_params["start"]} -> {tune_params["end"]}')
    if args.write_xml:
        print(f'  Path XML:    {xml_output_path}')
    if not args.skip_transoptr:
        print(f'  TRANSOPTR:   {transoptr_dir}/')
    if not args.skip_config:
        print(f'  YAML config: {output_yaml}')
        print(f'  Device types: {device_types}')
        print(f'  Measurement:  {measurement_device}')
    print('=' * 60)


if __name__ == '__main__':
    main()
