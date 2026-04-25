import os, sys
from pathlib import Path
from lxml import etree
from copy import deepcopy

import xml2optr
from xml2optr.utilities import fuzzy_find_path, fuzzy_find_tune
from xml2optr.operators import ideal_steerers, misalign_shift, misalign_bend
from xml2optr.elements import *
from xml2optr.constants import *
from xml2optr.utilities import *

from pyoptr import DataDat
from pyoptr.build import build


def custom_translate(path_root, tune_root, output_dir, path_root_source, tune_root_source, element_map, force, add_src_mis=False):
    """
    export_simulation creates and copies all necessary input files to run a transoptr simulation

    Args:
        path_root: the etree lxml object for a beam path
        tune_root: the etree lxml object for a tune file, it is only optional if force is set to True
        output_dir: the directory to output the created files to
        path_root_source: optional, the path to the beampath xml source, enables error reporting with line references
        tune_root_source: optional, the path to the tune xml source, enables error reporting with line references
        element_map: optional, the dictionary that maps types to functions, if not given will load from the acc configuration 
        force: try to ignore errors and proceed with the program
    """
    if element_map is None:
        element_map = get_type_func_mapping()

    # initialize Datadat  with default values
    datadat = DataDat(dict=DEFAULT_DATADAT)
    start_id = None
    end_id = None

    try:
        datadat, start_id, end_id = parse_tune_initial(datadat, tune_root, tune_root_source=tune_root_source)
    except Exception as e:
        if force:
            print("Warning, error processing optr tag in tune file, proceeding anyway...")
        else:
            raise e

    # Initialize state data structures
    syf = []
    counters = {'unit': 100}
    total_s = 0.0

    include_elements = True
    if (start_id is not None):
        include_elements = False

    for seq in path_root.iterfind('seq'):

        prev_s = 0.0
        prev_l = 0.0

        seq_source_file = os.path.split(seq.base)[1]
        if not seq_source_file:
            seq_source_file = path_root_source

        for elem in seq.iterfind('element'):

            elem_id = elem.get('id', None)

            if elem_id is None:
                elem_id = "undefined"
                print("Warning element with undefined id in file {} at line {}".format(seq_source_file, elem.sourceline))

            if (start_id is not None) and elem_id == start_id:
                include_elements = True
                prev_s = float(parse_attributes(elem, [('s', '0.0', '/cm')], source_file=seq_source_file)['s'])

            if include_elements:
                attrs = parse_attributes(elem, [('s', prev_s, '/cm'), ('l', '0.0', '/cm')], source_file=seq_source_file)

                try:
                    s = float(attrs['s'])
                except:
                    print("Warning element with id={} in file {} at line {}\n    Attribute 's' not defined".format(elem_id, seq_source_file, elem.sourceline))
                    continue

                elem_type = elem.attrib.get("type", "").replace(' ', '')

                elem_type_func = element_map.get(elem_type, UNKNOWN)

                if(elem.find("optr") is not None and
                   'disabled' in [key.lower() for key in elem.find("optr").attrib.keys()]):
                    elem_type_func = MARKER

                if hasattr(elem_type_func, 'zero_length'):
                    l = 0.0
                else:
                    l = float(attrs['l'])

                drift_l = s - prev_s - l/2.0 - prev_l/2.0

                prev_l = l
                prev_s = s

                # add a drift between the previous element and this one:
                if abs(drift_l) > MIN_DRIFT_LENGTH:
                    syf.extend(["call drift({:.6},{})".format(drift_l, '"."')])

                try:
                    syf, datadat, counters = elem_type_func(elem, syf, datadat, counters, output_dir)
                except Exception as e:
                    error_message = ["Element with id={} in file {} at line {}".format(elem_id, seq_source_file, elem.sourceline),
                                     *e.args]
                    traceback = sys.exc_info()[2]
                    raise type(e)("\n".join(error_message)).with_traceback(traceback) from None

                if (end_id is not None) and end_id == elem_id:
                    break
        else:
            total_s += prev_s
            continue

        total_s += prev_s
        break

    if(include_elements is False):
        raise ValueError('Error starting element with id={} was not found'.format(start_id))

    common_block = ""
    fort_vars = [element.get('var', '') for element in datadat['elements']]

    if add_src_mis:
        fort_vars = ['SGX1','SGY1','SGX2','SGY2'] + fort_vars
        
        for i, line in enumerate(syf):
            if len(line) > 10:
                if line[:10] == 'call drift':
                    break
        syf.insert(i, 'call genshift(SGX1,SGX2,SGY1,SGY2)')
        
    if len(fort_vars) > 0:
        common_block = "COMMON/BLOC1/"+','.join(fort_vars)

    header = [
        "SUBROUTINE TSYSTEM",
        "COMMON/SCPARM/QSC,ISC,CMPS",
        "COMMON/MOM/P,BRHO,pMASS,ENERGK,GSQ,ENERGKi,charge,current",
        "COMMON/PRINT/IPRINT",
        "COMMON/SS/SX(13,6)",
        common_block,  # more things go here!
        "",
        "CMPS={:.3}   ! Number of cm per step, for plotting only".format(float(total_s/MIN_PLOT_POINTS)),
        "wo=1.0 ! Weight aberration from optical elements",
        "",
        acc_version(),
        ""
    ]

    footer = [
        "call print_transfer_matrix",
        "return",
        "end",
        ""
    ]
        
    # Combine the different pieces of sy.f:
    syf = header + syf + footer
    # output to sy.f
    syf = format77(syf)  # format list of strings

    # Load Tune values for PVs
    try:
        datadat = parse_tune_pvs(datadat, tune_root, tune_root_source=tune_root_source)
    except Exception as e:
        if force:
            print("Warning, problem encountered loading PVs from tune file, proceeding anyway...")
        else:
            raise e

    # Set default values if the tune file was missing some
    for element in datadat['elements']:
        if element['value'] is None:
            if element['min'] <= 0.0 and 0.0 <= element['max']:
                element['value'] = 0.0
            else:
                element['value'] = element['min']

    # Write the sy.f and data.dat files
    with open(os.path.join(output_dir, 'sy.f'), 'w') as syf_file:
        syf_file.write('\n'.join(syf))

    with open(os.path.join(output_dir, 'data.dat'), 'w') as dat_file:
        dat_file.write(datadat.to_string())

    print("xml2optr translation completed sucessfully!")
    
    
def build_syf(transoptr_dir, tune_config, tune_path):
    path = tune_path
    name = transoptr_dir.split('/')[-1]
    given_tune = tune_config
    path_info = fuzzy_find_path(path)
    tune_info = fuzzy_find_tune(given_tune=given_tune, path_name=path)

    # with Path("path_dump.xml").open("wb") as f:
    #     f.write(etree.tostring(path_info['root'], pretty_print=True))
    # with Path("tune_dump.xml").open("wb") as f:
    #     f.write(etree.tostring(tune_info['root'], pretty_print=True))

    start_id = tune_info['root'].find('optr').get('start')
    end_id = tune_info['root'].find('optr').get('end')

    root = deepcopy(path_info['root'])

    for i, seq in enumerate(root):
        for j, elem in enumerate(seq):
            id = elem.get('id', None)
            if id == start_id:
                start_seq_index = i
                start_elem_index = j
            if id == end_id:
                end_seq_index = i
                end_elem_index = j
                break
        
    while(len(root) > end_seq_index+1):
        root.remove(root[-1])

    while(len(root[-1]) > end_elem_index+1):
        root[-1].remove(root[-1][-1])

    for i in range(start_seq_index):
        root.remove(root[0])
        
    for j in range(start_elem_index):
        root[0].remove(root[0][0])
        

    tree = etree.Element('seq')
    for seq in root:
        for elem in seq:
            tree.append(elem)
        
    with Path(os.path.join(transoptr_dir, name+"_path.xml")).open("wb") as f:
        f.write(etree.tostring(tree, pretty_print=True))

    default_element_map = xml2optr.main.get_type_func_mapping()

    misalign_elements = ['eq', 'mq']
    quads_misalign_bound = [1.0, 1.0] #+/- X and Y
    benders_misalign_bound = [1.0, 1.0, 1.0, 1.0, 1.0] #+/- X,Y and Z (in cm), +/- angle X and Y in rad

    benders = ['eb', 'yeb', 'mb', 'ymb']
    steerers = ['ecb', 'mcb']
    steerers_map = ideal_steerers(2.0, 1.0)  #Range of steerers in rad, firsts_input: +/- X ; second_input: +/- Y

    element_map = {}

    for element_type, element_func in default_element_map.items():
        if (element_type in misalign_elements):
            element_map[element_type] = misalign_shift(element_func, quads_misalign_bound[0], quads_misalign_bound[1])
        elif (element_type in benders):
            element_map[element_type] = misalign_bend(element_func, benders_misalign_bound[0], benders_misalign_bound[1], benders_misalign_bound[2], benders_misalign_bound[3], benders_misalign_bound[4])
        elif (element_type in steerers):
            element_map[element_type] = steerers_map[element_type]
        else:
            element_map[element_type] = element_func

    custom_translate(path_info['root'], tune_info['root'], transoptr_dir, path_info['source'], tune_info['source'], element_map, force=True, add_src_mis=True)

if __name__ == '__main__':
    directory = os.path.dirname(os.path.abspath(__file__))
    name = os.path.basename(directory)
    build_dir = f"/home/{os.getlogin()}/repos/bayesopt/build_utils/{name}"
    output_dir = os.path.join(f'/home/{os.getlogin()}/repos/bayesopt/transoptr/{name}')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tune_config = os.path.join(build_dir, 'tune_config.xml')
    
    # Parse the XML file
    tree = etree.parse(tune_config)
    root = tree.getroot()

    # Extract the value of the 'path' attribute
    tune_path = root.get("path")
    
    build_syf(transoptr_dir=output_dir, tune_config=tune_config, tune_path=tune_path)
    
    datadat = DataDat(file = os.path.join(output_dir, 'data.dat'))
    src_misalignments = [
            {'name': 'MISALIGNX1', 'min': -1.0, 'max': 1.0, 'unit': 'cm', 'var': 'SGX1', 'optimize': 0, 'value': 0.0},
            {'name': 'MISALIGNY1', 'min': -1.0, 'max': 1.0, 'unit': 'cm', 'var': 'SGY1', 'optimize': 0, 'value': 0.0},
            {'name': 'MISALIGNX2', 'min': -1.0, 'max': 1.0, 'unit': 'rad', 'var': 'SGX2', 'optimize': 0, 'value': 0.0},
            {'name': 'MISALIGNY2', 'min': -1.0, 'max': 1.0, 'unit': 'rad', 'var': 'SGY2', 'optimize': 0, 'value': 0.0},
        ]
    for i in reversed(src_misalignments):
        datadat.data['elements'].insert(0, i)
        
    datadat.data['num-element'] += len(src_misalignments)
    
    with open(os.path.join(output_dir, 'data.dat'), 'w') as dat_file:
        dat_file.write(datadat.to_string())
        
    #changing the OPTRDIR to local
    user = os.getlogin()
    os.environ['OPTRDIR'] = f'/home/{user}/repos/transoptr'

    #build the executable optr in build_dir with optimize=True (faster executable)
    build(output_dir,optimize=True)
            
        



