"""
Module 1: Build a trimmed beamline path XML file.

Loads the full accelerator path from $ACCDIR, resolves XIncludes,
and trims to the specified start and end elements.
"""

import os
from copy import deepcopy
from lxml import etree
from xml.etree.ElementTree import ElementTree, fromstring


def load_xml_with_include(path, as_ET=False):
    """
    Load an XML file using lxml, resolving XIncludes and DTD entities.

    Args:
        path: path to the XML file
        as_ET: if True, convert the lxml tree to a stdlib ElementTree

    Returns:
        The root element (lxml or stdlib depending on as_ET)
    """
    parser = etree.XMLParser(load_dtd=True, resolve_entities=True)
    tree = etree.parse(path, parser)
    tree.xinclude()
    root = tree.getroot()

    if as_ET:
        xml_bytes = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        root = ElementTree(fromstring(xml_bytes)).getroot()

    return root


def trim_path(root, start_id, end_id):
    """
    Trim a beamline path XML tree to only include elements between
    start_id and end_id (inclusive).

    Args:
        root: lxml root element containing <seq>/<element> structure
        start_id: the 'id' attribute of the first element to keep
        end_id: the 'id' attribute of the last element to keep

    Returns:
        An lxml Element containing the trimmed path as a flat <seq>.

    Raises:
        ValueError: if start or end elements are not found, or end before start.
    """
    root = deepcopy(root)

    start_found = False
    end_found = False
    start_seq_idx = None
    start_elem_idx = None
    end_seq_idx = None
    end_elem_idx = None

    for i, seq in enumerate(root.findall('seq')):
        for j, elem in enumerate(seq.findall('element')):
            elem_id = elem.get('id', None)
            if elem_id == start_id:
                start_found = True
                start_seq_idx = i
                start_elem_idx = j
            if elem_id == end_id:
                if not start_found:
                    raise ValueError(
                        f"End element '{end_id}' appears before start element "
                        f"'{start_id}', or '{start_id}' does not exist in the path."
                    )
                end_found = True
                end_seq_idx = i
                end_elem_idx = j
                break
        if end_found:
            break

    if not start_found:
        raise ValueError(f"Start element '{start_id}' not found in path.")
    if not end_found:
        raise ValueError(f"End element '{end_id}' not found in path.")

    # Trim trailing sequences
    while len(root) > end_seq_idx + 1:
        root.remove(root[-1])

    # Trim trailing elements in last kept sequence
    last_seq = root[end_seq_idx]
    elements = last_seq.findall('element')
    while len(elements) > end_elem_idx + 1:
        last_seq.remove(elements[-1])
        elements = last_seq.findall('element')

    # Trim leading sequences
    for _ in range(start_seq_idx):
        root.remove(root[0])

    # Trim leading elements in first kept sequence
    first_seq = root[0]
    for _ in range(start_elem_idx):
        first_seq.remove(first_seq.findall('element')[0])

    # Flatten into a single <seq>
    flat_seq = etree.Element('seq')
    for seq in root:
        for elem in seq:
            flat_seq.append(elem)

    return flat_seq


def build_trimmed_xml(path_name, start_id, end_id, output_path=None):
    """
    Load a beamline path from $ACCDIR, trim it, and optionally write to disk.

    Args:
        path_name: accelerator path name (e.g. 'ios-mws-hebt1-prague').
                   Can also be a direct file path to an XML file.
        start_id: id of the starting element
        end_id: id of the ending element
        output_path: file path to write the trimmed XML, or None to skip writing

    Returns:
        The trimmed lxml Element (a flat <seq> with all elements).
    """
    from pathlib import Path

    path_file = Path(path_name)
    if path_file.is_file():
        full_root = load_xml_with_include(str(path_file))
    else:
        acc_dir = os.environ.get('ACCDIR')
        if acc_dir is None:
            raise EnvironmentError("ACCDIR environment variable is not set.")

        # Search across facility directories for the path
        xml_file_path = None
        for facility in os.listdir(acc_dir):
            candidate = os.path.join(acc_dir, facility, 'path', path_name + '.xml')
            if os.path.isfile(candidate):
                xml_file_path = candidate
                break

        if xml_file_path is None:
            raise FileNotFoundError(
                f"Could not find path '{path_name}' in $ACCDIR ({acc_dir})"
            )

        full_root = load_xml_with_include(xml_file_path)

    trimmed = trim_path(full_root, start_id, end_id)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(etree.tostring(trimmed, pretty_print=True))
        print(f"Trimmed path XML written to {output_path}")

    return trimmed
