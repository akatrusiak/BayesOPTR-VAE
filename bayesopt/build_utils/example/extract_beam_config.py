import xml.etree.ElementTree as ET
from lxml import etree
from xml.etree.ElementTree import ElementTree, fromstring
import yaml
import os
from pyoptr import DataDat
import argparse
#this python script convert .xml to config file for BO

def load_xml_with_include(path,ET=False):
    parser = etree.XMLParser(load_dtd=True, resolve_entities=True)
    tree = etree.parse(path, parser)
    tree.xinclude()
    root = tree.getroot()
    
    if ET:
        xml_bytes = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        element_tree = ElementTree(fromstring(xml_bytes))
        root = element_tree.getroot()
    
    return root

def parse_from_acc(facility, path,start,end):#end element is always a faraday cup
    acc_dir = os.environ.get('ACCDIR')
    path_dir = os.path.join(f'{facility}/path/',path+'.xml')
    xml_file_path = os.path.join(acc_dir,path_dir)
    root = load_xml_with_include(xml_file_path,ET=False) #load the lxml tree so we can edit (was getting weird errors trying to trim the ET trees)
    startfound = False
    endfound = False
    # Traverse each child in the root
    for sequence in root:
        if endfound:
            root.remove(sequence)
            next
        # Find all 'element' tags within the sequence
        for element in sequence.findall('element'):
            if element.attrib['id'] == start:
                startfound = True  # Set the flag when the desired element is found
            elif  element.attrib['id']==end:
                if not startfound:
                    print(f"Error: End element is before start element, or '{start}'  does not exits in path")
                    exit()
                else:
                    endfound = True
            elif startfound and not endfound:
                pass            
            else:
                sequence.remove(element)
                # Break the inner loop
        if not startfound:
            root.remove(sequence)
    if not endfound:
        print(f"'{end}' not found in path. ")
        exit()
    return ElementTree(root)


#input xml file
def build_optr(name,starting_element, measurement_device, facility, path ,device_types = ['steerer'],jaya_server='beta'):

    # build_dir = f'/home/{os.getlogin()}/repos/bayesopt/build_utils/{name}'
    # optr_dir = f'/home/{os.getlogin()}/repos/bayesopt/transoptr/{name}'
    
    
    outputfile = f'/home/{os.getlogin()}/repos/bayesopt/config/beam/{name}.yaml'
    
    # tree = ET.parse(os.path.join(optr_dir, f'{name}_path.xml')) 
    tree = parse_from_acc(facility, path,starting_element,measurement_device)  


    # Get the root element
    root = tree.getroot()
    data={}
    first_seq = root.findall('seq')[0]
    first_elem = first_seq.findall('element')[0]
    starting_length = float(first_elem.attrib['s'].split('*')[0])
    loc_unit = first_elem.attrib['s'].split('*')[1]
    if(loc_unit=="mm"):
        starting_length=starting_length/10

    # Iterate over the child elements of the root element and print their tags and attributes
    end_of_last_seq = 0
    prev_loc = 0 
    for sequence in root.findall('seq'):
        for parent in sequence.findall('element'):
            
            id = parent.attrib['id']
            
            ele_type = parent.attrib['type']
            if(ele_type =='ecb' or ele_type == 'mcb' ):
                ele_type = 'steerer'
            elif(ele_type =='col' or ele_type =='slit'):
                ele_type = 'slit'
            elif(ele_type =='mq' or ele_type =='eq'):
                ele_type = 'quad'
            elif(ele_type =='mb'):
                ele_type = 'dipole'
            elif(ele_type =='fc'):
                ele_type = 'fc'
                
            
            # Split the input string on the '*' character and take the first element
            loc = float(parent.attrib['s'].split('*')[0])
            
            loc_unit = parent.attrib['s'].split('*')[1]
            if(loc_unit=="mm"):
                loc=loc/10
            elif(loc_unit=="cm"):
                loc=loc
            
            if loc < prev_loc:
                end_of_last_seq += prev_loc
            
            prev_loc = loc
            loc = loc + end_of_last_seq -starting_length
            
            length = float(parent.attrib['l'].split('*')[0])
            if '*' in parent.attrib['l'].split('*'):
                length_unit = parent.attrib['l'].split('*')[1]
        
            for child in parent.findall('layout'):
                x = float(child.attrib['x'].split('*')[0])
                x_unit = child.attrib['x'].split('*')[1]
                if(x_unit=="mm"):
                    x=x/10

                y = float(child.attrib['y'].split('*')[0])
                y_unit = child.attrib['y'].split('*')[1]
                if(y_unit=="mm"):
                    y=y/10

                z = float(child.attrib['z'].split('*')[0])
                z_unit = child.attrib['z'].split('*')[1]
                if(z_unit=="mm"):
                    z=z/10

            new_data = {}


            for child in parent.findall('epics'):
                
                
                for grandchild in child.findall('setpoint'):
                    
                    if 'min' in grandchild.attrib:
                        
                        pv_min = float(grandchild.attrib['min'])
                        pv_max = float(grandchild.attrib['max'])

                        # Add new data to the existing data
                        if ele_type == 'quad':
                            id = id + ":CUR"
                        if ele_type in ['dipole']:
                            pv_min = -0.02
                            pv_max = 0.02
                        
                        if pv_min * pv_max < 0:
                            set_zero_crossing = True
                        else:
                            set_zero_crossing = False
                        
                        if grandchild.attrib['pv'] not in data:
                            new_data = {grandchild.attrib['pv']:{
                            'type' : ele_type,
                            'loc': loc,
                            'lower_bound': pv_min,
                            'upper_bound': pv_max,
                            'starting_mean': (pv_max+pv_min)/2,
                            'starting_var': (pv_max-pv_min)/10,
                            'max_incremental_change': pv_max - pv_min,
                            'set_zero_crossing': set_zero_crossing,
                            'commonto': []
                            }}

                        if ele_type == 'steerer':
                            try:
                                if grandchild.attrib['type'] == 'common':
                                    if grandchild.attrib['pv'] not in data:
                                        # print('A',grandchild.attrib['pv'], id)
                                        steerer_pv = id + ':' + grandchild.attrib['pv'].split(':')[-1]
                                        new_data[grandchild.attrib['pv']]['commonto'].append(steerer_pv)
                                    else:
                                        steerer_pv = id + ':' + grandchild.attrib['pv'].split(':')[-1]
                                        data[grandchild.attrib['pv']]['commonto'].append(steerer_pv)
                            except KeyError:
                                pass
                        try:
                            theta = parent.find('optr').attrib['theta']
                            new_data[grandchild.attrib['pv']]['theta'] = theta
                        except Exception as e:
                            pass
                

                        data.update(new_data)
                    ##FIND FC
                        

                for grandchild in child.findall('readback'):
                
                    if 'min' not in grandchild.attrib:
                        
                        if 'unit' in grandchild.attrib:
                            scale_factor = 0
                            if grandchild.attrib['unit'] == 'nA':
                                scale_factor = 1000

                            if grandchild.attrib['unit'] == 'A':
                                scale_factor = 1000000000000
            
                            if grandchild.attrib['unit'] == 'pA':
                                scale_factor = 1
                            if scale_factor:
                                new_data = {id:{
                                'type' : ele_type,
                                'loc': loc,
                                'pv': grandchild.attrib['pv'],
                                'measure_points': 20,
                                'trim_ratio': 0.2,
                                'ptime': 0.01,
                                'scale_factor': scale_factor
                                }}
                            
                            
                        
                            data.update(new_data)
    #go through full acc path xml and look for all steerers that share a common with steerers in data already
    acc_dir = os.environ.get('ACCDIR')
    path_dir = os.path.join(f'{facility}/path/',path+'.xml')
    xml_file_path = os.path.join(acc_dir,path_dir)
    root = load_xml_with_include(xml_file_path,ET=False)
    in_range = False
    exit=False
    for sequence in root.findall('seq'):
        for parent in sequence.findall('element'):
            if parent.attrib.get('type')=='ecb':
                id = parent.attrib['id']
                # print(f"element: {id}")
                for child in parent.findall('epics'):
                    for grandchild in child.findall('setpoint'):
                        # print('first loop')
                        if 'CCB' not in grandchild.attrib.get('pv'):
                            steerer_pv = grandchild.attrib.get('pv')
                            # print(f"\tsteerer: {steerer_pv}")
                        elif 'CCB' in grandchild.attrib.get('pv'):
                            common_pv = grandchild.attrib.get('pv')
                            # print(f"\tcommon: {common_pv}")
                    for grandchild in child.findall('setpoint'):
                        # print('second loop')
                        if 'CCB' not in grandchild.attrib.get('pv'):
                            if steerer_pv in data:
                                in_range=True
                                # print(f"{steerer_pv} in data")
                            elif in_range==True:
                                # print(f"{steerer_pv} not in data")
                                in_range=False
                                exit=True #comment out this line and the next if you want to include steerers attached to commons after the tuning section as well
                                break
                        elif 'CCB' in grandchild.attrib.get('pv'):
                            if steerer_pv not in data:
                                # print(f"{steerer_pv} not in data")
                                if common_pv in data:
                                    # print(f"{common_pv} in data")
                                    steerer_pv = id + ':' + grandchild.attrib['pv'].split(':')[-1]
                                    data[grandchild.attrib['pv']]['commonto'].append(steerer_pv)#add the steerer pv to the list of steerers common to the common plate
                                    # print(f"{steerer_pv} added to {common_pv}")
                    if exit:
                        break
            if exit:
                break
        if exit:
            break
   
    
    # Loop over the dictionary items and group them based on the 'type' attribute
    grouped_steering_elements = {}
    grouped_measurement_device = {}
    grouped_commons = {}
    grouped_commons = {}

    # Sort the steering elements by their "loc" values
    


    for item_key, item_value in data.items():
        item_type = item_value['type']
        # print(f"item_key: {item_key}")
        if 'commonto' in item_value and item_value['commonto'] == []:
            item_value.pop('commonto')
        if item_type in device_types and 'CCB' not in item_key:
            grouped_steering_elements[item_key] = item_value
        
        if item_type in device_types and 'CCB' in item_key:
            grouped_commons[item_key] = item_value
        if item_key == measurement_device:
            grouped_measurement_device[item_key] = item_value
        
    for common_key, common_value in grouped_commons.items():
        for element_key in common_value['commonto']:
            if element_key in grouped_steering_elements:
                # Add or update the 'commonis' attribute for this steering element
                grouped_steering_elements[element_key]['commonis'] = common_key

        # TODO: add part here where it adds the required extra steerers to commonto that aren't in the reduced tree

    grouped_steering_elements = dict(sorted(grouped_steering_elements.items(),key=lambda x: x[1]["loc"]))
    for element in grouped_steering_elements.values():
        element.pop("loc", None)
    
    element_data = {'steering_elements': grouped_steering_elements, 'measurement_device' : grouped_measurement_device[measurement_device], 'common_elements' : grouped_commons} 
    if (element_data.get('measurement_device').get('scale_factor')) <= 0:
        print(f'Double check FC units, scale_factor: {element_data.get("measurement_device").get("scale_factor")}')

    with open(outputfile, 'w') as file:
        print('jaya_url: \"https://vpn.'+jaya_server+'.hla.triumf.ca/jaya/\"', file=file)
        print('initial_transmission: #write value initial cup reading in pA', file= file)
        print('beam_energy: #beam energy in keV', file= file)
        print('beam_mass: # beam mass in amu (only needed for magnetic steering)', file= file)
        print('beam_chargestate: # beam chargestate (only needed for magnetic steering)', file= file)
        yaml.dump(element_data, file, default_flow_style=False, sort_keys=False)
        print('#scale factor is a fudge factor to make all FC measurements be in pA', file = file)
        print('config yaml generated successfully!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--name')
  parser.add_argument('--starting_element', required=True)
  parser.add_argument('--measurement_device', required=True)
  parser.add_argument('--facility',required=True)
  parser.add_argument('--path',required=True)
  parser.add_argument('--device_types', nargs='+', required=True)
  parser.add_argument('--jaya_server',required=True)
  args = parser.parse_args()

  directory = os.path.dirname(os.path.abspath(__file__))
  name = args.name  
  starting_element = args.starting_element
  measurement_device = args.measurement_device    
  facility = args.facility
  path = args.path
  device_types = args.device_types  
  jaya_server = args.jaya_server

  build_optr(name, starting_element ,measurement_device, facility, path, device_types, jaya_server)
