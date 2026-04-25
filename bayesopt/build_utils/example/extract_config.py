import xml.etree.ElementTree as ET
from lxml import etree
import yaml
import os
from pyoptr import DataDat
import argparse
#this python script converts .xml to config file for BO

#input xml file
def build_optr(name, apertures, device_types, measurement_device):

    build_dir = f'/home/{os.getlogin()}/repos/bayesopt/build_utils/{name}'
    optr_dir = f'/home/{os.getlogin()}/repos/bayesopt/transoptr/{name}'
    outputfile = f'/home/{os.getlogin()}/repos/bayesopt/config/sim/{name}.yaml'
    tree = ET.parse(os.path.join(optr_dir, f'{name}_path.xml'))
    datadat = DataDat(file=os.path.join(optr_dir, 'data.dat'))

    #give the headers of the output config file
    sim_type="beamline"   #beamline or experiment
    random_seed = "Null"
    offset = 1

    tuning_elements = device_types
    misalignment_elements=['quad', 'src_mis']

    # tune_config = os.path.join(build_dir, 'tune_config.xml')
    
    # # Parse the XML file
    # tree = etree.parse(tune_config)
    root = tree.getroot()

    # Extract the value of the 'path' attribute

    #if sim_type=beamline only define acc_pth and optr_pth
    acc_pth= root.get("path") #"ios-mws-hebt2-dragon"
    optr_pth=f"/home/USERNAME/repos/bayesopt/transoptr/{name}"

    #define misalignments bounds for steerer, quads, slits and dipoles
    mis_quad=[0.,0.05]
    mis_slit=[0.,1.]
    mis_dipole_pos=[0.,0.05]
    mis_dipole_ang=[0.,0.01]
    source_mis = {"src_mis": {
        'x1':{'min':0.0, 'max':0.1},
        'y1':{'min':0.0, 'max':0.1},
        'x2':{'min':0.0, 'max':0.01},
        'y2':{'min':0.0, 'max':0.01}
        }}
    slit_widths = {
        'x_lower': 0.5,
        'x_upper':0.5,
        'y_lower':1.0,
        'y_upper':1.0
    }

    #manually define bound for steerer bound in simulation, limit to 2mrad
    steer_lower_bound = -0.002 
    steer_upper_bound = 0.002

    beamline_wall_width = 2.54 # radius in cm
    aper=beamline_wall_width

    # Get the root element
    root = tree.getroot()
    data={}

    first_elem = root.findall('element')[0]
    starting_length = float(first_elem.attrib['s'].split('*')[0])
    loc_unit = first_elem.attrib['s'].split('*')[1]
    if(loc_unit=="mm"):
        starting_length=starting_length/10

    # Iterate over the child elements of the root element and print their tags and attributes
    end_of_last_seq = 0
    prev_loc = 0 
    for parent in root.findall('element'):
        
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

        for child in parent.findall('optr'):
            if 'aperR' in child.attrib:
                aper = float(child.attrib['aperR'].split('*')[0])
                aper_unit = child.attrib['aperR'].split('*')[1]
                if(aper_unit=="mm"):
                    aper=aper/10
            else:
                aper=beamline_wall_width

        for child in parent.findall('epics'):
            for grandchild in child.findall('setpoint'):
                if 'min' in grandchild.attrib:
                    if(ele_type=="steerer"):
                        pv_min = steer_lower_bound
                        pv_max = steer_upper_bound
                    else:
                        pv_min = float(grandchild.attrib['min'])
                        pv_max = float(grandchild.attrib['max'])

                        # Add new data to the existing data
                        if ele_type == 'quad':
                            id = id + ":CUR"
                        if ele_type in ['steerer', 'dipole']:
                            pv_min = -0.02
                            pv_max = 0.02
                    
                    try:
                        start_val = datadat.get_element(name=id)['value']
                    except:
                        print(id, 'is not defined in data.dat')
                        start_val=(pv_max-pv_min)/2
                    
                    if id in apertures:
                        aper = apertures[id]
                    else:
                        aper = beamline_wall_width
                        
                    new_data = {id:{'type' : ele_type,
                    'lower_bound': pv_min,
                    'upper_bound': pv_max,
                    'loc': loc,
                    'starting_mean': start_val,
                    'starting_var': (pv_max-pv_min)/10,
                    'aperture': aper,
                    'plate_length': length}}
                else:
                    new_data = {id:{
                    'type' : ele_type,
                    'loc': loc}
                    }
                    
                data.update(new_data)
    
    
    # Loop over the dictionary items and group them based on the 'type' attribute
    grouped_data = { }

    for item_key, item_value in data.items():
        item_type = item_value['type']
        #item_value.pop('type')  # Remove the 'type' key from item_value
        if item_type in grouped_data:
            grouped_data[item_type][item_key] = item_value
            if(item_type=="quad"):
                grouped_data[item_type][item_key]["misalignments_min"]=mis_quad[0]
                grouped_data[item_type][item_key]["misalignments_max"]=mis_quad[1]
            elif(item_type=="dipole"):
                grouped_data[item_type][item_key]["pos_mis_min"]=mis_dipole_pos[0]
                grouped_data[item_type][item_key]["pos_mis_max"]=mis_dipole_pos[1]
                grouped_data[item_type][item_key]["ang_mis_min"]=mis_dipole_ang[0]
                grouped_data[item_type][item_key]["ang_mis_max"]=mis_dipole_ang[1]
            elif(item_type=="slit"):
                grouped_data[item_type][item_key]["misalignments_min"]=mis_slit[0]
                grouped_data[item_type][item_key]["misalignments_max"]=mis_slit[1]
                
                grouped_data[item_type][item_key]["x_lower"]=slit_widths['x_lower']
                grouped_data[item_type][item_key]["y_lower"]=slit_widths['y_lower']
                grouped_data[item_type][item_key]["x_upper"]=slit_widths['x_upper']
                grouped_data[item_type][item_key]["y_upper"]=slit_widths['y_upper']
        else:
            grouped_data[item_type] = {item_key: item_value}
            # if(item_type=="steerer"):
            #     grouped_data[item_type][item_key]["misalignments_min"]=mis_steer[0]
            #     grouped_data[item_type][item_key]["misalignments_max"]=mis_steer[1]
            if(item_type=="quad"):
                grouped_data[item_type][item_key]["misalignments_min"]=mis_quad[0]
                grouped_data[item_type][item_key]["misalignments_max"]=mis_quad[1]
            elif(item_type=="dipole"):
                grouped_data[item_type][item_key]["pos_mis_min"]=mis_dipole_pos[0]
                grouped_data[item_type][item_key]["pos_mis_max"]=mis_dipole_pos[1]
                grouped_data[item_type][item_key]["ang_mis_min"]=mis_dipole_ang[0]
                grouped_data[item_type][item_key]["ang_mis_max"]=mis_dipole_ang[1]
            elif(item_type=="slit"):
                grouped_data[item_type][item_key]["misalignments_min"]=mis_slit[0]
                grouped_data[item_type][item_key]["misalignments_max"]=mis_slit[1]
                grouped_data[item_type][item_key]["x_lower"]=slit_widths['x_lower']
                grouped_data[item_type][item_key]["y_lower"]=slit_widths['y_lower']
                grouped_data[item_type][item_key]["x_upper"]=slit_widths['x_upper']
                grouped_data[item_type][item_key]["y_upper"]=slit_widths['y_upper']
        
    grouped_data = dict(sorted(grouped_data.items()))

    element_data = {'elements': grouped_data} 
    #misalignment_data = {'misalignments': grouped2_data}

    with open(outputfile, 'w') as file:
        print('name: ', name, file=file) 
        print('type: ',sim_type, file=file)
        print('random_seed: ',random_seed, file=file)

        if(sim_type=="beamline"):
            print('\nacc_path: ', acc_pth, file=file)

            print(f'\ndatadat_path: {optr_pth}/data.dat',file=file)
            print(f'optr_dir: {optr_pth}',file=file)
        
            print(f'\nbeamline_wall_width: {beamline_wall_width}', file=file)
            print(f'offset: {offset}', file=file)
            print(f'tuning_elements: {str(tuning_elements)}', file=file)
            print(f'misalignment_elements: {str(misalignment_elements)}', file=file)
            print(f'measurement_device: {measurement_device}', file=file)
            
            yaml.dump(source_mis, file, default_flow_style=False, sort_keys=False)

        yaml.dump(element_data, file, default_flow_style=False, sort_keys=False)
        #yaml.dump(misalignment_data, file, default_flow_style=False, sort_keys=False)
        print('config yaml generated successfully!')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--name')
  parser.add_argument('--measurement_device', required=True)
  parser.add_argument('--device_types', nargs='+', required=True)
  args = parser.parse_args()

  directory = os.path.dirname(os.path.abspath(__file__))
  name = args.name  
  measurement_device = args.measurement_device  
  device_types = args.device_types  

  apertures = {
    'SAC1:DTL1': 0.69977,
    'ISAC1:DTL2': 0.499999,
    'ISAC1:DTL3': 0.8001,
    'ISAC1:DTL4': 0.8001,
    'ISAC1:DTL5': 0.8001,
    'ISAC1:MEBT': 1.0
    }
  
  build_optr(name, apertures, device_types, measurement_device)    
    