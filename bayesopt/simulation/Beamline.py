from contextlib import redirect_stdout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from copy import copy
import random
import numpy as np
import torch
import cv2
import os

# import pyoptr library, written by Paul Jung, to process data.dat file
# download pyoptr repo and add to PYTHONPATH
import pyoptr
from pyoptr import DataDat
from omegaconf import OmegaConf
from pyoptr.postprocess import get_column

import logging, sys
logging.disable(sys.maxsize)

from scipy import special

import time
from os import mkdir
from accpy import acxml
from contextlib import redirect_stdout

''' updating beamline class to deal with the newest version of transoptr'''
class Beamline():
    def __init__(self, config):
        self.config = config
        sim_config = config.sim 
        # seeds the simulation
        self.seed = sim_config.random_seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        self.elements = sim_config.elements
        # self.zeros_offset=3.7875   #for mebt only
        self.zeros_offset=sim_config.offset

        self.generated_misalignments = False # will be set to True after first set of misalignments has been generated, we only want this to happen once since beamline retains its state of misalignments throughout the entire tuning session
        
        self.steering_elements = self.elements.steerer
        self.quads = self.elements.quad
        self.dipoles = getattr(self.elements, 'dipole', None)
        self.slits = self.elements.slit

        self.measurement_device = sim_config.measurement_device
        self.profile_device = getattr(sim_config,'profile_device', None) # not used right now

        self.tuning_elements = {}
        self._get_tuning_elements()
        
        self.misalignment_dict = {}
        self._generate_misalignments()

        self.action_limits_low  = [action['lower_bound'] for action in self.tuning_elements.values()]
        self.action_limits_high = [action['upper_bound'] for action in self.tuning_elements.values()]


        if 'slits' in sim_config.elements:
            print('slits in config file')
            self.slits = sim_config.elements.slit
            
        self.wall_width=sim_config.beamline_wall_width

        self.last_actions = {}
        for name in self.tuning_elements:
            self.last_actions[name] = 0.0

        # self.optr_data = DataDat(file=sim_config.datadat_path.replace("USERNAME", os.getlogin()))
        self.optr_data = DataDat(file=sim_config.datadat_path)
        
        # generate baseline measurements (no misalignment)
        self.s = None
        self.x_cm = None
        self.y_cm = None
        self.x_ev = None
        self.y_ev = None
        
        self.x_cm_baseline = None
        self.y_cm_baseline = None
        self.x_ev_baseline = None
        self.y_ev_baseline = None
        
        self._generate_baseline_measurements()
        
        # create dictionary for the location of all the quad skimmers
        self._generate_skimmers_dict()
        self._generate_slits_dict()

        self.set_misalignments()
        self._run_simulation()

        self._calculate_transmission_integral()
        
        public_methods = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("_")]
        formatted_print = "\n- "+"\n- ".join(public_methods)
        print(f'\nEnd of Beamline init...\nPlease interface only via public methods {formatted_print}')
        
    def _get_tuning_elements(self):
        for elem_type in self.config.sim.tuning_elements:
            if elem_type in self.config.sim.elements:
                self.tuning_elements.update(self.config.sim.elements[elem_type])
                                 
    def _generate_skimmers_dict(self):
        '''
        Ask Olivier for further clarification on how skimmer plates are calcuated
        '''
        self.skimmers_dict = {}

        # TODO: confirm if we want to have this
        # for quad_name, val in self.elements.quad.items():
        #     element = acxml.get_element_by_id(quad_name, path=self.config.sim.acc_path, json=True)
        #     if element['optr']['@aperr'].split('*')[1] == 'mm':
        #         skimmer_diameter = float(element['optr']['@aperr'].split('*')[0])/10
        #     if element['optr']['@aperr'].split('*')[1] == 'cm':
        #         skimmer_diameter = float(element['optr']['@aperr'].split('*')[0])
        #     left_skimmer_loc = val.loc-val.quad_length/2-val.skimmer_distance_before_quad
        #     right_skimmer_loc = val.loc+val.quad_length/2+val.skimmer_distance_after_quad
        #     self.skimmers_dict[left_skimmer_loc] = skimmer_diameter/2
        #     self.skimmers_dict[right_skimmer_loc] = skimmer_diameter/2
            
        # for steerer_name, val in self.tuning_elements.items():
        #     # print(steerer_name)
        #     element = acxml.get_element_by_id(steerer_name, path=self.config.sim.acc_path, json=True)
        #     if element['@type'] == 'ecb':
        #         left_skimmer_loc = val.loc-val.plate_length/2-val.skimmer_distance_before_steerer
        #         right_skimmer_loc = val.loc+val.plate_length/2+val.skimmer_distance_after_steerer
        #         self.skimmers_dict[left_skimmer_loc] = val.skimmer_diameter/2
        #         self.skimmers_dict[right_skimmer_loc] = val.skimmer_diameter/2
                
    def _generate_slits_dict(self):
        '''
        uses all the skimmers, wall width, and any specified slits
        '''
        if 'slits' in self.elements.slit:
            self.x_upper_slits_dict = {**self.skimmers_dict, **{slit_dict['loc']:slit_dict['x_upper'] for slit_dict in self.slits.values()}}
            self.y_upper_slits_dict = {**self.skimmers_dict, **{slit_dict['loc']:slit_dict['y_upper'] for slit_dict in self.slits.values()}}

            self.x_lower_slits_dict = {**self.skimmers_dict, **{slit_dict['loc']:slit_dict['x_lower'] for slit_dict in self.slits.values()}}
            self.y_lower_slits_dict = {**self.skimmers_dict, **{slit_dict['loc']:slit_dict['y_lower'] for slit_dict in self.slits.values()}}

        else:
            self.x_upper_slits_dict = self.skimmers_dict
            self.y_upper_slits_dict = self.skimmers_dict

            self.x_lower_slits_dict = self.skimmers_dict
            self.y_lower_slits_dict = self.skimmers_dict
            

    def _generate_misalignments(self):
        '''
        Generates a set of misalignments from random draws, then applies any
        overrides from config.sim.misalignment_override on top.
        Saves the result to self.misalignment_dict and writes Misalignments.yaml.
        '''
        if self.generated_misalignments:
            print('!!!!Warning: calling beamline._generate_misalignments() more than '
                'once will result in different beamline misalignments between runs')

        # --- random draws ---
        self._draw_src_misalignments()

        for elem in self.elements:
            if elem not in self.config.sim.misalignment_elements:
                continue
            if elem == 'quad':
                self._draw_quad_misalignments()
            elif elem == 'dipole':
                self._draw_dipole_misalignments()
            elif elem == 'src_mis':
                # already handled by _draw_src_misalignments
                continue
            else:
                raise ValueError(f'!!! warning: {elem} type not implemented for misalignments')

        # --- apply overrides last so they take precedence ---
        self._apply_misalignment_overrides()

        self.generated_misalignments = True
        OmegaConf.save(config=self.misalignment_dict, f="Misalignments.yaml")


    def _signed_uniform(self, lo, hi):
        """Uniform draw in [lo, hi] with random sign."""
        sign = 1 if np.random.rand() > 0.5 else -1
        return random.uniform(lo, hi) * sign


    def _draw_src_misalignments(self):
        for name in self.config.sim.src_mis:
            tag = "MISALIGN" + name.upper()
            lo = self.config.sim.src_mis[name].min
            hi = self.config.sim.src_mis[name].max
            self.misalignment_dict[tag] = self._signed_uniform(lo, hi)


    def _draw_quad_misalignments(self):
        for quad_name, quad in self.elements.quad.items():
            for mis_name in (":MISALIGNX", ":MISALIGNY"):
                if ":CUR" in quad_name:
                    quad_name = quad_name.replace(":CUR", "")
                self.misalignment_dict[quad_name + mis_name] = self._signed_uniform(
                    quad.misalignments_min, quad.misalignments_max
                )


    def _draw_dipole_misalignments(self):
        for dipole_name, dipole in self.elements.dipole.items():
            for mis_name in (":MISALIGNX", ":MISALIGNY", ":MISALIGNZ"):
                self.misalignment_dict[dipole_name + mis_name] = self._signed_uniform(
                    dipole.pos_mis_min, dipole.pos_mis_max
                )
            for mis_name in (":MBX", ":MBY"):
                self.misalignment_dict[dipole_name + mis_name] = self._signed_uniform(
                    dipole.ang_mis_min, dipole.ang_mis_max
                )


    def _apply_misalignment_overrides(self):
        overrides = self.config.get("misalignments", None)
        if not overrides:
            return
        
        # If overrides is a string, treat it as a file path and load it
        if isinstance(overrides, str):
            overrides = OmegaConf.load(overrides)
        
        if not isinstance(overrides, dict):
            return
        
        for key, value in overrides.items():
            if key not in self.misalignment_dict:
                print(f"Warning: override '{key}' has no matching random-drawn entry; "
                    f"adding anyway")
            self.misalignment_dict[key] = float(value)
        print(f"Applied {len(overrides)} misalignment override(s)")
    
    
    def _generate_baseline_measurements(self):
        '''
        resets self.optr_data to baseline, baseline measurement refer to the beam without any steering and with no misalignments
        '''
        self.clear_misalignments()
        self._run_simulation(set_baseline=True)

    def _run_simulation(self, set_baseline=False):
        ''' 
        runs the transoptr simulation to
        measure the positions, x-centroids, y-centroids, x-envelopes, and y-envelopes 
        updates arrays in self.x_cm, self.y_cm, self.x_ev, self.y_ev
        if set_baseline flag is true, uses current simulation to set baselines for cm and ev arrays 
        '''

        with redirect_stdout(open(os.devnull, 'w')):
            # optr_outputs = pyoptr.run(self.config.sim.optr_dir.replace("USERNAME", os.getlogin()), input_class=self.optr_data) # old method was using fixed directory strings. 
            optr_outputs = pyoptr.run(self.config.sim.optr_dir, input_class=self.optr_data)

        s, sigma_x, sigma_y, cm_x, cm_y = get_column(optr_outputs, 's', 'x-envelope', 'y-envelope', 'x-centroid', 'y-centroid')
        self.s = copy(s)
        self.x_cm = copy(cm_x)
        self.y_cm = copy(cm_y)
        self.x_ev = copy(sigma_x)
        self.y_ev = copy(sigma_y)

        if set_baseline:
            self.x_cm_baseline, self.y_cm_baseline = copy(cm_x), copy(cm_y)
            self.x_ev_baseline, self.y_ev_baseline = copy(sigma_x), copy(sigma_y)
            
    def _calculate_transmission_integral(self):
        x_losses_erf, x_propagated_transmissions = self._get_erf_losses(coord='x')
        y_losses_erf, y_propagated_transmissions = self._get_erf_losses(coord='y')

        self.x_propagated_transmissions = x_propagated_transmissions
        self.y_propagated_transmissions = y_propagated_transmissions
        self.propagated_transmissions = x_propagated_transmissions*y_propagated_transmissions

    def _generate_slits_array(self):
        '''
        generates an array containing the slit widths for one dimension
        slits_dict: dictionary with (pos: slit_width) pairings.
              pos is the length (cm) along the beamline where the slit is
              slit_width is the half the width of the slit (length from center to one side)
        '''
        slits_array = np.full_like(self.s, self.wall_width)

        for name, slit in self.slits.items():
            loc = slit['loc']
            index = np.argmin(abs(self.s-loc+self.zeros_offset))
            if 'aperture' in slit:
                if(slit['aperture']<self.wall_width):
                    slits_array[index] = slit['aperture']

        for name, quad in self.quads.items():
            loc = quad['loc']
            index = np.argmin(abs(self.s-loc+self.zeros_offset))
            lower_loc=quad['loc']-quad['plate_length']/2
            upper_loc=quad['loc']+quad['plate_length']/2
            low_index = np.argmin(abs(self.s-lower_loc+self.zeros_offset))
            upper_index = np.argmin(abs(self.s-upper_loc+self.zeros_offset))
            if 'aperture' in quad:
                if(quad['aperture']<self.wall_width):
                    slits_array[index] = quad['aperture']
                    
        if 'rf' in self.elements:
            for name, rf in self.elements.rf.items():
                loc = rf['loc']
                index = np.argmin(abs(self.s-loc+self.zeros_offset))
                lower_loc=rf['loc']-rf['plate_length']/2
                upper_loc=rf['loc']+rf['plate_length']/2
                low_index = np.argmin(abs(self.s-lower_loc+self.zeros_offset))
                upper_index = np.argmin(abs(self.s-upper_loc+self.zeros_offset))
                if 'aperture' in rf:
                    if(rf['aperture']<self.wall_width):
                        #slits_array[index] = rf['aperture']
                        for i in range(low_index,upper_index):
                            slits_array[i] = rf['aperture']
    
        return slits_array
    
    def _generate_simulation(self, show_steerer_pos=False, show_transmission=False):
        fig = plt.figure(1)
        fig.set_size_inches(16,5)
        plt.clf()
        plt.plot(self.s, self.x_cm, color='b',label = 'x plane')
        plt.plot(self.s, self.x_cm + self.x_ev, linestyle='dashed', color='b')
        plt.plot(self.s , self.x_cm - self.x_ev, linestyle='dashed', color ='b')
        plt.fill_between(self.s, self.x_cm- self.x_ev, self.x_cm +self.x_ev, color='b', alpha=0.2  )
        plt.plot(self.s, self.y_cm, color='r',label = 'y plane')
        plt.plot(self.s, self.y_cm + self.y_ev, linestyle='dashed', color='r')
        plt.plot(self.s , self.y_cm - self.y_ev, linestyle='dashed', color ='r')
        plt.fill_between(self.s, self.y_cm- self.y_ev, self.y_cm +self.y_ev, color='r', alpha=0.2  )

        self.x_slits_array = self._generate_slits_array()
        self.y_slits_array = self._generate_slits_array()
        
        plt.step(self.s, self.x_slits_array, where ='mid', color='b', alpha=0.5, label='slit radius x')
        plt.step(self.s, -self.y_slits_array, where='mid',color='r', alpha=0.5, label='slit radius y')

        plt.fill_between(self.s, self.y_cm-self.y_ev, self.y_cm + self.y_ev, color='r', alpha=0.2  )
        plt.axhline(0, linestyle='solid', color='gray', alpha=0.7)
        if show_steerer_pos:
            prev_loc = 0
            SIZE = 10
            for name,element in self.tuning_elements.items():
                if name in self.quads:
                    line_color='purple'
                else:
                    line_color='violet'
                    
                plt.axvline(element['loc']-self.zeros_offset, linestyle='dashed', color=line_color)
                loc = element['loc']-self.zeros_offset
                if abs(prev_loc-loc) < SIZE:
                    loc += SIZE
                plt.text(loc-SIZE, 1.2, name, rotation=70, size=SIZE, color=line_color)
                prev_loc = loc
            
            prev_loc = 0
            if 'rf' in self.elements:
                for name, rf in self.elements.rf.items():
                    plt.axvline(rf['loc']-self.zeros_offset, linewidth=0.5, color='grey', alpha=0.3)
                    loc = rf['loc']-self.zeros_offset
                    if abs(prev_loc-loc) < SIZE:
                        loc += SIZE
                    plt.text(loc-SIZE, self.wall_width+0.02, name, rotation=0, size=5, color='black')
                    prev_loc = loc
      
            for name, fc in self.elements.fc.items():
                loc = fc['loc']-self.zeros_offset
                plt.axvline(loc, linestyle='solid', color='orange', alpha=0.6, linewidth=3)
                plt.text(loc-SIZE/2, -1.9, name, rotation=90, size=SIZE, color='black')

        if show_transmission:
            plt.axhline(1, linestyle='dashed', color='black', linewidth=1)
            plt.step(self.s, self.propagated_transmissions, color="green", where='post',label="transmissions")
            plt.fill_between(self.s, 0, self.propagated_transmissions, color="green", alpha=0.4)
            
        plotMaximum = 0
   
        loc = self.elements.fc[self.measurement_device].loc-self.zeros_offset
        if (loc > plotMaximum):
            plotMaximum = loc

        plt.xlim(-10, plotMaximum+30)
        plt.ylim(-self.wall_width,self.wall_width)
        plt.xlabel("Distance Along Beam Axis (cm)")
        plt.ylabel("Distance in Radial Direction (cm)")
        plt.legend(loc = 'lower left')
        plt.tight_layout()
        # fig.canvas.draw()
        # img = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        return img
     
    def _get_erf_losses(self, coord):
        # calculate losses of beam transmission using error function 1/2*{Erf[(w-mu)/(sqrt2*sigma)] +Erf[(w+mu)/(sqrt2*sigma)]}
        # measures the propagated losses through the beamline by assuming uncorrelated gaussian distributions in x and y
        # coord = the dimension (x or y) to calculate the losses returns
        # losses: array holding the percentage of beam loss at each position along the beamline
        # propagated_transmissions: array showing the estimated propagated transmissions based on multiplying next element in array by previous transmission

        #initialization of lower_slits_array
        x_slits_array=self._generate_slits_array()
        y_slits_array=self._generate_slits_array()

        if coord == 'x' or coord == 'X':
            sigma_array = copy(self.x_ev)/2  #obtaining 1 sigma
            mu_array = copy(self.x_cm)
            cm_plot = copy(self.x_cm)
            ev_plot = copy(self.x_ev)
            slits_array=copy(x_slits_array)
            title='x'
            color='b'

        elif coord == 'y' or coord == 'Y':
            sigma_array = copy(self.y_ev)/2  #obtaining 1 sigma
            mu_array = copy(self.y_cm)
            cm_plot = copy(self.y_cm)
            ev_plot = copy(self.y_ev)
            slits_array=copy(y_slits_array)
            title='y'
            color='r'

        else:
            raise ValueError('coord', coord, 'is not defined. Should be one of x (X) or y (Y)')

        #initializing the transmission array to be zero
        transmissions = np.zeros(len(self.s))

        for i in range(len(transmissions)):
            if(sigma_array[i]==0):
                sigma_array[i]=0.00001
            else:
                transmissions[i] = 0.5 * (special.erf((slits_array[i]-mu_array[i])/(np.sqrt(2)*sigma_array[i]))
                                     - special.erf((-slits_array[i]-mu_array[i])/(np.sqrt(2)*sigma_array[i])))
        losses = np.ones(len(self.s)) - transmissions
        propagated_transmissions = copy(transmissions)
        for i in range(1, len(propagated_transmissions)):
            propagated_transmissions[i] *= propagated_transmissions[i-1]
        return losses, propagated_transmissions
    
    def clear_misalignments(self):
        for name in self.misalignment_dict:
            for element in self.optr_data.data["elements"]:
                if name == element['name']:
                    self.optr_data.update_element(name=name, value=0.0)

    def set_misalignments(self):
        for name, mis_val in self.misalignment_dict.items():
            for element in self.optr_data.data["elements"]:
                if name == element['name']:
                    self.optr_data.update_element(name=name, value=mis_val)

    def update_beamline_state(self, actions_dict=None):
        '''
        updates all the tuning parameters of the beamline with a dictionary corresponding to the values of each steerer
        then does the integral calculations to obtain the transmission state throughout the beamline      
        if actions dict == None, will read the state from the last values in self.optr_data
        '''
        # print('Updating beamline state with actions dict:', actions_dict)
        # print(f"Current datadat: {self.optr_data.data['elements']}")
        if(actions_dict!=None):
            # uses the given dict to populate the actions
            if len(actions_dict)!=len(self.tuning_elements):
                print('warning: given action does not match size of action space')
            # print('Updating beamline state with actions dict:', actions_dict)
            for (name, action) in actions_dict.items():
                
                
                # print(f"Updating element {name} with action {action}")
                match = self.optr_data.data["elements"]
                # print(f"Current elements in datadat: {[el['name'] for el in match]}")
                match = [el for el in match if el.get("name") == name]
                # print(f"Match for element {name}: {match}")
                
                
                self.optr_data.update_element(name=name, value=action)
                self.last_actions[name] = action
        else:
            # uses the default data.dat values
            for name in self.tuning_elements:
                action = self.optr_data.get_element(name=name)['value']
                self.last_actions[name] = action
        self._run_simulation()
        self._calculate_transmission_integral()
        
        return self.read_beamline_state()
        
    def read_beamline_state(self):
        '''
        returns 
            steerers_dict: current value of all steering elements
            measurement_dict: current tranmission at all fcs
        '''
        steerers_dict = {}
        measurement_dict = {}
        for name, value in self.last_actions.items():
            steerers_dict[name] = value
            
        if self.measurement_device in self.config.sim.elements.fc:
            measurement_dict[name] = self.measure_fc(self.measurement_device)
        
        return steerers_dict, measurement_dict

    def render(self, show_steerer_pos=False, show_transmission=False):
        img = self._generate_simulation(show_steerer_pos=show_steerer_pos, show_transmission=show_transmission)
        return img

    def measure_fc(self,name):
        # gets the beam transmission at some FC position, as a percentage (1 is highest)
        
        index = np.argmin(abs(self.s-self.elements.fc[name].loc+self.zeros_offset))
        if max(self.s) - self.elements.fc[name].loc + self.zeros_offset < 0:
            print('WARNING WARNING WARNING: Measurement device %s name is outside of beamline by %s cm'%(name, max(self.s) - self.elements.fc[name].loc + self.zeros_offset))
            print('Problem is either location of %s or wrong drift lengths in sy.f.'%(self.measurement_device))
            time.sleep(1)

        fc_measurement = self.x_propagated_transmissions[index] * self.y_propagated_transmissions[index]
        return fc_measurement

    def measure_size(self):
        name=self.profile_device
        index = np.argmin(abs(self.s-self.elements.lpm[name].loc)+self.zeros_offset)
        sizex=self.x_ev[index]
        sizey=self.y_ev[index]
        cx=self.x_cm[index]
        cy=self.y_cm[index]
        return(sizex,sizey,cx,cy)

if __name__ == "__main__":
    user = os.getlogin()
    config = OmegaConf.create()
    # config.sim = OmegaConf.load(f'/home/{user}/repos/bayesopt/build_utils/olis/outputs/sim/olis.yaml')
    # config.sim = OmegaConf.load(f'/home/{user}/repos/bayesopt/config/sim/DTL3.yaml')
    config.sim = OmegaConf.load(f'/home/{user}/repos/bayesopt/config/sim/mebthebt.yaml')
    beamline = Beamline(config)
    
    steerers_dict, measurement_dict = beamline.read_beamline_state()
    cv2.imwrite('test_beamline_mis.png', beamline.render(show_steerer_pos=True, show_transmission=True))
    
    beamline.clear_misalignments()
    beamline.update_beamline_state(None)
    cv2.imwrite('test_beamline_nomis.png', beamline.render(show_steerer_pos=True, show_transmission=True))

    print('done')

