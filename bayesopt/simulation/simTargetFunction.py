import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from simulation.Beamline import Beamline
import torch as torch
torch.set_default_dtype(torch.float64)
import cv2

class SimTargetFunction():
    def __init__(self, config):
        self.config = config
        self.beamline = Beamline(config)
        self.beamline_states = []
        self.best_beamline_state = {}
        self.max_transmission = -1
        self.step = 0

    def __call__(self, x_set,png=None):

        if (png==None):
            png=self.config.save_gif

        """
        x_set is a list of lists, with each list corresponding to the value of each steerer
        contains a set of different x tensors, run a new simulation for each x. 
        We will grab the observation in the end that represents transmission.   
        """

        # the last steerer before FC6 is in first place. So optimize in opposite order.
        transmissions = []
        action_set = []
        # for name,value in self.beamline.tuning_elements.items():
        #     if(value['type']=="quad"):
        #         self.beamline.tuning_elements[f'{name}+:CUR'] = self.beamline.tuning_elements.pop(name)

        for x in x_set:
            assert len(x) == len(self.beamline.tuning_elements)
            action_dict = {name: value for name, value in zip(self.beamline.tuning_elements, x)}
            #print(action_dict)
            action_set.append(action_dict)
        
        for action_dict in action_set:
            self.beamline.update_beamline_state(action_dict)
            
            #if self.config.sim.measurement_device.type == 'fc':
            transmission = self.beamline.measure_fc(self.config.sim.measurement_device)
            transmissions.append(transmission)
            # else:
            #     raise ValueError(f"!!! error: measurement device of type {self.config.sim.measurement_device.type} not supported")

            if png:
                img = self.beamline.render(show_steerer_pos=True, show_transmission=True)
                beamline_state = {"step": self.step, "img": img, "transmission": transmission, "action": action_dict}
                self.beamline_states.append(
                    beamline_state
                )
                if transmission > self.max_transmission:
                    self.best_beamline_state = beamline_state
                    self.max_transmission = transmission
        self.step += 1       

        transmissions = torch.tensor(transmissions).unsqueeze(-1)
        return transmissions, [0] # this is because of the harp readings, which we should really just return as a single object instead of a tuple

    def read_inputs(self):
        '''Target functions need a method to read the current state of the system, which will be used as the initial point for optimization. This is less useful for simulation.'''
        # TODO: could possibly use the current state of the real machine. Would need PVs associate in the config file though
        # TODO: could also just use torch.tensor(self.mean) pretty sure
        return torch.tensor([value['starting_mean'] for name, value in self.beamline.tuning_elements.items()])
        # return self.mean

# if __name__ == "__main__":
#     user = os.getlogin()
#     config = OmegaConf.create()
#     config.sim = OmegaConf.load(f'/home/{user}/repos/bayesopt/config/sim/DTL3.yaml')
#     target_function = SimTargetFunction(config)
#     steerers_dict = target_function.beamline.tuning_elements
#     dim = len(steerers_dict.keys())
#     x_init = torch.zeros((1, dim))
#     y_set=target_function(x_init,png=False)
