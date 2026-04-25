import os, time, logging, requests
import torch as torch
import numpy as np
import pandas as pd 
from omegaconf import OmegaConf
import getpass
from .getMeasurementDevice import getMeasurementDevice

class DirectBeamlineRequest():
    
    def __init__(self, config):
        self.config = config

        # urls
        self.jaya_url = self.config.beam.get('jaya_url')
        self.get_url = self.jaya_url + "get"
        self.set_url = self.jaya_url + "set"
        print(self.jaya_url) 

        self.steering_elements = config.beam.steering_elements
        self.transmissions_history = []
        self.transmissions_history_mean_error = []
        self.count = 0
        
        #start a TCP connection 
        self.session=requests.Session()

        self.measurement_device = getMeasurementDevice(self)
    
    def check_auth(self):
        code = None
        check_url = self.jaya_url+"checkAuth" 

        while code !=200:
            # user input username and password
            username = input("input TRIDENT username: ")
            pw = getpass.getpass("Password: ")

            # set user information and check authorization
            self.session.auth = requests.auth.HTTPBasicAuth(username,pw)
            response = self.session.get(check_url,auth=(username,pw))
            code=response.status_code

        # get list of pvs that will be set
        pv_list = list(self.steering_elements.keys())

        # get live values of those pvs
        response = self.session.post(self.get_url, json={'readPvList': pv_list})

        # set live values back to sets setting privileges of user
        payload = {'setPvDict':response.json()['readPvDict']}
        response = self.session.post(self.set_url, json=payload)
        status = response.json()['status']
        
        # if not a success message, quit program
        if status!="Set Post Received. Success - set PV values via jaya-isac":
            print(f"error testing user access:\n{status}")
            exit()
    
    def __call__(self, x_set):
        transmission_set = []
    

        for new_x in x_set:
            assert len(new_x) == len(self.steering_elements) #checks number of pvs t set is the num of steering elements
            x_dict = {name: value.item() for (name, value) in zip(self.steering_elements.keys(), new_x)}
            
            self.incremental_set_pvs(new_x=x_dict)
            t_value, _ = self.measurement_device(self)
            transmission_set.append(t_value)
                
        transmission = torch.tensor(transmission_set)
        
        return transmission.unsqueeze(-1), [0]
    
    def get_incremental_setpoints(self, start, end, max_incremental_change, set_zero_crossing):
        '''
        start: float value at start
        end: final float value
        max_incremental_change: the maximum increment for decrement interval each step
        set_zero_crossing: whether 0 must be included if sign change occurs
        
        returns: list of each setpoint from start (excluding) to end (including)
        '''
        epsilon = 0.01
        if start > end:
            max_incremental_change *= -1
        if set_zero_crossing and np.sign(start) != np.sign(end):
            set_points1 = np.arange(start, end, max_incremental_change)
            set_points1 = np.append(set_points1,[epsilon,0,-epsilon])
            if start < end:
                set_points = np.sort(set_points1)
            else:
                set_points = np.sort(set_points1)[::-1]

            self.count +=1
            
        else:
            set_points = np.arange(start, end, max_incremental_change)
            
        set_points = np.append(set_points, end)
        if len(set_points) > 1:
            set_points = np.delete(set_points, 0)
            
        return set_points.tolist()
        
    def incremental_set_pvs(self, new_x):

        get_url = self.jaya_url + "get"
        pv_list = [name for name in new_x.keys()]
        current_x = self.session.post(get_url, json={'readPvList': pv_list}).json()["readPvDict"] #How will this work?
        current_x = {pv: float(value) for pv, value in current_x.items()}
        
        set_points_dict = {}
        max_len = 0
        for name in current_x.keys():
            start = current_x[name]
            end = new_x[name]
            max_incremental_change = self.config.beam.steering_elements[name].max_incremental_change
            set_zero_crossing = self.config.beam.steering_elements[name].set_zero_crossing
            set_points = self.get_incremental_setpoints(start, end, max_incremental_change, set_zero_crossing)
            set_points_dict[name] = set_points
            max_len = max(max_len, len(set_points))
            
        set_list = []
        for i in range(max_len):
            set_list.append({name: set_points_dict[name][i] for name in current_x.keys() if len(set_points_dict[name]) > i})
        
        #set_list is a dictionary {PV1: val1, PV2: val2, ...}
        for x in set_list:
            self.send_measurement(x)
            # time.sleep(0.1)
            
        if np.any([self.config.beam.steering_elements[steerer].set_zero_crossing for steerer in self.config.beam.steering_elements]):
            time.sleep(self.config.switch_time) #wait n seconds after sending measurements if ANY pvs cross zero
            
    def send_measurement(self, setpoints):
        #setpoints is a dictionary {PV1: val1, PV2: val2, ...} 
        logging.debug(f"Initializing request with setpoints: {setpoints}")

        payload = {"setPvDict": dict(setpoints)}

        #post the requests to JAYA (payload is placed as a json file)
        response = self.session.post(self.set_url, json=payload)

        if response.status_code == 200:
            time.sleep(1) # wait time for jaya/epics to update
            return
        else:
            raise RuntimeError(
                f"Error {response.status_code} in POST request to set PVs: {response.text}"
            )

    def read_inputs(self):
        pv_list = list(self.steering_elements.keys())
        response = self.session.post(self.get_url, json={'readPvList': pv_list})
        data = torch.tensor([float(response.json()["readPvDict"][pv]) for pv in pv_list])
        return data
    
    def read_inputs_full(self):
        pv_list = list(self.steering_elements.keys())
        response = self.session.post(self.get_url, json={'readPvList': pv_list})
        return response.json()["readPvDict"]


if __name__ == '__main__':
    user = os.getlogin()
    config = OmegaConf.create()
    config.beam = OmegaConf.load(f'/home/{user}/repos/bayesopt/config/beam/MEBT.yaml')

    request = DirectBeamlineRequest(config)
    request.send_measurement({'MEBT:Q7:CUR' : 42})
