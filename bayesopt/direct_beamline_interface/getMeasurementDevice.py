import torch
import numpy as np
import pandas as pd 
import time, ast
from tqdm import tqdm
torch.set_default_dtype(torch.float64)

def getMeasurementDevice(self):
    
    function_name = self.config.beam.measurement_device.type
    types = {
        'fc': _faraday_cup,
        'lpm': _profile_monitor,
        'multiple': _multiple
    }

    return types[function_name]

def _faraday_cup(self):
    
    measurements = []
    pv = self.config.beam.measurement_device.pv
    measure_points = 1 # self.config.beam.measurement_device.measure_points ##### placeholder
    trim_ratio = self.config.beam.measurement_device.trim_ratio
    scale_factor = self.config.beam.measurement_device.scale_factor
    
    while len(measurements) < measure_points:
        time.sleep(0.25)
        try:
            payload = {'readPvList': [pv]}
            response = self.session.post(self.get_url, json=payload)

            if response.status_code == 200:
                cup_value = response.json()["readPvDict"][pv]
                if cup_value!= 'None':
                    measurements.append(float(cup_value))
                else:
                    measurements.append(0.00)
            else:
                print(f"Error {response.status_code}")
        
        except Exception as e:
            print(f"Error encounted in measurement polling loop: {e}")

    self.transmissions_history.append(measurements)
    pd.DataFrame(self.transmissions_history).to_csv("all_transmission_measurements.csv")
    
    arr_meas = np.array(measurements)
    arr_meas.sort()

    trim_index = int(0.5*trim_ratio*measure_points)

    if trim_index > 0 and (measure_points-2*trim_index) > 0:
        mid_meas = arr_meas[trim_index:-trim_index]
    else:
        mid_meas = arr_meas

    final_measurement = np.mean(mid_meas)

    error_on_final_measurement = np.std(mid_meas) # variance for model
    
    self.transmissions_history_mean_error.append([final_measurement, error_on_final_measurement])
    pd.DataFrame(self.transmissions_history_mean_error).to_csv("transmission_mean_error.csv")
    
    return final_measurement * scale_factor, []


def _profile_monitor(self):
    self.profile_history = []
    self.position_history = []
    self.centroids_history = []
    scan_pv = self.config.beam.measurement_device.pv
    profile_pv = self.config.beam.measurement_device.profile_pv
    position_pv = self.config.beam.measurement_device.position_pv

    x_center = self.config.beam.measurement_device.x_center
    y_center = self.config.beam.measurement_device.y_center
    
    try:
        payload = {'setPvList': dict(zip([scan_pv], [1]))}
        response = self.session.post(self.set_url, json=payload)

        print(response.status_code)
        if response.status_code == 200:
            time.sleep(25)

            try:
                payload = {'setPvList': [profile_pv, position_pv]}
                response = self.session.post(self.get_url, json=payload)

                if response.status_code == 200:
                    data = response.json()["readPvDict"][profile_pv, position_pv]
                else:
                    print(f"Error {response.status_code}")
            except Exception as e:
                print(f"Error encounted in reading profiles: {e}")
        else:
            print(f"Error {response.status_code}")
        
    except Exception as e:
        print(f"Error encounted in scanning profiles: {e}")
    
    # get centroids
    centroid_x, centroid_y = _pm_processing(data, profile_pv, position_pv, x_center, y_center)

    # define beam alignment objective
    alignment_x = np.abs(centroid_x - x_center)
    alignment_y = np.abs(centroid_y - y_center)
    alignment = -(alignment_x + alignment_y)

    return alignment

def _pm_processing(self, data, profile_pv, position_pv, x_center, y_center):
    profile = np.array(ast.literal_eval(data[profile_pv]))
    profile[profile < 0] = 0
    position = np.array(ast.literal_eval(data[position_pv]))

    # profile_1 = profile[:1000]
    # profile_2 = profile[1000:][::-1]
    # position_1 = position[:1000]
    # position_2 = position[1000:][::-1]

    index_x, index_y = np.argmin(np.abs(position - x_center)), np.argmin(np.abs(profile - x_center))
    # index_y, index_y2 = np.argmin(np.abs(position_1 - y_center)), np.argmin(np.abs(position_2 - y_center))

    n = 100  # how many points off center to take

    # Ensure indices don't go out of bounds when slicing:
    def safe_slice_pair(arr_pos, arr_prof, center, n):
        """Returns slices from arr_pos and arr_prof centered at 'center' with n elements on each side."""
        start = max(0, center - n)
        end = min(len(arr_pos), center + n + 1)  # +1 to include the center itself
        return arr_pos[start:end], arr_prof[start:end]

    # Extract the subarrays:
    pos_slice_in_x, prof_slice_in_x = safe_slice_pair(position, profile, index_x, n)
    # pos_slice_out_x, prof_slice_out_x = safe_slice_pair(position_2, profile_2, index_x2, n)
    pos_slice_in_y, prof_slice_in_y = safe_slice_pair(position, profile, index_y, n)
    # pos_slice_out_y, prof_slice_out_y = safe_slice_pair(position_2, profile_2, index_y2, n)

    def merge_sorted_pairs(pairs1, pairs2):
        """
        Merge two lists of (position, profile) tuples into a single sorted list,
        sorted by the position value.
        """
        i, j = 0, 0
        merged = []
        while i < len(pairs1) and j < len(pairs2):
            if pairs1[i][0] < pairs2[j][0]:
                merged.append(pairs1[i])
                i += 1
            else:
                merged.append(pairs2[j])
                j += 1
        # Append any remaining items from either list:
        merged.extend(pairs1[i:])
        merged.extend(pairs2[j:])
        return merged

    # For x:
    pairs_in_x = list(zip(pos_slice_in_x, prof_slice_in_x))
    # pairs_out_x = list(zip(pos_slice_out_x, prof_slice_out_x))
    # merged_pairs_x = merge_sorted_pairs(pairs_in_x, pairs_out_x)

    # For y:
    pairs_in_y = list(zip(pos_slice_in_y, prof_slice_in_y))
    # pairs_out_y = list(zip(pos_slice_out_y, prof_slice_out_y))
    # merged_pairs_y = merge_sorted_pairs(pairs_in_y, pairs_out_y)

    position_x, profile_x = map(list, zip(*pairs_in_x))
    position_y, profile_y = map(list, zip(*pairs_in_y))

    self.profile_history.append({'profile_x': profile_x, 'profile_y': profile_y})
    self.position_history.append({'position_x': position_x, 'position_y': position_y})
    pd.DataFrame(self.profile_history).to_csv("all_profile_data.csv")
    pd.DataFrame(self.position_history).to_csv("all_position_data.csv")

    def weighted_centroid(x_data, y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        numerator = np.sum(x_data * y_data)
        denominator = np.sum(y_data)
        return numerator / denominator

    # Calculate the centroids for x and y data
    centroid_x = weighted_centroid(position_x, profile_x)
    centroid_y = weighted_centroid(position_y, profile_y)
    self.centroids_history.append({'centroid_x': centroid_x, 'centroid_y': centroid_y})
    pd.DataFrame(self.centroids_history).to_csv("centroids_data.csv")

    return centroid_x, centroid_y

def _multiple(self):
    """
    Custom function to handle problems with multiple objectives.

    """

