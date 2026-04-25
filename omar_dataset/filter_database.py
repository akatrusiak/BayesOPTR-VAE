import os, csv, inspect, shutil, argparse
from omegaconf import OmegaConf
import pandas as pd


class ExperimentData:
    ''''
    class to store all relevant information in a run
    '''
    def __init__(self, run_name, experiment, beta, num_elements, benders, first_element, fc, initial_current, final_current, transmission, energy, charge_state, isotope, path, bayesopt_type, bound_steering, step_weighted, num_iterations):
        self.run_name = run_name
        self.experiment = experiment
        self.beta = beta
        self.num_elements = num_elements
        self.first_element = first_element
        self.benders = benders
        self.fc = fc
        self.initial_current = initial_current
        self.final_current = final_current
        self.transmission = transmission
        self.energy = energy
        self.charge_state = charge_state
        self.isotope = isotope
        self.path = path
        self.bayesopt_type = bayesopt_type
        self.bound_steering = bound_steering
        self.step_weighted = step_weighted
        self.num_iterations = num_iterations

class DataLoader:
    '''
    load run data, deals with unconsistent config files and output in our experiment season. wicked code :)
    '''
    def __init__(self, root_directory, experiment_dict, path_dict):
        self.root_directory = root_directory
        self.experiment_dict = experiment_dict
        self.path_dict = path_dict
    
    def extract_two(self, input_string):
        segments = input_string.split(':')
        return ':'.join(segments[:2])

    def load_data(self, extract_two = True):
        data = []
        for directory in os.listdir(self.root_directory):
            if os.path.exists(os.path.join(self.root_directory, directory, 'optimal_input.yaml')):
                run_name = directory
                experiment_full = self.experiment_dict[run_name[:8]]
                config = OmegaConf.load(os.path.join(self.root_directory, directory, '.hydra/config.yaml'))
                num_steering_elements = len(config.beam.steering_elements)
                first = next(iter(config.beam.steering_elements.keys()))
                fc = config.beam.measurement_device.pv
                initial_current = round(config.beam.initial_transmission)
                beta = config.model.beta

                benders = 0
                for steerer_key in config.beam.steering_elements:
                    steerer_data = config.beam.steering_elements[steerer_key]
                    if "type" in steerer_data and steerer_data["type"] != "steerer":
                        # print(f"{run_name} bender 1 ")
                        benders = 1
                        break 
                
                # if benders == 0:
                #     print(f"{run_name} bender 0 ")
                

                optimal_input = OmegaConf.load(os.path.join(self.root_directory, directory, 'optimal_input.yaml'))
                final_current = round(optimal_input.transmission)

                transmission = round(optimal_input.transmission/config.beam.initial_transmission, 2)

                all_trans_path = os.path.join(self.root_directory, directory, "all_transmission_measurements.csv")
                num_iterations =  len(pd.read_csv(all_trans_path))

                if extract_two:
                    first = self.extract_two(first)
                    fc = self.extract_two(fc)

                bayesopt_type = getattr(config, 'bayesopt_type', "standard (old)")

                if hasattr(config, "bound_steering") and config.bound_steering is True:
                    bound_steering = 1
                else:
                    bound_steering = 0

                if hasattr(config, "step_weighted") and config.step_weighted is True:
                    step_weighted = 1
                else:
                    step_weighted = 0

                if hasattr(config, "error_on_train_y") and config.error_on_train_y is True:
                    bayesopt_type = "error (old)"
                
                energy, charge_state, isotope, path = self.path_dict[experiment_full]

                experiment_name = (experiment_full.split())[0]

                # print(run_name, initial_current, final_current, transmission)
                data.append(ExperimentData(run_name, experiment_name, beta, num_steering_elements, benders, first, fc, initial_current, final_current, transmission, energy, charge_state, isotope, path, bayesopt_type, bound_steering, step_weighted, num_iterations))

        return data


class DataFilter:
    ''' 
    Implement filtering logic based on criteria.
    '''
    @staticmethod
    def filter_data(data, criteria):
        filtered_data = []
        for item in data:
            match = True
            for key, value in criteria.items():
                if value != []:
                    # Special handling for range-based filtering (like transmission)
                    if key == "transmission":
                        if not (value[0] <= getattr(item, key, None) and (getattr(item, key, None) <= value[1])):
                            match = False
                            break
                    elif key == "date":
                        print(value, item.run_name[:8])
                        run_date = item.run_name[:8]
                        if isinstance(value, list):
                            if run_date not in value:
                                match = False
                                break
                        else:
                            if run_date != value:
                                match = False
                                break 
                    elif key == "num_iterations":
                        if not value[0] <= getattr(item, key, None):
                            match = False
                            break
                    elif key == "benders" or key == "bound_steering" or key == "step_weighted":
                        if value[0] != getattr(item, key, None):
                            match = False
                            break
                    else:
                        # Default handling for other criteria
                        if isinstance(value, list):
                            if getattr(item, key, None) not in value:
                                match = False
                                break
                        else:
                            if getattr(item, key, None) != value:
                                # print(getattr(item, key, None), value)
                                match = False
                                break 
            if match:
                filtered_data.append(item)
        return filtered_data


class CSVWriter:
    def __init__(self, root_directory):
        self.folder=os.path.join(os.path.dirname(root_directory),'filtered_database')
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def write_to_csv(self, data, file_name):

        sorted_data = sorted(data, key = lambda item : item.run_name)
        file_path = os.path.join(self.folder, f"{file_name}.csv")
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)

            # Assuming all ExperimentData have the same attributes
            params = inspect.signature(ExperimentData.__init__).parameters
            headers = list(params.keys())[1:] # exclude self
            writer.writerow(headers)
            for item in sorted_data:
                writer.writerow([getattr(item, attr) for attr in headers])
            print(f"Data written to {file_path}")
            runs_list = [item.run_name for item in data]
            print(runs_list)

class FilterFolder:
    def __init__(self, root_directory, subfolder_name):
        self.root_directory = root_directory
        self.prefolder=os.path.join(os.path.dirname(root_directory),'filtered_database')
        self.folder = os.path.join(self.prefolder, subfolder_name)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def create_filtered_folder(self, data):
        for item in data:
            src = os.path.join(self.root_directory, item.run_name)
            dst = os.path.join(self.folder, item.run_name)
            if os.path.exists(src):
                shutil.copytree(src, dst)
            else:
                print(f"Source directory {src} does not exist.")

    
class UserInterface:
    def __init__(self, data_loader, data_filter, csv_writer, filter_folder_name, filter_folder_enabled = False):
        self.data_loader = data_loader
        self.data_filter = data_filter
        self.csv_writer = csv_writer
        self.filter_folder_enabled = filter_folder_enabled
        if self.filter_folder_enabled:
            self.filter_folder = FilterFolder(root_directory, filter_folder_name)

    def run(self, criteria, file_name):
        data = self.data_loader.load_data()
        filtered_data = self.data_filter.filter_data(data, criteria)
        self.csv_writer.write_to_csv(filtered_data, file_name)
        print(f"csv database summary created under {file_name}")

        if self.filter_folder_enabled:
            self.filter_folder.create_filtered_folder(filtered_data)

##### USER INPUT #####

# Define the root directory where the run data is stored
root_directory = "/mnt/c/users/alexa/onedrive/triumf/bayesoptr_research/omar_dataset/species/12_C_3"

# Define the mapping of experiment dates to their corresponding names
experiment_dict = {
    "20241021": "HEBT 21 Oct",
    "20241106": "HEBT 6 Nov",
}

# Define the mapping of experiment names to their corresponding parameters
#energy (keV/u), charge state, isotope, path
path_dict = {
    "HEBT 21 Oct": [464, "3+", '12C', "ios-mws-hebt2-dragon"],
    "HEBT 6 Nov": [464, "3+", '12C', "ios-mws-hebt2-dragon"]
}


def get_user_input():
    parser = argparse.ArgumentParser(description='Filter Run information and Filter Options')
    
        # Arguments for file name and folder enable flag
    parser.add_argument('file_name', help="File name for the output CSV file.")
    parser.add_argument('--filter_folder_enabled', action='store_true', help='Save filtered runs in a folder if set. Default is False.')

    # Arguments for filtering criteria
    parser.add_argument('--run_name', nargs='*', default=[], help='Specific run names, e.g. 20231107_095856')
    parser.add_argument('--experiment', nargs='*', default=[], help='Experiment name, e.g. bnmr, hebt')
    parser.add_argument('--num_elements', nargs='*', default=[], help='Number of tuning elements')
    parser.add_argument('--benders', nargs = '*', default = [], help = '1 for benders, 0 otherwise')
    parser.add_argument('--first_element', nargs='*', default=[], help='First element tuned')
    parser.add_argument('--fc', nargs='*', default=[], help='FC used in run')
    parser.add_argument('--transmission', nargs=2, type=float, default=[], help='Transmission range to include, e.g. --transmission 0.5 1.0')
    parser.add_argument('--energy', nargs='*', default=[], help='energy in keV/u')
    parser.add_argument('--charge_state', nargs='*', default=[], help='Charge states, e.g. 1+ or 4+')
    parser.add_argument('--isotope', nargs='*', default=[], help='Isotopes to include')
    parser.add_argument('--path', nargs='*', default=[], help='Paths to include')
    parser.add_argument('--bayesopt_type', nargs='*', default=[], help='BayesOpt type: standard, standard (old), weighted, error')
    parser.add_argument('--bound_steering', nargs='*', default=[], help='1 if required, 0 if not')
    parser.add_argument('--step_weighted', nargs='*', default=[], help='1 if required, 0 if not required')
    parser.add_argument('--date', nargs = '*', default=[], help='Date in yyyymmdd format (GMT time)')
    parser.add_argument('--num_iterations', nargs=1, type=float, default=[], help='Minimum number of iterations, e.g. --num_iterations 40')
    parser.add_argument('--beta', nargs=1, type=float, default=[], help='beta value')
    args = parser.parse_args()

    if args.num_elements:
        args.num_elements = [int(element) for element in args.num_elements]

    if args.benders:
        args.benders = [int(bender) for bender in args.benders]

    return args

# Parsing the arguments
user_input = get_user_input()

# Extracting the criteria and other information from user_input
criteria = {
    "run_name": user_input.run_name,
    "experiment": user_input.experiment,
    "num_elements": user_input.num_elements,
    "benders": user_input.benders,
    "first_element": user_input.first_element,
    "fc": user_input.fc,
    "transmission": user_input.transmission,
    "energy": user_input.energy,
    "charge_state": user_input.charge_state,
    "isotope": user_input.isotope,
    "path": user_input.path,
    "bayesopt_type": user_input.bayesopt_type,
    "bound_steering": user_input.bound_steering,
    "step_weighted": user_input.step_weighted,
    "date": user_input.date,
    "num_iterations": user_input.num_iterations
}

file_name = user_input.file_name
filter_folder_enabled = user_input.filter_folder_enabled

data_loader = DataLoader(root_directory, experiment_dict, path_dict)
data_filter = DataFilter()
csv_writer = CSVWriter(root_directory)

ui = UserInterface(data_loader, data_filter, csv_writer, file_name, filter_folder_enabled)
ui.run(criteria, file_name)
 