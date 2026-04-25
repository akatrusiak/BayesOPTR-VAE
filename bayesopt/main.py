import os
import sys
import warnings
import time  
sys.path.append('..')
import hydra
import torch as torch
torch.set_default_dtype(torch.float64)

from bayesopt import BayesOpt

warnings.filterwarnings("ignore", category=UserWarning, module="botorch.models.utils.assorted")
os.environ['HYDRA_FULL_ERROR'] = "1"

@hydra.main(config_path='config/', config_name='config',version_base=None)
def main(config):
    start_time = time.time()

    BO = BayesOpt(config=config)
    
    model = BO.optimize()     
    BO.target_function(BO.optimal_input.unsqueeze(0))
    end_time = time.time()

    print(f"Total time taken: {end_time - start_time:.4f} seconds")

                       
if __name__ == '__main__':
    main()