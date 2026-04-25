import sys
sys.path.append('..')
import torch as torch
torch.set_default_dtype(torch.float64)
from bayesopt import BayesOpt
import os
from omegaconf import OmegaConf

run_path = '/home/ohassan/repos/bayesopt/bo_runs/beam/20250727_182736'
# 20250601_114615 # 20250601_122553 # 20250601_130156 # 20250601_133915
config = OmegaConf.load(os.path.join(run_path, '.hydra/config.yaml'))

BO = BayesOpt(config=config)

optimal_input = OmegaConf.load(os.path.join(run_path, 'optimal_input.yaml'))

input = [v for k,v in optimal_input.items() if k != 'transmission']
print("Optimal input:", input)
BO.optimal_input = torch.tensor(input).unsqueeze(0)
BO.target_function(BO.optimal_input)