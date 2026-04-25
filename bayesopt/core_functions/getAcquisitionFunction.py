import sys
import torch as torch
sys.path.append('..')

from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition import qLogNoisyExpectedImprovement 
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling import SobolQMCNormalSampler
from functools import partial

def getAcquisitionFunction(config):
    
    function_name = config.acquisition_function
    types = {
        'UCB':_ucb,
        'EI':_ei,
        'qNEI': _qnei,
        'EHVI':_ehvi
    }

    beta = config.beta
    num_restarts = config.num_restarts
    raw_samples = config.raw_samples

    return partial(types[function_name], e=beta, num_restarts=num_restarts, raw_samples=raw_samples)


def _ucb(model, bounds, e=1.5, q_points=1, num_restarts=20, raw_samples=100, *args, **kwargs):
    
    UCB = UpperConfidenceBound(model, beta=e)
    
    candidates, acqf_list = optimize_acqf(acq_function=UCB, 
                                  bounds=bounds, 
                                  q=q_points, 
                                  num_restarts=num_restarts, 
                                  raw_samples=raw_samples,
                                  options= {"maxiter": 500}
                                )
    
    return candidates, acqf_list, UCB

def _ei(model, bounds, e=0.2, q_points=1, num_restarts=20, raw_samples=100, *args, **kwargs):

    obj_max = kwargs.pop('obj_max')

    EI = LogExpectedImprovement(model=model, best_f=(obj_max + e)) 

    candidates, acqf_list = optimize_acqf(acq_function=EI, 
                                  bounds=bounds, 
                                  q=q_points, 
                                  num_restarts=num_restarts, 
                                  raw_samples=raw_samples,
                                  options= {"maxiter": 500})
    
    return candidates, acqf_list, EI

def _qnei(model, bounds, e=0.2, q_points=1, num_restarts=20, raw_samples=100, *args, **kwargs):

    train_x = kwargs.pop('train_x')
    objective = GenericMCObjective(lambda Y, X=None: Y[..., 0])

    # Define constraint: current output (index 1) must be >= 0.95, i.e., 0.95 - current <= 0
    constraint_func = lambda Y: 0.95 - Y[..., 1]

    # Set up the qNEI acquisition function with the constraint
    sampler = SobolQMCNormalSampler(num_samples=500)  
    qNEI = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=sampler,
        objective=objective,
        constraints=[constraint_func]  
    )

    candidate, acq_value = optimize_acqf(
        acq_function=qNEI,
        bounds=bounds,
        q=q_points,                
        num_restarts=num_restarts,    
        raw_samples=raw_samples,    
        options={"maxiter": 200}
    )
    
    return candidate.detach(), acq_value, qNEI

def _ehvi(model, ref, y_data, bounds, e, q_points=1, num_restarts=20, raw_samples=100):

    partitioning = FastNondominatedPartitioning(ref_point=ref, Y=y_data)

    EHVI = ExpectedHypervolumeImprovement(model, ref, partitioning)

    candidates, acqf_list = optimize_acqf(acq_function=EHVI, 
                                          bounds=bounds, 
                                          q=q_points, 
                                          num_restarts=num_restarts, 
                                          raw_samples=raw_samples)

    return candidates, acqf_list, EHVI