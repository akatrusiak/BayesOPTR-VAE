from xml.parsers.expat import model

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from functools import partial

def getModel(self):
    """
    Defines an optimization loop for different models, getModel() returns a callable with a minimum of two parameters (train_x, train_y). 

    The optimization loop has three steps:

    1. Define the Gaussian process.
    2. Perform the hyperparameter fit.
    3. Optimize the acquisition function.

    Returns:
    model: the instantiated model with fitted hyperparameters.\\
    train_x (Tensor): updated train_x with the newest inputs as determined by the acquisition function. \\
    train_y (Tensor): updated train_y with the newest measured outputs. \\
    obj_curr (float): latest point in train_y. \\
    obj_max (float): maximum point in train_y. 
    """
    
    function_name = self.config.model
    types = {
        'standard': _standard,
        'multi-objective': _multi_obj,
        'constrained': _multi_obj
    }

    return lambda train_x, train_y, *args, **kwargs: types[function_name](self, train_x, train_y, *args, **kwargs)

def _standard(self, train_x, train_y, *args, **kwargs):
    
    obj_max = kwargs.pop('obj_max')
    
    model = SingleTaskGP(train_X=train_x, 
                        train_Y=train_y,  
                        outcome_transform=Standardize(m=1),
                        input_transform=Normalize(d=train_x.shape[1], 
                        bounds=self.bounds), 
                        mean_module=self.mean_module,
                        covar_module = self.kernel(ard_num_dims=train_x.shape[1],
                        batch_shape=torch.Size()))

    # If the mean module is a SurrogateMean, tell it about BoTorch's
    # transforms so it can operate in the correct spaces. Without this
    # the mean sees [0,1] inputs as if they were physical units and
    # returns physical y into a standardized target — the resulting
    # residuals grow with the dataset and eventually the Cholesky fit
    # blows up (manifests as a silent crash around iter 30).
    from core_functions.surrogate_mean import SurrogateMean
    if isinstance(self.mean_module, SurrogateMean):
        self.mean_module.set_context(
            bounds=self.bounds,
            y_mean_gp=model.outcome_transform.means,
            y_std_gp=model.outcome_transform.stdvs,
        )

    # performs the hyperparameter fit
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    
    # optimize acquisition function
    new_x, _, _ = self.generator(model, self.bounds, obj_max=obj_max)
    
    results, _ = self.target_function(new_x) 
    self.memory.append(results.item())
    
    # append the results to our training set and update the maximum
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, results])

    obj_max = train_y.max().item()
    obj_curr = train_y[-1].item()
    
    if isinstance(self.mean_module, SurrogateMean):
        print("GP y_mean:", model.outcome_transform.means.item(),
        "GP y_std:", model.outcome_transform.stdvs.item(),flush=True)
        print("surrogate y_mean:", self.mean_module.y_mean_surr.item(),
        "surrogate y_std:", self.mean_module.y_std_surr.item(),flush=True)

    return model, train_x, train_y, obj_curr, obj_max

def _multi_obj(self, train_x, train_y, *args, **kwargs): 

    train_y2 = kwargs.pop('train_y2')

    model_1 = SingleTaskGP(train_X=train_x, 
                        train_Y=train_y,  
                        outcome_transform=Standardize(m=1),
                        input_transform=Normalize(d=train_x.shape[1], 
                        bounds=self.bounds), 
                        covar_module = self.kernel(ard_num_dims=train_x.shape[1],
                        batch_shape=torch.Size()))
    
    model_2 = SingleTaskGP(train_X=train_x, 
                        train_Y=train_y2,  
                        outcome_transform=Standardize(m=1),
                        input_transform=Normalize(d=train_x.shape[1], 
                        bounds=self.bounds), 
                        covar_module = self.kernel(ard_num_dims=train_x.shape[1],
                        batch_shape=torch.Size()))
    
    model = ModelListGP(model_1, model_2)

    # performs the hyperparameter fit
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    
    # optimize acquisition function
    new_x, _, _ = self.generator(model, self.bounds)
    
    results_1, results_2 = self.target_function(new_x) 
    self.memory.append(results_1.item())
    
    # append the results to our training set and update the maximum
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, results_1])
    train_y2 = torch.cat([train_y2, results_2])

    obj_max = [train_y.max().item(), train_y2.max().item()]
    obj_curr = [train_y[-1].item(), train_y[-1].item()]

    return model, train_x, train_y, train_y2, obj_curr, obj_max