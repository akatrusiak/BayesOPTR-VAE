from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.priors.torch_priors import GammaPrior

def getKernel(config):
    if config.kernel.lengthscale_prior.type == 'Gamma':
        lengthscale_prior= GammaPrior(config.kernel.lengthscale_prior.concentration, config.kernel.lengthscale_prior.rate)
    else:
        raise KeyError(f"!!!check config: lengthscale prior of type {config.kernel.lengthscale_prior.type} not defined")
    
    if config.kernel.outputscale_prior.type == 'Gamma':
        outputscale_prior= GammaPrior(config.kernel.outputscale_prior.concentration, config.kernel.outputscale_prior.rate)
    else:
        raise KeyError(f"!!!check config: outputscale prior of type {config.kernel.outputscale_prior.type} not defined")
          
    if config.kernel.type == 'Matern':
        def Matern(ard_num_dims, batch_shape):
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=config.kernel.nu,
                    ard_num_dims=ard_num_dims,
                    batch_shape=batch_shape,
                    lengthscale_prior= lengthscale_prior
                ),
                batch_shape=batch_shape,
                outputscale_prior=outputscale_prior
            )
            return covar_module
        return Matern

    elif config.kernel.type == 'Gaussian':
        def Gaussian(ard_num_dims, batch_shape):
            covar_module = ScaleKernel(
                base_kernel = RBFKernel(
                                ard_num_dims,
                                lengthscale_prior=lengthscale_prior,     
                                batch_shape=batch_shape,
                                outputscale_prior=outputscale_prior
                            )
                )
            return covar_module
        return Gaussian
    else:
        raise KeyError(f"!!!check config: kernel of type {config.kernel.type} not defined")