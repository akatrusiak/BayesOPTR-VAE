import torch 
from tqdm import tqdm
torch.set_default_dtype(torch.float64)

def getSamplingFunction(self):
    
    function_name = self.config.sampling    
    types = {
        'fixed': _fixed,
        'variable': _variable,
        'informed': _informed,
    }

    return types[function_name]

def _fixed(self):
        """
        Collects a fixed number n of initial samples based on user input.

        This function generates a set of initial points (x_init) and their corresponding function values (y_init).
        It can either randomly sample points or use known theoretical means and variances to generate points.

        Returns:
        x_init (Tensor): Initial input points, shape (initial_points, dim).
        y_init (Tensor): Corresponding function values, shape (initial_points,).
        y_max (float): Maximum function value in the initial set.
        """
        self.initial_points = self.config.initial_points

        # a for loop to go back and fix the entries so that they lie between lower bound (lb) and upper bound (ub)
        print(f"Collecting {self.initial_points} initial training points")
        
        if self.config.random_starting_means:
            x_init = torch.rand((self.initial_points, self.dim))
            for dimension in range(self.dim):
                # torch.rand generates a tensor of given size but all entries are between 0, 1
                lb, ub = self.bounds[0, dimension].detach(), self.bounds[1, dimension].detach()
                x_init[:, dimension] = (ub-lb)*x_init[:, dimension] + lb
        else:

            # if the theoretical is known, enter mean and variance and the initial points will be sampled
            x_init = torch.zeros((self.initial_points, self.dim))
            for index in range(self.initial_points):
                if index == 0:
                    x_init[index, :] = self.target_function.read_inputs()
                    # print('Initial point from read_inputs():', x_init[index, :])
                else:
                    lb, ub = self.bounds[0].detach(), self.bounds[1].detach()
                    x_init[index,:] = torch.clip(torch.normal(self.mean, self.var), lb, ub)

        y_init, _ = self.target_function(x_init)
        y_max = y_init.max().item()
        index = torch.argmax(y_init)
        self.optimal_input = x_init[index]

        print(f"Done Training\nMaximum sampling {self.objective}: {y_max} {self.objective_units}")
     
        return x_init, y_init, y_max

def _variable(self):
    """
    Collects a variable number n of initial samples based on input dimensions d, where n is 2d+1 < n < 10d. Stops when user defined threshold and the minimum iterations is reached or maximum iterations reached.

    This function generates a set of initial points (x_init) and their corresponding function values (y_init).
    It can either randomly sample points or use known theoretical means and variances to generate points.

    Returns:
    x_init (Tensor): Initial input points, shape (initial_points, dim).
    y_init (Tensor): Corresponding function values, shape (initial_points,).
    y_max (float): Maximum function value in the initial set.
    """

    sampling_threshold = self.config.sampling_threshold#*(1/self.transmission_coeff)
    init_sp_min = 2*self.dim+1 
    init_sp_max = 10*self.dim

    obj_max = 0
    
    print('Variable sampling mode:')
    print(f"Collecting between {init_sp_min} and {init_sp_max} sampling points.")
    print(f"Ending sampling when {sampling_threshold*(1/self.transmission_coeff)} {self.objective_units} found")

    index=0 

    #loop until threshold hit and more than minimum iterations, or max iterations hit
    for i in (pbar := tqdm(range(1, init_sp_max))):
        if obj_max*self.transmission_coeff>sampling_threshold and i>init_sp_min:
            break
        x_init = torch.zeros((1, self.dim))
        if i == 1:
            x_init[0, :] = self.target_function.read_inputs()
        
        elif self.config.random_starting_means:
            x_init = torch.rand((1, self.dim))
            for dimension in range(self.dim):
                # torch.rand generates a tensor of given size but all entries are between 0, 1
                lb, ub = self.bounds[0, dimension].detach(), self.bounds[1, dimension].detach()
                x_init[:, dimension] = (ub-lb)*x_init[:, dimension] + lb
        else:
            # if the theoretical is known, enter mean and variance and the initial points will be sampled
            x_init = torch.zeros((1, self.dim))
            for index in range(1):
                    lb, ub = self.bounds[0].detach(), self.bounds[1].detach()
                    x_init[index,:] = torch.clip(torch.normal(self.mean, self.var), lb, ub)

        if self.config.model != 'standard':
                y1_init, y2_init = self.target_function(x_init)
                
                if i == 1:
                    y1_train = y1_init.clone().detach()
                    y2_train = y2_init.clone().detach()
                    x_train = x_init.clone().detach()
                else:
                    y1_train=torch.cat([y1_train,y1_init])
                    y2_train=torch.cat([y2_train,y2_init])
                    x_train=torch.cat([x_train,x_init])
                    
                y_max = [y1_train.max().item(), y2_train.max().item()]
                index = torch.argmax(y_init)
                self.optimal_input = x_init[index]
                pbar.set_description(f'{self.objective}: {y1_init.item():.3f} {self.objective_units}, objective_2: {y2_init.item():.3f}, max: {y_max[-1]} {self.objective_units}, iter: {i}')

        else:
            y_init, _ = self.target_function(x_init)

            if i == 1:
                y_train = y_init.clone().detach()
                x_train = x_init.clone().detach()
            else:
                y_train=torch.cat([y_train,y_init])
                x_train=torch.cat([x_train,x_init])
                
            y_max = y_train.max().item()
            obj_max = y_max
            best = torch.argmax(y_init)
            self.optimal_input = x_init[best]
            pbar.set_description(f'latest: {y_init[0,0].item()*self.transmission_coeff:.2f}%, max: {obj_max*self.transmission_coeff:.2f}%, iter: {i}')

        self.initial_points = i

    if self.config.model != 'standard':
        print(f"Done Training\nMaximum Training [{self.objective}: {y_max[0]} {self.objective_units}, objective_2: {y_max[1]}]")
        return x_train, y1_train, y2_train, y_max
    else:
        print(f"Done Training\nMaximum Training {self.objective}: {y_max} {self.objective_units}")
        return x_train, y_train, y_max


def _informed(self):
    return
