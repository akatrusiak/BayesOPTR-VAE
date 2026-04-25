# bayesopt

Using Bayesian Optimization for Beamline Tuning

For documentation and related notes, see the [BO Notion Page](https://stump-milkshake-736.notion.site/TRIUMF-Bayesian-Optimization-For-Beam-Tuning-d00ae8fa0baf4ccfa34c199207b8849f?pvs=4) 

## Installation

Currently, only main_sim.py is working, to tune simulated beamline using BO.

### To run Bayesian optimization with a simulated beamline
- All configurations are in the [config](config) folder
- [sim_run_config.yaml](config/sim_run_config.yaml) intializes the hydra folder where the run will reside. All corresponding files will be stored under a folder with the run-time timestamp
- in the [config/sim/](config/beam) folder, place the yaml file corresponding to the beamline simulation that will be used
- in [standard_bo.yaml](config/model/standard_bo.yaml), all parameters corresponding to the model can be set, including kernel, acquisition functions, sample points, etc
- run [main_sim.py](main_sim.py) to begin bayesian optimization on selected beamline sim.


### To run Bayesian optimization with the real beamline
- [beam_run_config.yaml](config/beam_run_config.yaml) intializes the hydra folder where the run will reside. All corresponding files will be stored under a folder with the run-time timestamp
- in the [config/sim/](config/beam) folder, place the yaml file corresponding to the section beamline that will be used
- in [standard_bo.yaml](config/model/standard_bo.yaml), all parameters corresponding to the model can be set, including kernel, acquisition functions, sample points, etc
- run [main_real_beamline.py](main_real_beamline.py) to begin bayesian optimization on selected beamline sim.


# Bayesopt Primer 

### Defne Tanyer (dtanyer@uwaterloo.ca), David Wang (davidw0311@gmail.com), Hui Wen Koay 

# Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Bayesian Optimization](#bayesian-optimization)
4. [Configuration](#configuration)
5. [Simulation](#simulation)
6. [Real Beamline Interface](#real-beamline-interface)
7. [Recommendations/Future Works](#recommendations)


<div id='overview'/>

# Overview 

This repository contains the scripts and interface required to train a bayesian optimization (BO) agent for AI-based tuning of the particle beamlines. Two of the main functions of the repo include runing on simulation, where BO is used for tuning a simulated section of a beamline created from TRANSOPTR, and running on real beam, where the BO agent interfaces with JAYA to communicate with EPICS and tune certain beamline elements in real time.   

<div id='requirements'/>

# REQUIREMENTS <a name="#requirements"></a>

Required dependencies: 
- Python 
- Pytorch 
- Matplotlib.pyplot 
- Hydra (configuring code) 
- Joblib (“) 
- Omegaconf (“) 
- Accpy 
- Pyoptr 
- cProfile (runtime monitor) 
- Transoptr 

Required: local ACC install

the ACC database must be cloned from https://gitlab.triumf.ca/hla/acc/-/tree/master?ref_type=heads

it does not matter if it is a user install or a global install, as long as you define a variable in your ~/.bashrc:
```
export ACCDIR=/path/to/my/install/acc/
```
important to end your path with /acc/


<div id='bayesian-optimization'/>

# BAYESIAN OPTIMIZATION  

## [Bayesopt.py](bayesopt.py) 

Bayesian optimization uses a Gaussian process to optimize an expensive “black box” function. With the current implementation, we use a single-task BO agent, meaning the output of this black box function is a single value (for our case this is the transmission measured at a FC).  

The goal is to obtain the maximum of this black box function. There can be many inputs to this function.  

Below is the framework that we follow. 

All that is needed to run bayesopt.py is to set the values in the config file located under /config/model/<model_name>.yaml 

Note: Most of the hyperparameters from the given example are good to go for general case. You’ll mostly play with beta, initial sample points and iterations. 

**User action:**

- Modify relevant config files. 

- Run either main_sim.py or main_real_beamline.py 

- If running main_real_beamline.py make sure middleman.py is running on a trusted computer. 

- All output files will be stored under /bo_runs/ 

## [bayesopt](bayesopt.py): 

Generates a PyTorch tensor of size initial_sample_points.  

Consisting of random points scattered in the parameter space or sampled points from a normal distribution centered around user input value for each dimension (if random_starting_means is False). 

Passes this tensor through the objective function creating initial set of observations D. 

$\mathbb{D} = \{(x_i, f(x_i)): i = 1, 2,.. \}$  

(optimization loop starts)  

for i in range(iterations) do 

Feeds the observation set to SingleTaskGP 

Normalizes input x using min-max normalization 

Standardizes output y 

Note: Currently the default BoTorch prior $\mathbb{M} \sim \mathbb{N}(0, 1)$ is used. We use strong hyperparameters for the kernel hence the normalization and standardization steps are important. https://botorch.org/api/models.html#botorch.models.gp_regression.SingleTaskGP  

Using ExactMarginalLogLikelihood compute the likelihood and compute posterior model using Bayes’ Rule. 

The posterior model will output a mean $\mu$ and uncertainty $\sigma$ for each x. 

Evaluate the model and return $x_{new} = \arg \max_{x \in X} \mu + \beta\sigma$ 

$\mathbb{D} \longleftarrow \mathbb{D} \cup \{ (x_{new}, f(x_new)) \} 

Saves the current training dataset to “training_set.csv” 

(end of optimization loop) 

Plots progress. 

If make_plots is True, plots the posterior predictions, the kernel, lengthscale vs iteration, x_explored. 

If scan_vs_fit is True, plots the posterior prediction with real data overlaid. It’s another config parameter since getting the data takes a long time. We don’t want it every time. 

Lastly if it’s a simulation, saves a gif of beam images. 


<div id='configuration'/>

# CONFIGURATION

To ensure the clarity of all experimental runs and saving of data, the main scripts in the repo make use of the Hydra (https://hydra.cc/) library.  

This ensures that  

- Data from each different run can be clearly saved and identified later 

- All configurations and hyperparameters of the agent and interface can be controlled via a few yaml configuration files 

- The ease of parallelization and performing hyperparameter sweeps (on simulation) 

To start a run, the following three config files must be edited, under /config/ 

### If running on simulation: 

[/config/sim_run_config.yaml](/config/sim_run_config.yaml)

    The top level config, specifies which model config and which sim config to use 

    Also includes flags for performing run time profiling, visualization, saved runs naming convention, and multi-run settings (using joblib) 

/config/sim/<name_of_simulation>.yaml 

    The sim config, name of the file must match what is specified in sim_run_config.yaml 

    This file configures the transoptr simulation and configures the section to simulate, their starting values, any misalignments, measurement devices, and location information for running the simulation 

/config/model/<name_of_model>.yaml 

    The model config, name of the file must match what is specified in sim_run_config.yaml 

    Configures all settings for the bayesian optimization model, hyperparameters, kernel types, beta, number of iterations, whether to do scans, etc. 

### If running on real beamline 

[/config/beam_run_config.yaml](/config/beam_run_config.yaml)

    The top level config, specifies which model config and which beamline config to use 

    Also includes flags for performing run time profiling and saved runs naming convention 

/config/beam/<name_of_beamline>.yaml 

    Beam config, name of the file must match what is specified in beam_run_config.yaml 

    Configures the settings for all the PVs on the real beamline, as well as which faraday cup will be used for the measurement 

/config/model/<name_of_model>.yaml 

    The model config, name of the file must match what is specified in beam_run_config.yaml 

    Configures all settings for the bayesian optimization model, hyperparameters, kernel types, beta, number of iterations, whether to do scans, etc. 

 

Example: /config/model/standard_bo.yaml 

 

- Iterations: how many iterations in the optimization loop 

- Initial sample points: how many points in the initial observation set (before optimization) 

- Candidate points: how many new points generated each loop (currently only supports 1) 

- Num restarts and raw samples: parameters of singletaskGP can read more on https://botorch.org/docs/models  

- Scan vs fit (bool). if True, runs a 1D scan at the end of the optimization loop, sets each element to found optimum and sweeps the range for each and plots data vs prediction. 

- Scan vs fit points: how many points are we plugging into the function 

- Beta: acquisition function hyperparameter 

- Prior on length scale (Gamma Distribution Γ(α, β)) https://www.desmos.com/calculator/vk2tqrxpk5 : 

- α: rate (usually 3.0) 

- β: concentration (usually 6.0) 

## Covariance module: 

- Type: “Gaussian”, “Matern” 

- Nu: 0.5, 1.5, 2.5 (if type is Matern) 

- The length scale (Θ) is a value computed for each dimension that characterizes the maximum scale such that you can say for x and x+ Θ values for f(x) and f (x+ Θ) are similar. In other words, if the length scale is high your function is flat otherwise it is sensitive to small changes. 

- Beta hyperparameter characterizes the trade-off between exploration and exploitation. Our acquisition function is Upper Confidence Bound (UCB) $x = \arg \max_{x \in X}\mu + \beta \sigma$. Too low beta, then you’re prone to get stuck on local maxima from lack of exploration and too high beta, you will likely not maximize the function since you are prioritizing minimizing uncertainty over maximizing the function which is not the objective. 

<div id='simulation'/>

# SIMULATION 

[Beamline.py](simulation/Beamline.py)

A class which creates a simulation of a beamline section, using TRANSOPTR, and wrapped around python for easy interface. 

Requires the right config file under bayesopt/sim/ with a path to relevant data.dat and sy.f file for the beamline section. 

data.dat and sy.f files can be generated in the right formatting from accpy database using /build_utils/ 

Considers the beam a 2D Gaussian in the xy-plane initially with 100% transmission. 

Based on the apertures stated in the config and the location of the mean of the Gaussian for each axis (computed by Beamline.py), generates an array of lost transmission percentage using an erf loss function 

Slit widths, misalignments, transmission and loss are all stored in arrays. 

All info needed for calculation and running the simulation should be in the config file. 

The Beamline.py class has the following public methods which should be used while interfacing with the simulated beamline. All other methods are private and direct access to these methods should be avoided. 

**clear_misalignments**
Sets all misalignments in the beamline to 0 

**measure_fc**
Measures the transmission at the location of the faraday’s cup (specified in config) 

**measure_size**
Measures the beam profile at the location of the rpm/lpm (specified in config) 

**read_beamline_state** 
Reads the current state of the beamline, including transmission at the fc and the current value set at each quad or steerer 

**render **
produces an image for visualization for the current state of the beamline 

**set_misalignments **
turns on the misalignments for the beamline. Each time a new instance of Beamline is instantiated, it will be initialized with some set of misalignments. This method applies those misalignments, not generate new ones.  

**update_beamline_state **
If passed a dictionary of values, will set the corresponding quads or steerers given in the dictionary, otherwise, uses the values that are in the data.dat file for all PVs.  

## [simTargetFunction.py](simulation/simTargetFunction.py)

The target function that is being maximized by the optimizer, uses Beamline.py to generate transmission, it is a callable object that can be configured to be compatible with any beamline section so long as the config file is in /config/sim/ 

From /config/sim the object is initiated with the names of components to be optimized. 

When called the function iterates through the list of elements and sets the desired value in the beamline simulation. 

Once iterated through, it runs the simulation and obtains the propagated transmission. 

Returns this transmission. 

**If png is set to True: **
Keeps an array of images of beamline from each step of the optimization. 


Stores a gif of the process. 

<div id='real-beamline-interface'>

# REAL BEAMLINE INTERFACE 

## [/beamline_interface/](beamline_interface)

[BeamlineRequest.py](beamline_interface/BeamlineRequest.py)

This class is used as the target function for running the Bayesian optimizer while interfacing with a real beamline section. 

Once an instance of this class has been instantiated, the object acts as the black box target function. On each iteration, the object takes in an array of values at input, which correspond to the values for each steerer or quadrupole (in order given on the config file), and outputs the measured transmission at the given Faraday Cup (also specified on the config file).   

In order to set PVs on the real beamline, at each iteration, BeamlineRequest writes to a file called request.json, which posts the names and values of the PVs it wishes to set on Jaya. Middleman must be running at the same time on a trusted server in order for this request file to be sent successfully through JAYA. Middleman continuously checks for the request.json file, and once it is created, sends the values on the file to JAYA to set the PVs. Once set, request.json gets deleted by Middleman. BeamlineRequest waits for the deletion of request.json before starting the next iteration. 

# Setting up Bayesian Optimizer for real beamline section. 

To run the optimizer for a real beamline section, only one .yaml file is required to be created in the bayesopt repository. Under config/beam/ create a new .yaml file for the experiment. This file should contain all the PVs, for the section of the beamline that will be tuned, under the steering_elements tag.  


In the example config above, the first two elements which are to be optimized are MEBT:Q6 and MEBT:Q7.  

Their names are the names of the PV which should be set on JAYA.  

Lower and upper bounds specifiy the min and max range the PV should be set at 

Starting_mean is the know best value of the PV, a predefined setpoint that is usually calculated using MCAT 

Starting_var is the width of the gaussian for which initial random points should be sampled around the mean 

Max_incremental_change is the biggest step the PVs should be incremented when setting. If the optimizer tries to take a step larger than max_incremental_change, then multiple rounds of requests will be sent to set the PV incrementally to the setpoints in max step sizes of max_incremental_change 

Set_zero_crossing: if set to True, whenever the PV requires a polarity switch, it will always be set to zero before the polarity switch. (see the get_incremental_setpoints() and. incremental_set_pvs() methods in BeamlineRequest.py for more details, example tests for this method can be seen on Notion)  

 

# Simulation: 


Config file should have all the keys (coloured in blue on the image above) formatted as needed. 

- Name: name of beamline section 

- Description: what will be the name of the file 
- Type: beamline 
- Random seed: seed used in simulation for random offset/misalignment 
- Acc path: path to acc database (not yet incorporated) 
- Datadat path: path to data.dat file for the beamline section 
- Optr dir: path to the folder that contains all transoptr documents 
- Beamline wall width: max aperture aka the width of the beamline 
- Offset: offset that the beam is coming in with initially 
- Tuning elements: which class of elements are we optimizing? [quad, steerer, dipole] 
- Misalignment elements: 
    - Measurement device: where is the transmission taken? 
    - Profile device: where is the beam profiling measurement taken? 

- Each element under the elements list should have “type” and “loc” information and some extra for types: rf, quad, steerer, slit, dipole. 

 

 

# SETTING UP BO FOR BRAND NEW BEAMLINE SECTION 

- Start in the build_utils folder 

- Create a new folder, giving it the name of the new simulation (for example, mebthebt) 

- Copy over the files build_optr.py, extract_config.py, and tune_config.xml from the examples folder 

- Edit tune_config.xml with the proper beam parameters, preset PVs, and the starting and ending elements of the section that you are interested in 

- In build_optr.py, change the name to match the name of the new beamline section 

- Run build_optr.py 

    This will generate the syf, data.dat, the transoptr executable required to run the simulation, which will be populated in the transoptr folder in bayesopt 

- Run extract_config.py 

    This will generate the <name>.yaml file which contains the configurations for the simulation, under config/sim 

    Change the apertures dict to include apertures of interest

- Test the simulation 

- Go to simulation/beamline, changing the config file to the newly generated yaml, and running Beamline.py 

    This will generate two images in the simulation folder, one showing the beamline without any misalignments, and the other with misalignments.  

    This can serve as a playground to play around with the beamline simulation 

- Setting up configuration 

Inside the yaml configuration file, configure the following: 

- misalignment magnitudes 

- Comment out the steerers and quads which do not need to be optimized 

- Change config/sim_run_config to point to the new yaml file 

- Change parameters in the model config under config/model 

- Run main_sim.py to begin optimizing 

 

# BAYESIAN OPTIMIZER: 

Overall, the procedure is as follows: 

Edit 3 config files. 

Make sure middleman is running if running real beamline. 

Run main_<type>.py 

<div id='recommendations'>

# RECOMMENDATIONS: 

Future suggestions for improving the code repository include: 

- Incorporating the use of gpu accelerating for the BO-Agent, to speed up the training times of the agent 

- To aid future hyperparameter sweeps of the agent, investigate Weights and Biases (https://wandb.ai/site) for tracking hyperparameter sweeps.  

- Integrate a random seed into the simulation. If seed is set as Null, then assign a random seed seed to the simlation which gets saved and can be used later to reproduce the results.  