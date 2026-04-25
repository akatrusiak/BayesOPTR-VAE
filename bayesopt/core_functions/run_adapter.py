"""
run_adapter.py

Thin adapter layer that normalizes the beam and sim config structures
into a uniform interface that BayesOptParent can consume without
if/else branching.

Each adapter exposes:
    .target_function   — callable(x) -> y
    .steering_elements — dict of {pv_name: {lower_bound, upper_bound, starting_mean, starting_var, ...}}
    .common_elements   — dict (may be empty)
    .initial_transmission — float (for computing transmission_coeff)
    .bounds            — torch.Tensor shape [2, dim]
    .mean              — torch.Tensor shape [dim]
    .var               — torch.Tensor shape [dim]
    .dim               — int
"""

import torch
from abc import ABC, abstractmethod


class RunAdapter(ABC):
    """Uniform interface that BayesOptParent programs against."""

    @abstractmethod
    def __init__(self, config):
        ...

    @abstractmethod
    def setup(self):
        """Any interactive or expensive setup (auth, confirmation prompts, etc.)"""
        ...

    # Subclasses set these as attributes in __init__:
    target_function: callable
    steering_elements: dict
    common_elements: dict
    initial_transmission: float
    bounds: torch.Tensor
    mean: torch.Tensor
    var: torch.Tensor
    dim: int


class BeamRunAdapter(RunAdapter):

    def __init__(self, config):
        from direct_beamline_interface.DirectBeamlineRequest import DirectBeamlineRequest

        run = config.beam  # everything mode-specific lives here

        self.target_function = DirectBeamlineRequest(config)
        self.steering_elements = dict(run.steering_elements)
        self.common_elements = dict(run.get("common_elements", {}))
        self.initial_transmission = run.initial_transmission
        self.transmission_coeff = 100.0 / self.initial_transmission
        self.measurement_device = dict(run.measurement_device)

        # Build bounds, mean, var from the steering elements
        names = list(self.steering_elements.keys())
        self.dim = len(names)

        lower = [self.steering_elements[n].lower_bound for n in names]
        upper = [self.steering_elements[n].upper_bound for n in names]
        self.bounds = torch.tensor([lower, upper])

        self.mean = torch.tensor([self.steering_elements[n].starting_mean for n in names])
        self.var = torch.tensor([self.steering_elements[n].starting_var for n in names])
        self.range_dict = {n: self.steering_elements[n].upper_bound - self.steering_elements[n].lower_bound
                           for n in names}

    def setup(self):
        """Beam-specific: print PVs, ask for confirmation, authenticate."""
        for key in self.steering_elements:
            elem = self.steering_elements[key]
            print(key, elem.lower_bound, elem.upper_bound)
        print(f"Above are all the {self.dim} PVs you are going to change")

        confirmation = input("Please confirm that all the PVs are correct (Yes/No): ")
        if confirmation.lower() != "yes":
            print("Exiting the program.")
            exit()

        print("Please sign in...")
        self.target_function.check_auth()


class SimRunAdapter(RunAdapter):

    def __init__(self, config):
        from simulation.simTargetFunction import SimTargetFunction

        run = config.sim 

        self.target_function = SimTargetFunction(config)
        self.common_elements = {}

        # For sim, we need to extract the tuning elements from the
        # nested elements dict based on tuning_elements list
        # e.g. tuning_elements: ['steerer'] -> pull all from elements.steerer
        self.steering_elements = {}
        for elem_type in run.tuning_elements:
            if elem_type in run.elements:
                self.steering_elements.update(dict(run.elements[elem_type]))

        self.initial_transmission = run.get("initial_transmission", 1.0)
        self.transmission_coeff = 100.0 / self.initial_transmission
        print(f"SimRunAdapter: using initial_transmission={self.initial_transmission}")

        names = list(self.steering_elements.keys())
        self.dim = len(names)

        lower = [self.steering_elements[n].lower_bound for n in names]
        upper = [self.steering_elements[n].upper_bound for n in names]
        self.bounds = torch.tensor([lower, upper])

        self.mean = torch.tensor([self.steering_elements[n].starting_mean for n in names])
        self.var = torch.tensor([self.steering_elements[n].starting_var for n in names])
        self.range_dict = {n: self.steering_elements[n].upper_bound - self.steering_elements[n].lower_bound
                           for n in names}

    def setup(self):
        """Sim-specific: no auth needed, just print what we're tuning."""
        print(f"Simulation mode: tuning {self.dim} elements")
        for key in self.steering_elements:
            elem = self.steering_elements[key]
            print(f"  {key}: [{elem.lower_bound}, {elem.upper_bound}]")


def get_run_adapter(config) -> RunAdapter:
    """Factory: returns the right adapter based on config.run.mode"""
    adapters = {
        "beam": BeamRunAdapter,
        "sim": SimRunAdapter,
    }
    mode = config.mode
    if mode not in adapters:
        raise ValueError(f"Unknown mode '{mode}', expected one of {list(adapters.keys())}")
    return adapters[mode](config)
