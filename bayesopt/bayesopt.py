"""
bayesopt.py  (refactored)

BayesOptParent no longer knows or cares whether it's talking to
a real beamline or a simulation. The RunAdapter handles all of that.
"""

import os, sys, warnings
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from core_functions.getKernel import getKernel
from core_functions.getAcquisitionFunction import getAcquisitionFunction
from core_functions.getSamplingFunction import getSamplingFunction
from core_functions.getModel import getModel
from core_functions.scaling import getScalingFunction
from core_functions.plotting import Plotting
from core_functions.run_adapter import get_run_adapter
from core_functions.surrogate_mean import build_surrogate_mean_from_config

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=UserWarning, module="botorch.models.utils.assorted")

import traceback
import logging
log = logging.getLogger(__name__)


class BayesOptParent:

    def __init__(self, config):
        self.config = config

        # ─── Mode-specific setup via adapter ─────────────────────
        run = get_run_adapter(config)
        run.setup()  # auth + confirmation for beam; prints for sim

        self.target_function = run.target_function
        self.steerers_dict = run.steering_elements
        self.commons_dict = run.common_elements
        self.bounds = run.bounds
        self.dim = run.dim
        self.mean = run.mean
        self.var = run.var
        self.range_dict = run.range_dict
        self.transmission_coeff = run.transmission_coeff

        # ─── Optimization machinery (mode-independent) ───────────
        self.mode = config.mode
        self.objective = config.objective
        self.objective_units = config.objective_units

        self.generator = getAcquisitionFunction(config)
        self.kernel = getKernel(config)
        self.sampler = getSamplingFunction(self)
        self.optimize_loop = getModel(self)
        self.iterations = config.iterations

        # Optional surrogate-as-mean-function (defaults to None → BoTorch
        # ConstantMean via SingleTaskGP default).
        self.mean_module = build_surrogate_mean_from_config(
            mean_cfg=getattr(config, "mean_function", None),
            bois_input_names=list(self.steerers_dict.keys()),
        )



        # Plotting
        self.plot = Plotting(self)
        self.make_plots = config.make_plots
        self.xlabel = config.xlabel
        self.ylabel = config.ylabel

        # ─── Override bounds if using physics-informed scaling ────
        if config.bound in ("steerers", "quads"):
            self.mean_dict, self.var_dict, self.range_dict = getScalingFunction(self)(self)
            lower = [self.mean_dict[s] - 0.5 * self.range_dict[s] for s in self.steerers_dict]
            upper = [self.mean_dict[s] + 0.5 * self.range_dict[s] for s in self.steerers_dict]
            self.bounds = torch.tensor([lower, upper])
            self.mean = torch.tensor([self.mean_dict[s] for s in self.steerers_dict])
            self.var = torch.tensor([self.var_dict[s] for s in self.steerers_dict])
        else:
            # Use the adapter-provided values directly; also populate
            # mean_dict / var_dict for anything that reads them
            self.var_dict = {k: v.starting_var for k, v in self.steerers_dict.items()} # TODO I think this is repeated from the mean and var above from the adapter
            self.mean_dict = {k: v.starting_mean for k, v in self.steerers_dict.items()}

        # ─── State ───────────────────────────────────────────────
        self.memory = []
        self.max = float("-inf")
        self.model = None
        self.optimal_input = []
        self.y_variance = 0.0001

    # ─── Optimization loops ──────────────────────────────────────

    def _optimize(self):

        train_x, train_y, obj_max = self.sampler(self)
        self.memory = [obj_max]
        initial_length = train_x.shape[0]
        width = []
        print(f"Max {self.objective} from sampling:", obj_max, self.objective_units)

        try:
            for i in (pbar := tqdm(range(1, self.iterations + 1))):
                model, train_x, train_y, obj_curr, obj_max = self.optimize_loop(
                    train_x, train_y, obj_max=obj_max
                )

                width.append(model.covar_module.base_kernel.lengthscale.squeeze().tolist())
                index = torch.argmax(train_y)
                self.optimal_input = train_x[index]

                pbar.set_description(
                    f"{self.objective}: {obj_curr * self.transmission_coeff:.2f}%, "
                    f"max: {obj_max * self.transmission_coeff:.2f}%, iter: {i}"
                )

        except KeyboardInterrupt:
            print("Keyboard interrupted.")
            saved_length = len(self.memory)
            train_x = train_x[: saved_length + initial_length, :]
            train_y = train_y[: saved_length + initial_length, :]
            
        except Exception as e:
            log.exception("An error occurred during optimization: %s", str(e))
            traceback.print_exc()
            raise

        self.model = model
        self.max = obj_max

        optimal_settings = OmegaConf.create(
            {name: value.item() for name, value in zip(self.steerers_dict.keys(), self.optimal_input.squeeze())}
        )
        optimal_settings["objective_best"] = obj_max
        with open("optimal_input.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(optimal_settings))

        self.plot.final_data_processing(train_x, train_y, width)

        return self.model

    def _optimize_multi(self):
        return

    def _optimize_const(self):

        train_x, train_y, train_y2, obj_max = self.sampler(self)
        self.memory = obj_max
        initial_length = train_x.shape[0]
        width = []
        print(f"Max {self.objective} from sampling:", obj_max, self.objective_units)

        try:
            for i in (pbar := tqdm(range(1, self.iterations + 1))):
                model, train_x, train_y, train_y2, obj_curr, obj_max = self.optimize_loop(
                    train_x, train_y, obj_max
                )

                width.append(model.covar_module.base_kernel.lengthscale.squeeze().tolist())
                index = torch.argmax(train_y)
                self.optimal_input = train_x[index]

                pbar.set_description(
                    f"current: {obj_curr:.2f} pA, "
                    f"tx: {obj_curr * self.transmission_coeff:.2f}%, "
                    f"max: {obj_max:.2f}%, iter: {i}"
                )

        except KeyboardInterrupt:
            print("Keyboard interrupted.")
            saved_length = len(self.memory)
            train_x = train_x[: saved_length + initial_length, :]
            train_y = train_y[: saved_length + initial_length, :]

        self.model = model
        self.max = obj_max

        optimal_settings = OmegaConf.create(
            {name: value.item() for name, value in zip(self.steerers_dict.keys(), self.optimal_input.squeeze())}
        )
        optimal_settings["objective_best"] = obj_max
        with open("optimal_input.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(optimal_settings))

        self.plot.final_data_processing(train_x, train_y, width)

        return self.model


class BayesOpt(BayesOptParent):

    def __init__(self, config):
        super().__init__(config=config)

    def optimize(self):
        types = {
            "standard": self._optimize,
            "multi-objective": self._optimize_multi,
            "constrained": self._optimize_const,
        }
        return types[self.config.model]()
