from __future__ import annotations
import os
import numpy as np
import pymc as pm
print(pm.__version__)
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

from .sampling      import get_logp_func
from .utils         import create_directories, get_posterior_dim
from .reporting     import save_sample_info


class PosteriorExample:
    """Base class for different posterior types."""
    
    def __init__(self):
        self.model = None
    
    def _define_posterior(self):
        """Subclasses should implement this method to define the posterior."""
        raise NotImplementedError("Subclasses must implement _define_posterior()")

    def run_sampling(self, sampler_name, num_samples=2000, tune=1000, num_chains=2, eval_mode=None, initvals=None, run_id=None, plot_first_sample=None, init_folder=None, value=None, means=None, posterior_type=None, run_random_seed=None):
        """Runs MCMC sampling using the chosen sampler."""

        with self.model:

            if sampler_name == "SMC":

                trace = pm.sample_smc(num_samples, chains=num_chains, progressbar=False, random_seed=run_random_seed)

                # # Print available diagnostic variables
                # print("sample_stats variables:", list(trace.sample_stats.data_vars))

                # # Print the beta schedule for each chain (temperature progression)
                # print("Beta schedule:")
                # print(trace.sample_stats["beta"])

                # # Print the acceptance rates per chain and stage
                # print("Acceptance rates:")
                # print(trace.sample_stats["accept_rate"])

                # # Print the log marginal likelihood (if available)
                # print("Log marginal likelihood:")
                # print(trace.sample_stats["log_marginal_likelihood"])

                # if run_id == 1:

                #     Plot how inverse temperature evolves (more stages = more gradual)
                #     az.plot_ess(trace, kind="evolution")
                #     plt.title("ESS Evolution")
                #     plt.show()

                #     Plot posterior distributions
                #     az.plot_posterior(trace)
                #     plt.show()


            else:
                
                # Define which sampler to use
                if sampler_name == "Metro":
                    sampler = pm.Metropolis()
                elif sampler_name == "HMC":
                    sampler = pm.NUTS()
                elif sampler_name == "DEMetro":
                    sampler = pm.DEMetropolis()
                elif sampler_name == "DEMetro_Z":
                    sampler = pm.DEMetropolisZ()
                elif sampler_name == "Slice":
                    sampler = pm.Slice()
                else:
                    raise ValueError(f"Unknown sampler: {sampler_name}")

                if run_id == 1:
                    discard_tuned_samples = False
                else:
                    discard_tuned_samples = True

                if initvals is not None:
                    trace = pm.sample(num_samples, tune=tune, step=sampler,initvals=initvals, chains=num_chains, return_inferencedata=True, discard_tuned_samples=discard_tuned_samples, progressbar=False, random_seed=run_random_seed)   
                else:
                    trace = pm.sample(num_samples, tune=tune, step=sampler, chains=num_chains, return_inferencedata=True, discard_tuned_samples=discard_tuned_samples, progressbar=False, random_seed=run_random_seed)

                if run_id == 1 and plot_first_sample and eval_mode == "pooled":
                    first_warmup_samples = trace.warmup_posterior["posterior"].isel(draw=0).values
                    dim = first_warmup_samples.shape[1] if first_warmup_samples.ndim > 1 else 1

                    warmup_info = {
                        "sampler": sampler_name,
                        "value": value,
                        "means_array": means,
                        "case": "Warmup Samples",
                        "dim": dim,
                        "samples": first_warmup_samples.tolist(),
                    }

                    # Define file paths
                    parent_folder = os.path.join(init_folder, f"{sampler_name}")
                    create_directories(parent_folder)
                    warmup_base = os.path.join(parent_folder, "first warm up samples")
                    warmup_json_path = f"{warmup_base}.json"
                    warmup_plot_path = f"{warmup_base}.pdf"

                    save_sample_info(sample_info=warmup_info, json_path=warmup_json_path, plot_path=warmup_plot_path, label="Samples")

                    # also plot first posterior sample
                    first_posterior_samples = trace.posterior["posterior"].isel(draw=0).values
                    posterior_info = {
                        "sampler": sampler_name,
                        "value": value,
                        "means_array": means,
                        "case": "Posterior Samples",
                        "dim": dim,
                        "samples": first_posterior_samples.tolist(),
                    }
                    
                    # Define file paths
                    posterior_base = os.path.join(parent_folder, "first posterior samples")
                    posterior_json_path = f"{posterior_base}.json"
                    posterior_plot_path = f"{posterior_base}.pdf"
                    save_sample_info(sample_info=posterior_info, json_path=posterior_json_path, plot_path=posterior_plot_path, label="Samples")
                    
        return trace


class SinglePosterior(PosteriorExample):
    def __init__(self, dist_name, dist_params, low=None, high= None, use_smc=False):
        """
        A flexible class for defining unimodal posteriors.

        Parameters:
        - dist_name: String specifying the name of the PyMC distribution (e.g., "Normal", "StudentT").
        - dist_params: Dictionary containing the parameters for the distribution.
        """
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.use_smc = use_smc
        self.low = low
        self.high = high
        super().__init__()
        self.model = self._define_posterior()

    def _define_posterior(self):
       
        dim = get_posterior_dim(self.dist_name, self.dist_params)
        shape = (dim,) if dim > 1 else ()
       
        # Retrieve the distribution class from PyMC
        dist_class = getattr(pm, self.dist_name)   
        dist = dist_class.dist(**self.dist_params, shape=shape)
        logp_func = lambda x: pm.logp(dist, x).sum() 

        if dim == 1 and self.low is not None and self.high is not None:
            low = self.low.item() if isinstance(self.low, np.ndarray) else self.low
            high = self.high.item() if isinstance(self.high, np.ndarray) else self.high
            self.low = low
            self.high = high

        with pm.Model() as model:
            if self.use_smc:
                x = pm.Uniform("posterior", lower=self.low, upper=self.high, shape=shape)
                pm.Potential("logp", logp_func(x))
            else:
                dist_class("posterior", **self.dist_params, shape=shape)
            
            #graph = pm.model_to_graphviz(model)

            #display(graph)      
        return model
        

class MixturePosterior(PosteriorExample):
    def __init__(self, component_types, component_params, weights=None, varying_component=None, low=None, high=None, use_smc=False): 
        """
        A flexible mixture posterior allowing any number of components and arbitrary distributions.

        Parameters:
        - component_types: List of strings specifying the type of each component (e.g., ["normal", "beta"]).
        - component_params: List of dictionaries, where each dictionary contains the parameters for the corresponding distribution.
        - weights: List of weights for the mixture components (defaults to uniform).
        """
        if len(component_types) != len(component_params):
            raise ValueError("Each component type must have a corresponding parameter dictionary.")

        if weights is None:
            weights = np.ones(len(component_types))  # Default: Equal weights

        if len(weights) != len(component_types):
            raise ValueError("Number of weights must match number of components.")

        self.component_types = component_types
        self.component_params = component_params
        self.weights = weights
        self.use_smc = use_smc
        self.low = low
        self.high = high

        # Normalize weights
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
        super().__init__()
        self.model = self._define_posterior()


    def _define_posterior(self):

        first_type = self.component_types[0]
        first_params = self.component_params[0]

        dim = get_posterior_dim(first_type, first_params)
        shape = (dim,) if dim > 1 else ()

        if dim == 1 and self.low is not None and self.high is not None:
            low = self.low.item() if isinstance(self.low, np.ndarray) else self.low
            high = self.high.item() if isinstance(self.high, np.ndarray) else self.high
            self.low = low
            self.high = high
        
        # Construct component distributions dynamically
        components = []
        for dist_type, params in zip(self.component_types, self.component_params):
                dist_class = getattr(pm, dist_type)  
                components.append(dist_class.dist(**params, shape=shape)) 
        
        # Construct logp_func for mixtures
        tensor_weights = pt.as_tensor_variable(self.weights)
        logp_func = get_logp_func(tensor_weights, components)

        # Define the mixture model    
        with pm.Model() as model:
            # Mixture model
            if self.use_smc:
                x = pm.Uniform("posterior", lower=self.low, upper=self.high, shape=shape)
                pm.Potential("logp", logp_func(x))
            else:
                pm.Mixture("posterior", w=self.weights, comp_dists=components, shape=shape) 
                        
            #graph = pm.model_to_graphviz(model)

            #display(graph)   
        return model
    

class CustomPosterior(PosteriorExample):
    """
    A flexible class to define custom posteriors using a user-specified log-probability function.
    """

    def __init__(self, logp_func):
        """
        Parameters:
        - logp_func: Callable function that defines the log-probability.
                     Must accept PyMC symbolic variables.
        - param_names: List of parameter names required by logp_func.
        - initvals: Optional dictionary for initial values.
        """
        super().__init__()
        self.logp_func = logp_func
        self.model = self._define_posterior()

    def _define_posterior(self):
        with pm.Model() as model:

            # Define the custom distribution
            pm.CustomDist("posterior", logp=self.logp_func)

        return model
