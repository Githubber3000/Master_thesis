from __future__ import annotations
import numpy as np
import pymc as pm
import os
import logging
import scipy.stats as sp
import pytensor.tensor as pt

from .utils         import build_correlation_cov_matrix, create_directories, get_posterior_dim, ensure_2d
from .config        import save_adjusted_posterior_config, adjust_dimension_of_kwargs, adjust_circle_layout
from .reporting     import save_sample_info, plot_histogram
from .metrics       import sliced_wasserstein_distance, compute_mmd_rff, compute_mmd

   
logger = logging.getLogger(__name__)


def get_uniform_prior_bounds(means_array, expansion_factor=0.25, unimodal_init_margin=None):
    
    min_mode = np.min(means_array, axis=0)
    max_mode = np.max(means_array, axis=0)

    if len(means_array) == 1 and unimodal_init_margin is not None:
        # For unimodal: use a fixed margin
        border = unimodal_init_margin
        low = min_mode - border
        high = max_mode + border
    else:
        # For multimodal: compute bounding box
        diff = (max_mode - min_mode)
        min_margin = np.where(diff > 40, diff, 40.0)
        border = expansion_factor * min_margin
        low = min_mode - border
        high = max_mode + border
 
    return low, high, min_mode, max_mode, border


def get_initvals(init_scheme, means, eval_mode, num_chains, rng=None, run_id=None, init_folder=None, png_folder=None, varying_attribute=None, value=None, unimodal_init_margin = None):
    """Generates initialization values based on the chosen scheme.""" 

    if np.isscalar(means[0]):
        dim = 1
        means_array = np.array(means)[:, None]  # shape (n_modes, 1)
    else:
        means_array = np.array(means)
        dim = means_array.shape[1]


    if init_scheme == "thesis_scheme":
        # If multimodal posterior, use the means of the components, else spawn them randomly around the mean
        if len(means_array) >= 2:
            # Multimodal case
            # Compute bounding box across all dimensions
            low, high, min_mode, max_mode, border  = get_uniform_prior_bounds(means_array=means_array, expansion_factor=0.25)
            #single_init = rng.uniform(low, high).item() if dim == 1 else rng.uniform(low, high)
            #initvals = [{"posterior": single_init} for _ in range(num_chains)]
            initvals = [{"posterior": rng.uniform(low, high).item() if dim == 1 else rng.uniform(low, high)} for _ in range(num_chains)]

            if run_id == 1:
                init_info = {
                    "run_id": run_id,
                    "case": "multimodal",
                    "dim": dim,
                    "means_array": means_array.tolist(),
                    "min_mode": min_mode,
                    "max_mode": max_mode,
                    "border": border,
                    "low": low,
                    "high": high,
                    "samples": [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()} for d in initvals],
                }  
        else:
           
            low, high,_,_,_ = get_uniform_prior_bounds(means_array=means_array, expansion_factor=0.25, unimodal_init_margin=unimodal_init_margin)
            #single_init = rng.uniform(low, high).item() if dim == 1 else rng.uniform(low, high)
            #initvals = [{"posterior": single_init} for _ in range(num_chains)]
            initvals = [{"posterior": rng.uniform(low, high).item() if dim == 1 else rng.uniform(low, high)} for _ in range(num_chains)]

            if run_id == 1:          
                init_info = {
                    "run_id": run_id,
                    "case": "unimodal",
                    "dim": dim,
                    "low": low,
                    "high": high,
                    "means_array": means_array.tolist(),
                    "samples": [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()} for d in initvals],
                }

    elif init_scheme == "equal_per_mode":
        noise = 0.5
        initvals =[]
        for i in range(num_chains):
            mean = means_array[i % len(means_array)]
            center = mean + rng.normal(scale=noise)
            if dim == 1:
                center = center.item()
            initvals.append({"posterior": center})

        if run_id == 1:
            init_info = {
                "run_id": run_id,
                "case": "equal_per_mode",
                "dim": dim,
                "means_array": means_array.tolist(),
                "samples": [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()} for d in initvals],
            }

    elif init_scheme == "all_in_middle":
        middle_point = np.mean(means_array, axis=0)
        middle_point = middle_point.item() if dim == 1 else middle_point
        noise = 0.5
        initvals = [{"posterior": middle_point + rng.normal(scale=noise)} for _ in range(num_chains)]

        if run_id == 1:
            init_info = {
                "run_id": run_id,
                "case": "all_in_middle",
                "dim": dim,
                "means_array": means_array.tolist(),
                "middle_point": middle_point.tolist() if hasattr(middle_point, "tolist") else middle_point,
                "samples": [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()} for d in initvals],
            }

    elif init_scheme.startswith("all_near_mode_"):

        mode_index = int(init_scheme.split("_")[-1])
        if mode_index >= len(means):
            raise IndexError(f"Mode index {mode_index} out of bounds for available means.")
        
        target_mode = means_array[mode_index]
        target_mode = target_mode.item() if dim == 1 else target_mode
        noise = 0.5
        initvals = [{"posterior": target_mode + rng.normal(scale=noise)} for _ in range(num_chains)]

        if run_id == 1:
            init_info = {
                "run_id": run_id,
                "case": f"all_near_mode{mode_index}",
                "dim": dim,
                "means_array": means_array.tolist(),
                "mode_index": mode_index,
                "samples": [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()} for d in initvals],
            }

    if run_id == 1:
        parent_folder= os.path.join(init_folder, f"initvals_{eval_mode}")
        create_directories(parent_folder)

        chain_info_path = os.path.join(parent_folder, f"init_{varying_attribute}_{value}_{eval_mode}.json")
        chain_info_plot_path = os.path.join(parent_folder, f"init_{varying_attribute}_{value}_{eval_mode}.pdf")
        chain_info_png_path = os.path.join(png_folder, f"init_{varying_attribute}_{value}_{eval_mode}.png")
        
        save_sample_info(init_info, chain_info_path, chain_info_plot_path, chain_info_png_path, varying_attribute, value,"Init Values", init_info["case"])

    logger.debug(f"Generated initvals: {initvals}")
    return initvals


def get_logp_func(weights, components):
    def logp_func(x):
        logps = [pt.log(w) + pm.logp(comp, x).sum() for w, comp in zip(weights, components)]
        return pm.math.logsumexp(pt.stack(logps))
    return logp_func


def generate_iid_samples(posterior_type = None, num_samples=None, rng=None,**params):
    """
    Generate IID samples from a given posterior type.

    Parameters:
    - posterior_type: String specifying the type of the posterior (e.g., "Normal", "Mixture").
    - num_samples: Number of samples to generate.
    - rng: Optional random number generator.
    - **params: Additional parameters depending on posterior_type.
        - For "Mixture":
            - component_types: list of strings.
            - component_params: list of parameter dicts.
            - weights: list of floats.
        - For others: distribution-specific parameters.
    Returns:
    - iid_samples: Array of generated IID samples.
    """

    rng = rng or np.random.default_rng()

    # Mapping from string names to scipy sampling functions
    scipy_distributions = {
        "Normal": lambda p: sp.norm.rvs(loc=p["mu"], scale=p["sigma"], size=num_samples, random_state=rng),
        "SkewNormal": lambda p: sp.skewnorm.rvs(a=p["alpha"], loc=p["mu"], scale=p["sigma"], size=num_samples, random_state=rng),
        "StudentT": lambda p: sp.t.rvs(df=p["nu"], loc=p["mu"], scale=p["sigma"], size=num_samples, random_state=rng),
        "Beta": lambda p: sp.beta.rvs(a=p["a"], b=p["b"], size=num_samples, random_state=rng),
        "Cauchy": lambda p: sp.cauchy.rvs(loc=p["alpha"], scale=p["beta"], size=num_samples, random_state=rng),
        "Laplace": lambda p: sp.laplace.rvs(loc=p["mu"], scale=p["b"], size=num_samples, random_state=rng),
        "MvNormal": lambda p: sp.multivariate_normal.rvs(mean=np.array(p["mu"]), cov=np.array(p["cov"]), size=num_samples, random_state=rng),
        "MvStudentT": lambda p: sp.multivariate_t.rvs(df=p["nu"], loc=np.array(p["mu"]), shape=np.array(p["scale"]), size=num_samples, random_state=rng),
    }

    # Handle Skewed Student-T (which needs PyMC)
    if posterior_type == "SkewStudentT":
        with pm.Model():
            skewed_t = pm.SkewStudentT.dist(a=params["a"], b=params["b"], mu=params["mu"], sigma=params["sigma"])
            return pm.draw(skewed_t, draws=num_samples, random_seed=rng)

    # Handle single distributions
    if posterior_type in scipy_distributions:
        logger.debug(f"Generating {posterior_type} samples with parameters: {params}")
        return scipy_distributions[posterior_type](params)

    elif posterior_type == "Mixture":
        component_types = params["component_types"]
        component_params = params["component_params"]
        weights = params["weights"]

        if len(component_types) != len(component_params):
            raise ValueError("Each component type must have a corresponding parameter dictionary.")

        # normalize weights
        weights = np.array(weights) / np.sum(weights)

        # Choose which component each sample belongs to based on weights
        chosen_components = rng.choice(len(component_types), size=num_samples, p=weights)

        posterior_dim = get_posterior_dim("Mixture", {
            "component_types": component_types,
            "component_params": component_params,
            "weights": weights
        })

        if posterior_dim > 1:
            iid_samples = np.empty((num_samples, posterior_dim)) 
        else:
            iid_samples = np.empty(num_samples)

        for i, (comp_type, comp_params) in enumerate(zip(component_types, component_params)):
            # Select samples for this component
            mask = chosen_components == i  
            num_selected = mask.sum()
            if num_selected > 0:
                if comp_type in scipy_distributions or comp_type == "SkewStudentT":
                    iid_samples[mask] = generate_iid_samples(posterior_type=comp_type, num_samples=num_selected, rng=rng, **comp_params)

        return iid_samples
    
    else:
        raise ValueError(f"Unsupported posterior type: {posterior_type}")


def compute_and_store_iid_stats(
    iid_batches,
    value,
    num_iid_vs_iid_batches,
    iid_ref_stats_dict,
    iid_histogram_folder,
    png_folder,
    varying_attribute,
    posterior_type,
    do_mmd,
    do_mmd_rff,
    rng
):

    ref_swd_values = []
    ref_mmd_values = []
    ref_mmd_rff_values = []
    

    # get dimension of the first batch
    dim = ensure_2d(iid_batches[0]).shape[1]
    projections = 1 if dim == 1 else max(50, min(10*dim, 500))

    # Pairwise comparison for SWD/MMD stats
    for i in range(0, num_iid_vs_iid_batches, 2): 
        x = ensure_2d(iid_batches[i])
        y = ensure_2d(iid_batches[i + 1])

        swd = sliced_wasserstein_distance(x, y, L=projections, rng=rng)
        if do_mmd:
            mmd = compute_mmd(x, y, rng=rng)
        else:
            mmd = np.nan

        if do_mmd_rff:    
            mmd_rff = compute_mmd_rff(x, y, D=500, rng=rng)
        else:
            mmd_rff = np.nan
        
        ref_swd_values.append(swd)
        ref_mmd_values.append(mmd)
        ref_mmd_rff_values.append(mmd_rff)
        

    iid_ref_stats_dict[value] = {
        "mean_swd": np.mean(ref_swd_values),
        "std_swd": np.std(ref_swd_values, ddof=1),
        "median_swd": np.median(ref_swd_values),
        "q25_swd": np.quantile(ref_swd_values, 0.25),
        "q75_swd": np.quantile(ref_swd_values, 0.75),

        "mean_mmd": np.mean(ref_mmd_values),
        "std_mmd": np.std(ref_mmd_values, ddof=1),
        "median_mmd": np.median(ref_mmd_values),
        "q25_mmd": np.quantile(ref_mmd_values, 0.25),
        "q75_mmd": np.quantile(ref_mmd_values, 0.75),

        "mean_mmd_rff": np.mean(ref_mmd_rff_values),
        "std_mmd_rff": np.std(ref_mmd_rff_values, ddof=1),
        "median_mmd_rff": np.median(ref_mmd_rff_values),
        "q25_mmd_rff": np.quantile(ref_mmd_rff_values, 0.25),
        "q75_mmd_rff": np.quantile(ref_mmd_rff_values, 0.75)
    }

    plot_histogram(
        samples=iid_batches[0],
        title=f"IID Samples Histogram & KDE ({varying_attribute}={value})",
        save_path=os.path.join(iid_histogram_folder, f"iid_hist_kde_{varying_attribute}_{value}.pdf"),
        save_path_png=os.path.join(png_folder, f"iid_hist_kde_{varying_attribute}_{value}.png"),
        posterior_type=posterior_type,
        value=value
    )


def generate_all_iid_batches(
    posterior_type,
    posterior_kwargs,
    iid_kwargs_original,
    iid_kwargs,
    iid_posteriors_folder,
    varying_attribute,
    varying_values,
    num_total_iid_batches,
    num_iid_vs_iid_batches,
    num_samples,
    num_chains,
    rng,
    group_folder,
    png_folder,
    required_parameters,
    do_mmd,
    do_mmd_rff
):
    """
    Generates all IID batches for the given posterior type and varying attribute.
    
    Returns:
    - iid_batches_dict: Dictionary of generated IID batches.
    - iid_ref_stats_dict: Dictionary of reference statistics for SWD and MMD.
    """
    
    iid_histogram_folder = os.path.join(group_folder, "KDE and Histograms of IID Samples")
    create_directories(iid_histogram_folder)

    # === Handle Precomputed IID Samples for Varying Attributes ===
    # Dictionary to store generated IID batches and reference statistics
    iid_batches_dict = {}
    iid_ref_stats_dict = {}

    component_index = posterior_kwargs.get("varying_component", None)

    if posterior_type == "Mixture":

        is_studentt = posterior_kwargs["component_types"][0] == "MvStudentT"
        cov_param_key = "scale" if is_studentt else "cov" 

        # Loop through all varying values for Mixture posterior
        for value in varying_values:
            
            if varying_attribute == "weights":
                    iid_kwargs["weights"] = value
            elif varying_attribute == "dimension":
                adjust_dimension_of_kwargs(posterior_type, iid_kwargs_original, iid_kwargs, target_dim=value, required_parameters=required_parameters)
                save_adjusted_posterior_config(
                    iid_kwargs,
                    folder=iid_posteriors_folder,
                    dim_value=value
                )

            elif varying_attribute == "num_samples":
                num_samples = value
            elif varying_attribute == "num_chains":
                num_chains = value
            elif varying_attribute == "circle_radius":
                iid_kwargs["component_params"], \
                iid_kwargs["component_types"], \
                iid_kwargs["weights"] = adjust_circle_layout(
                    posterior_kwargs["component_number"],
                    posterior_kwargs["component_type"],
                    value,
                    posterior_kwargs["cov"],
                    posterior_kwargs["weight_type"]
                )

                save_adjusted_posterior_config(
                    iid_kwargs,
                    folder=iid_posteriors_folder,
                    dim_value=value
                )
            elif varying_attribute == "circle_modes":
                iid_kwargs["component_params"], \
                iid_kwargs["component_types"], \
                iid_kwargs["weights"] = adjust_circle_layout(
                    value,
                    posterior_kwargs["component_type"],
                    posterior_kwargs["radius"],
                    posterior_kwargs["cov"],
                    posterior_kwargs["weight_type"]
                )

                save_adjusted_posterior_config(
                    iid_kwargs,
                    folder=iid_posteriors_folder,
                    dim_value=value
                )
            elif varying_attribute == "correlation":
                
                posterior_dim = get_posterior_dim(posterior_type, iid_kwargs)

                for i, comp_params in enumerate(iid_kwargs["component_params"]):
                    
                    iid_kwargs["component_params"][i][cov_param_key] = build_correlation_cov_matrix(posterior_dim, value)

                save_adjusted_posterior_config(
                        iid_kwargs,
                        folder=iid_posteriors_folder,
                        dim_value=value
                )
            else:
                # Vary only the selected component's parameter
                iid_kwargs["component_params"][component_index][varying_attribute] = value

            samples_per_chain = num_samples // num_chains
            num_samples = samples_per_chain*num_chains

            iid_batches = [generate_iid_samples(
                posterior_type=posterior_type,
                component_types=iid_kwargs["component_types"],
                component_params=iid_kwargs["component_params"], 
                weights=iid_kwargs["weights"],
                num_samples= num_samples,
                rng=rng) for _ in range(num_total_iid_batches)]

            iid_batches_dict[value] = iid_batches

            compute_and_store_iid_stats(
                iid_batches=iid_batches,
                value=value,
                num_iid_vs_iid_batches=num_iid_vs_iid_batches,
                iid_ref_stats_dict=iid_ref_stats_dict,
                iid_histogram_folder=iid_histogram_folder,
                png_folder=png_folder,
                varying_attribute=varying_attribute,
                posterior_type=posterior_type,
                do_mmd=do_mmd,
                do_mmd_rff=do_mmd_rff,
                rng=rng
            )


    # Single posterior case
    else:
        is_studentt =  posterior_type == "MvStudentT"
        cov_param_key = "scale" if is_studentt else "cov" 

        for value in varying_values:
            
            if varying_attribute == "dimension":
                adjust_dimension_of_kwargs(posterior_type, iid_kwargs_original, iid_kwargs, target_dim=value, required_parameters=required_parameters)
                save_adjusted_posterior_config(
                    iid_kwargs,
                    folder=iid_posteriors_folder,
                    dim_value=value
                )
            elif varying_attribute == "num_samples":
                num_samples = value
            elif varying_attribute == "num_chains":
                num_chains = value
            elif varying_attribute == "correlation":
                posterior_dim = get_posterior_dim(posterior_type, iid_kwargs)
                iid_kwargs[cov_param_key] = build_correlation_cov_matrix(posterior_dim, value)
                save_adjusted_posterior_config(
                        iid_kwargs,
                        folder=iid_posteriors_folder,
                        dim_value=value
                )
            else:
                iid_kwargs[varying_attribute] = value  

            samples_per_chain = num_samples // num_chains
            num_samples = samples_per_chain*num_chains
            
            iid_batches = [generate_iid_samples(    
                posterior_type=posterior_type,
                **iid_kwargs,
                num_samples= num_samples,
                rng=rng) for _ in range(num_total_iid_batches)]

            iid_batches_dict[value] = iid_batches

            compute_and_store_iid_stats(
                iid_batches=iid_batches,
                value=value,
                num_iid_vs_iid_batches=num_iid_vs_iid_batches,
                iid_ref_stats_dict=iid_ref_stats_dict,
                iid_histogram_folder=iid_histogram_folder,
                png_folder=png_folder,
                varying_attribute=varying_attribute,
                posterior_type=posterior_type,
                do_mmd=do_mmd,
                do_mmd_rff=do_mmd_rff,
                rng=rng
            )

    return iid_batches_dict, iid_ref_stats_dict
