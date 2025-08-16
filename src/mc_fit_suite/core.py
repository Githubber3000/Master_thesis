from __future__ import annotations
import copy, logging, time, os
import arviz as az
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    set_logging_level,
    create_directories,
    get_git_tag,
    safe_json_dump,
    ensure_2d,
    build_correlation_cov_matrix,
    get_posterior_dim,
    extract_means_from_posterior,
)

from .config import (
    adjust_dimension_of_kwargs,
    adjust_circle_layout,
    save_adjusted_posterior_config,
    adjust_mode_means
)

from .posteriors import SinglePosterior, MixturePosterior, CustomPosterior

from .sampling import (
    generate_all_iid_batches,
    get_uniform_prior_bounds,
    get_initvals
)

from .metrics import (
    compute_summary_discrepancies,
    get_scalar_rhat_and_ess,
    sliced_wasserstein_distance,
    compute_mmd_rff,
    compute_mmd,
    compute_summary_discrepancies,
    count_mode_transitions
)

from .reporting import (
    plot_and_save_all_metrics,
    compute_and_save_global_metrics,
    handle_trace_plots,
    plot_pairwise_scatter
    )


def eval_trace(
        trace,
        runtime,
        eval_level,           
        run_id,
        sampler_name,
        value,
        posterior_type,
        posterior_kwargs,
        iid_batch,
        experiment_settings,
        folders,
        varying_attribute,
        results,
        do_mmd,
        do_mmd_rff,
        save_pairwise_scatter,
        rng
):
    
    # Plot trace plots in notebook if requested
    if experiment_settings.get("plot_traces_in_notebook", False):
        handle_trace_plots(
            eval_level=eval_level,
            trace=trace,
            sampler_name=sampler_name,
            varying_attribute=varying_attribute,
            value=value,
            show=True,
            save_path=None,
            save_individual=False,
        )

    trace_plot_mode = experiment_settings.get("trace_plots", "none")

    # Save trace plots to PDF if requested
    if trace_plot_mode == "all" or (trace_plot_mode == "first_run_only" and run_id == 1):

        save_path = os.path.join(folders["var_attr_folder"],f"{eval_level}_{sampler_name}_{value}_trace_plot.pdf")
        png_path = os.path.join(folders["png_folder_traces"], f"{eval_level}_{sampler_name}_{value}_trace_plot.png")
        html_path = os.path.join(folders["png_folder_scatter"], f"{eval_level}_{sampler_name}_{value}_pairwise_scatter.html")
        dim = get_posterior_dim(posterior_type, posterior_kwargs)

        handle_trace_plots(
            eval_level=eval_level,
            trace=trace,
            iid_samples=iid_batch,
            sampler_name=sampler_name,
            varying_attribute=varying_attribute,
            value=value,
            show=False,
            save_path=save_path,
            png_path=png_path,
            save_individual=experiment_settings.get("save_individual_traceplots_per_dim", False)
        )

        if save_pairwise_scatter and dim > 1:
            plot_pairwise_scatter(
                eval_level=eval_level,
                mcmc_trace=trace,        
                iid_samples=iid_batch, 
                sampler_name=sampler_name, 
                varying_attribute=varying_attribute,       
                value=value,
                marker_size=8,          
                marker_opacity=0.5,    
                html_path= html_path
            )

    # Save trace to NetCDF file if requested
    if experiment_settings.get("save_traces", False):
        trace_filename = os.path.join(folders["var_attr_folder"], f"{sampler_name}_trace.nc")

        az.to_netcdf(trace, trace_filename)


    # poolwise evaluation of posterior samples (multiple chains)
    posterior_samples = trace.posterior["posterior"].values

    # Ensure posterior_samples always has shape (N, dims)
    if posterior_samples.ndim == 2:
        posterior_samples = posterior_samples.reshape(-1, 1) 
    else:
        posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[-1])

    # Compute Wasserstein distance and MMD if not custom posterior
    if posterior_type != "Custom":

        mean_rmse, var_rmse = compute_summary_discrepancies(posterior_samples, iid_batch)

        dim = posterior_samples.shape[1]
        projections = 1 if dim == 1 else max(50, min(10*dim, 500))

        mcmc_vs_iid_swd = sliced_wasserstein_distance(posterior_samples, iid_batch, L=projections, rng=rng)

        if do_mmd:
            mmd_value = compute_mmd(posterior_samples, iid_batch, rng=rng)
        else:
            mmd_value = np.nan

        if do_mmd_rff:
            mmd_rff_value = compute_mmd_rff(posterior_samples, iid_batch, D=500, rng=rng)
        else:
            mmd_rff_value = np.nan


    else:
        mcmc_vs_iid_swd = np.nan
        mmd_value = np.nan
        mmd_rff_value = np.nan
        
    if eval_level == "pooled":
        # Compute R-hat and ESS (if not SMC)
        if not sampler_name == "SMC":
            r_hat, ess = get_scalar_rhat_and_ess(trace)
        else:
            r_hat = np.nan
            ess = np.nan
        mode_transitions = np.nan
    else:
        # For single chain, we can only compute ESS (if not SMC)
        if not sampler_name == "SMC":
            r_hat, ess = get_scalar_rhat_and_ess(trace, compute_rhat=False)
        else:
            r_hat = np.nan
            ess = np.nan
        if posterior_type == "Mixture" and not sampler_name == "SMC":
            mode_transitions = count_mode_transitions(posterior_samples)
        else:
            mode_transitions = np.nan


    results.append({
        "eval_level": eval_level,
        "run_id": run_id,
        varying_attribute: value,
        "sampler": sampler_name,
        "mean_rmse": mean_rmse,
        "var_rmse": var_rmse,
        "wasserstein_distance": mcmc_vs_iid_swd,
        "mmd_rff": mmd_rff_value,
        "mmd": mmd_value,
        "runtime": runtime,
        "ess": ess,
        "ess_per_sec": ess / runtime,
        "r_hat": r_hat,
        "mode_transitions": mode_transitions
    })


def run_experiment(
    results_folder,
    png_folder,
    experiment_settings,
    posterior_type,
    config_descr,
    runs,
    varying_attribute, 
    varying_values,      
    num_samples,
    num_chains,
    init_scheme=None,
    base_random_seed=None,
    progress_bar=None,
    group_name="default",
    **posterior_kwargs
):
    
    
    set_logging_level(experiment_settings.get("logging_level", "INFO"))
    logger = logging.getLogger()
    
    logger.info(f"===== Config {config_descr} started! =====")

    # Initialize random number generator
    rng = np.random.default_rng(base_random_seed)

    samples_per_chain = "varies" if varying_attribute in ["num_samples", "num_chains"] else num_samples // num_chains
    # Adjust total to match per-chain sample count
    num_samples = samples_per_chain*num_chains

    component_index = posterior_kwargs.get("varying_component")
    do_mmd = experiment_settings.get("do_mmd", False)
    do_mmd_rff = experiment_settings.get("do_mmd_rff", False)
    save_pairwise_scatter = experiment_settings.get("save_pairwise_scatter", False)

    # Number of IID batches for the IID vs IID comparison
    num_iid_vs_iid_batches = 2*runs
    num_mcmc_batches = runs
    # Total number of iid batches (needs a fresh iid batch for each mcmc run)
    num_total_iid_batches = num_iid_vs_iid_batches + num_mcmc_batches

    # Define required parameters for each posterior type
    required_parameters = {
        "Mixture": ["component_types", "component_params", "weights"],
        "Cauchy": ["alpha", "beta"],
        "Beta": ["a", "b"],
        "Normal": ["mu", "sigma"],
        "SkewNormal": ["mu", "sigma", "alpha"],
        "StudentT": ["nu", "mu", "sigma"],
        "SkewStudentT": ["a", "b", "mu", "sigma"],
        "Laplace": ["mu", "b"],
        "MvNormal": ["mu", "cov"],
        "MvStudentT": ["nu", "mu", "scale"],
        "Custom": []
    }

    # Validate that required keys exist (except for varying attribute)
    required_keys = [k for k in required_parameters.get(posterior_type) if k != varying_attribute]
    if not all(k in posterior_kwargs for k in required_keys):
        raise ValueError(f"{posterior_type} posterior requires {required_keys}")

    # Create keyword arguments for IID sample generation
    iid_kwargs = {key: posterior_kwargs.get(key, "varies") for key in required_parameters.get(posterior_type)}
 
    logger.debug(f"Using IID sample settings: {iid_kwargs}")

    # Create configuration and histogram folders inside the experiment root
    group_folder = os.path.join(results_folder, group_name, config_descr)
    init_folder = os.path.join(group_folder, f"init_info")
    runs_folder = os.path.join(group_folder, f"runs ({runs})")
    png_folder_kde = os.path.join(png_folder, "IID_KDE_and_Histograms")
    png_folder_traces = os.path.join(png_folder, "trace_plots")
    png_folder_scatter = os.path.join(png_folder, "pairwise_scatter")
    create_directories(group_folder, init_folder, runs_folder, png_folder_kde, png_folder_traces, png_folder_scatter)

    if varying_attribute in ["dimension", "correlation", "circle_radius", "circle_modes", "mm"]:
        posterior_kwargs_original = copy.deepcopy(posterior_kwargs)
        iid_kwargs_original = copy.deepcopy(iid_kwargs)
        iid_posteriors_folder  = os.path.join(init_folder, "iid_posteriors")
        regular_posteriors_folder = os.path.join(init_folder, "regular_posteriors")
        create_directories(iid_posteriors_folder, regular_posteriors_folder)
    else:
        posterior_kwargs_original = None
        iid_kwargs_original = None
        iid_posteriors_folder = None
        regular_posteriors_folder = None


    experiment_metadata = {
        "config_descr": config_descr,
        "samplers": experiment_settings["samplers"],
        "group_name": group_name,
        "runs": runs,
        "total_iid_batches": num_total_iid_batches,
        "iid_vs_iid_comparisons": num_iid_vs_iid_batches // 2,  
        "mcmc_vs_iid_comparisons": num_mcmc_batches,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "samples_per_chain": samples_per_chain,
        "posterior_type": posterior_type,
        "varying_attribute": varying_attribute,
        "varying_values": varying_values,
        "init_scheme": init_scheme,
        "base_random_seed": base_random_seed,
        "git_tag": get_git_tag(),
    }

    # Add posterior-specific parameters
    experiment_metadata.update(iid_kwargs)  

    # Save metadata
    metadata_filename = os.path.join(group_folder, f"metadata_config_{config_descr}.json")
    safe_json_dump(experiment_metadata, metadata_filename)

    # generate iid batches (not needed for Custom posterior, since no iid samples available)
    if posterior_type != "Custom":

        print (f"Generating {num_total_iid_batches} IID batches for {posterior_type} posterior with varying attribute '{varying_attribute}' ")
        print(f"with varying values {varying_values}...")

        iid_batches_dict, iid_ref_stats_dict, direction_dict = generate_all_iid_batches(
            posterior_type=posterior_type,
            posterior_kwargs=posterior_kwargs,
            iid_kwargs_original=iid_kwargs_original,
            iid_kwargs=iid_kwargs,
            iid_posteriors_folder=iid_posteriors_folder,
            varying_attribute=varying_attribute,
            varying_values=varying_values,
            num_total_iid_batches=num_total_iid_batches,
            num_iid_vs_iid_batches=num_iid_vs_iid_batches,
            num_samples=num_samples,
            num_chains=num_chains,
            rng=rng,
            group_folder=group_folder,
            png_folder=png_folder_kde,
            required_parameters=required_parameters,
            do_mmd=do_mmd,
            do_mmd_rff= do_mmd_rff,
        )       

    # Define fixed colors for each sampler
    sampler_colors = {
        "Metro": "blue",
        "HMC": "red",
        "DEMetro": "green",
        "DEMetro_Z": "purple",
        "SMC": "orange",
    }

    # move SMC to the end of the samplers list for efficient building of PyMC models
    samplers = list(experiment_settings["samplers"]) 
    if "SMC" in samplers:
        samplers.remove("SMC")
        samplers.append("SMC")

    chain_eval_mode = "chain"
    pooled_eval_mode = "pooled"

    png_folder_init_chain = os.path.join(png_folder, "chain_init")
    png_folder_init_pooled = os.path.join(png_folder, "pooled_init")
    
    create_directories(png_folder_init_chain, png_folder_init_pooled)

    plot_first_sample = experiment_settings.get("plot_first_sample", False)
    df_all_runs = []

    # === Run the Experiment ===
    for run_id in range(1, runs + 1):
        logger.info(f"Running {config_descr} - Run {run_id}")

        run_random_seed = int(rng.integers(1_000_000))
        run_rng = np.random.default_rng(run_random_seed)

        run_folder = os.path.join(runs_folder, f"run_{run_id}")
        traces_folder = os.path.join(run_folder, "trace_plots")
        create_directories(run_folder,traces_folder)

        if experiment_settings.get("save_plots_and_csv_per_run", False):
            csv_folder = os.path.join(run_folder, "result CSVs")
            csv_pool   = os.path.join(csv_folder,  "pooled")
            csv_chain  = os.path.join(csv_folder,  "chain")
            plots_folder = os.path.join(run_folder, "plots_of_run")
            plot_pool  = os.path.join(plots_folder, "pooled_global_plots")
            plot_chain = os.path.join(plots_folder, "chain_global_plots")
            
            create_directories( csv_folder, plots_folder,
                               csv_pool, csv_chain, plot_pool, plot_chain)

        results = []

        for value in varying_values:

            var_attr_folder = os.path.join(traces_folder, f"{varying_attribute}_{value}")
            create_directories(var_attr_folder)

            if run_id == 1:
                # create subfolder for value in init folder
                init_value_folder = os.path.join(init_folder, f"{varying_attribute}_{value}")
                create_directories(init_value_folder)

            # Handle parameter changes for Mixture case
            if posterior_type == "Mixture":

                is_studentt = posterior_kwargs["component_types"][0] == "MvStudentT"
                cov_param_key = "scale" if is_studentt else "cov"

                if varying_attribute == "weights":
                    posterior_kwargs["weights"] = value
                elif varying_attribute == "mm":
                    posterior_dim = get_posterior_dim(posterior_type, posterior_kwargs)

                    direction = np.array(direction_dict.get(value))
                    adjust_mode_means(posterior_kwargs["component_params"], posterior_dim, value, direction)

                    if run_id == 1:
                        save_adjusted_posterior_config(
                            posterior_kwargs,
                            folder=regular_posteriors_folder,
                            dim_value=value
                        )
                elif varying_attribute == "dimension":
                    adjust_dimension_of_kwargs(posterior_type, posterior_kwargs_original, posterior_kwargs, target_dim=value, required_parameters=required_parameters)

                    posterior_dim = get_posterior_dim(posterior_type, posterior_kwargs)
                    #base_delta = 4
                    #r = base_delta * np.sqrt(posterior_dim)
                    r = 12

                    direction = np.array(direction_dict.get(value))
                    adjust_mode_means(posterior_kwargs["component_params"], posterior_dim, r, direction)
                        
                    if run_id == 1:
                            save_adjusted_posterior_config(
                                posterior_kwargs,
                                folder=regular_posteriors_folder,
                                dim_value=value
                            )
                elif varying_attribute == "circle_radius":
                    posterior_kwargs["component_params"], \
                    posterior_kwargs["component_types"], \
                    posterior_kwargs["weights"] = adjust_circle_layout(
                                                                        posterior_kwargs["component_number"],
                                                                        posterior_kwargs["component_type"],
                                                                        value,
                                                                        posterior_kwargs[cov_param_key],
                                                                        posterior_kwargs["weight_type"]
                                                                        )
                    if run_id == 1:
                        save_adjusted_posterior_config(
                            posterior_kwargs,
                            folder=regular_posteriors_folder,
                            dim_value=value
                        )
                elif varying_attribute == "circle_modes":
                    posterior_kwargs["component_params"], \
                    posterior_kwargs["component_types"], \
                    posterior_kwargs["weights"] = adjust_circle_layout(
                                                                        value,
                                                                        posterior_kwargs["component_type"],
                                                                        posterior_kwargs["radius"],
                                                                        posterior_kwargs[cov_param_key],
                                                                        posterior_kwargs["weight_type"]
                                                                    )
                    if run_id == 1:
                        save_adjusted_posterior_config(
                            posterior_kwargs,
                            folder=regular_posteriors_folder,
                            dim_value=value
                        )
                elif varying_attribute == "init_scheme":
                    init_scheme = value
                elif varying_attribute == "correlation":
                    posterior_dim = get_posterior_dim(posterior_type, posterior_kwargs)

                    for i in range(len(posterior_kwargs["component_params"])):
                        
                        posterior_kwargs["component_params"][i][cov_param_key] = build_correlation_cov_matrix(posterior_dim, value)
 
                    if run_id == 1:
                        save_adjusted_posterior_config(
                                posterior_kwargs,
                                folder=regular_posteriors_folder,
                                dim_value=value
                        )
                elif varying_attribute == "num_samples":
                    num_samples = value
                elif varying_attribute == "num_chains":
                    num_chains = value
                else:

                    if component_index is None:
                        target_indices = range(len(posterior_kwargs["component_params"]))
                    else:
                        target_indices = [component_index]

                    for i in target_indices:
                        posterior_kwargs["component_params"][i][varying_attribute] = value

            else:

                is_studentt = posterior_type == "MvStudentT"
                cov_param_key = "scale" if is_studentt else "cov"

                # Handle parameter changes for single posteriors
                if varying_attribute == "dimension":
                    adjust_dimension_of_kwargs(posterior_type, posterior_kwargs_original, posterior_kwargs, target_dim=value, required_parameters=required_parameters)
                    if run_id == 1:
                            save_adjusted_posterior_config(
                                posterior_kwargs,
                                folder=regular_posteriors_folder,
                                dim_value=value
                            )                     
                elif varying_attribute == "init_scheme":
                    init_scheme = value
                elif varying_attribute == "correlation":
                    posterior_dim = get_posterior_dim(posterior_type, posterior_kwargs)

                    posterior_kwargs[cov_param_key] = build_correlation_cov_matrix(posterior_dim, value)

                    if run_id == 1:
                        save_adjusted_posterior_config(
                            posterior_kwargs,
                            folder=regular_posteriors_folder,
                            dim_value=value
                        )
                elif varying_attribute == "num_samples":
                    num_samples = value
                elif varying_attribute == "num_chains":
                    num_chains = value
                else:
                    # Vary only the specific parameter
                    posterior_kwargs[varying_attribute] = value

            # Ensure num_samples is normalized in case of varying num_chains or num_samples
            samples_per_chain = num_samples // num_chains
            num_samples = samples_per_chain*num_chains
            
            # base_posterior used for all samplers but SMC
            if posterior_type == "Mixture":
                base_posterior = MixturePosterior(
                    component_types=posterior_kwargs["component_types"],
                    component_params=posterior_kwargs["component_params"],
                    weights=posterior_kwargs["weights"],
                )
            elif posterior_type == "Custom":
                logp_func = posterior_kwargs["logp_func"]
                base_posterior = CustomPosterior(logp_func=logp_func)
            else:
                base_posterior = SinglePosterior(dist_name=posterior_type, dist_params=posterior_kwargs)

            # Get IID samples for the current varying value
            if posterior_type != "Custom" and varying_attribute not in ["init_scheme", "num_chains"]:
                iid_batches = iid_batches_dict[value]
            elif posterior_type == "Custom":
                iid_batches = None

            means = None
            init_pooled = None
            init_chain = None
  
            if init_scheme is not None:
                    means = extract_means_from_posterior(posterior_type, posterior_kwargs)
                    init_pooled = get_initvals(init_scheme, means, pooled_eval_mode, num_chains, rng, run_id, init_value_folder, png_folder_init_pooled, varying_attribute, value, iid_batch=iid_batches[0])
                    init_chain = get_initvals(init_scheme, means, chain_eval_mode, 1, rng, run_id, init_value_folder, png_folder_init_chain, varying_attribute, value, iid_batch=iid_batches[0])


            # Run sampling for all samplers
            for sampler_name in samplers:

                use_smc = sampler_name == "SMC"

                 # Reuse model if not SMC
                if not use_smc:
                    posterior = base_posterior
                else:
                     # Rebuild for SMC 
                    if np.isscalar(means[0]):
                            # shape (n_modes, 1)
                            means_array = np.array(means)[:, None] 
                    else:
                            means_array = np.array(means)

                    if posterior_type == "Mixture":
                        #compute higher and lower bound for init prior
                        low, high,_,_,_  = get_uniform_prior_bounds(means_array=means_array, iid_samples=iid_batches[0], quantile_mass=0.9999, expansion_factor=1.0)   
                        posterior = MixturePosterior(
                            component_types=posterior_kwargs["component_types"],
                            component_params=posterior_kwargs["component_params"],
                            weights=posterior_kwargs["weights"],
                            use_smc=True,
                            low=low,
                            high=high
                        )         

                    else :
                        # compute higher and lower bound for init prior
                        low, high, _,_,_ = get_uniform_prior_bounds(means_array=means_array, iid_samples=iid_batches[0], quantile_mass=0.9999, expansion_factor=1.0)
                        posterior = SinglePosterior(dist_name=posterior_type, dist_params=posterior_kwargs, use_smc=True, low=low, high=high)


                pooled_seed = int(run_rng.integers(1_000_000))

                # **Measure Computation Time**
                start_time = time.time()

                pooled_trace = posterior.run_sampling(
                    sampler_name=sampler_name,
                    num_samples=samples_per_chain,
                    num_chains=num_chains, 
                    eval_mode=pooled_eval_mode,
                    initvals=init_pooled, 
                    run_id=run_id, 
                    plot_first_sample=plot_first_sample,
                    init_folder=init_value_folder, 
                    value=value, 
                    means=means, 
                    posterior_type=posterior_type, 
                    run_random_seed=pooled_seed
                )


                # 2*runs have already been used for iid vs iid comparison
                fresh_iid_index = num_iid_vs_iid_batches + run_id-1
                iid_batch = ensure_2d(iid_batches[fresh_iid_index]) 


                end_time = time.time()
                pooled_runtime = end_time - start_time


                eval_trace(trace=pooled_trace, runtime=pooled_runtime, eval_level="pooled", run_id=run_id, sampler_name=sampler_name, value=value,
                           posterior_type=posterior_type, posterior_kwargs=posterior_kwargs, iid_batch=iid_batch,
                           experiment_settings=experiment_settings, folders={ "var_attr_folder": var_attr_folder,"png_folder_traces": png_folder_traces, "png_folder_scatter": png_folder_scatter},
                           varying_attribute=varying_attribute, results=results, do_mmd=do_mmd, do_mmd_rff=do_mmd_rff, save_pairwise_scatter=save_pairwise_scatter, rng=run_rng)


                chain_seed = int(run_rng.integers(1_000_000))

                # **Measure Computation Time**
                start_time = time.time()

                chain_trace = posterior.run_sampling(
                    sampler_name=sampler_name,
                    num_samples=num_samples,
                    num_chains=1,
                    eval_mode=chain_eval_mode, 
                    initvals=init_chain, 
                    run_id=run_id, 
                    plot_first_sample=plot_first_sample,
                    init_folder=init_value_folder, 
                    value=value, 
                    means=means, 
                    posterior_type=posterior_type, 
                    run_random_seed=chain_seed
                )

                end_time = time.time()
                chain_runtime = end_time - start_time

                eval_trace(trace=chain_trace, runtime=chain_runtime, eval_level="chain", run_id=run_id, sampler_name=sampler_name, value=value,
                        posterior_type=posterior_type, posterior_kwargs=posterior_kwargs, iid_batch=iid_batch,
                        experiment_settings=experiment_settings, folders={ "var_attr_folder": var_attr_folder, "png_folder_traces": png_folder_traces, "png_folder_scatter": png_folder_scatter},
                        varying_attribute=varying_attribute, results=results, do_mmd=do_mmd, do_mmd_rff=do_mmd_rff, save_pairwise_scatter=save_pairwise_scatter, rng=run_rng)
                
            # Now increments the TQDM progress bar if it's provided
            if progress_bar is not None:
                progress_bar.update(1)

        # Convert results to DataFrame and save
        df_results = pd.DataFrame(results)

        var_attr_is_tuple = False

        # Handle tuple-based attributes consistently
        if isinstance(df_results[varying_attribute].iloc[0], tuple):
            var_attr_is_tuple = True
            df_results[varying_attribute] = df_results[varying_attribute].apply(str)

        df_results = df_results.sort_values(varying_attribute, ascending=True)

        df_pooled = df_results.query("eval_level == 'pooled'").copy()
        df_chain  = df_results.query("eval_level == 'chain'").copy()

        if experiment_settings.get("save_plots_and_csv_per_run", False):

            # pooled
            plot_and_save_all_metrics(
                df_results=df_pooled,
                sampler_colors=sampler_colors,
                varying_attribute=varying_attribute,
                varying_attribute_for_plot=varying_attribute,
                csv_folder=csv_pool,
                plots_folder=plot_pool,
                run_id=run_id,
                config_descr=f"{config_descr}_pooled"
            )

            # chain
            plot_and_save_all_metrics(
                df_results=df_chain,
                sampler_colors=sampler_colors,
                varying_attribute=varying_attribute,
                varying_attribute_for_plot=varying_attribute,
                csv_folder=csv_chain,
                plots_folder=plot_chain,
                run_id=run_id,
                config_descr=f"{config_descr}_chain"
            )

        df_all_runs.append(df_results)

    logger.info("All runs completed successfully.")

    # ===== GLOBAL RESULTS FOLDER =====
    global_folder = os.path.join(group_folder, "global_results")
    create_directories(global_folder)
    
    # Combine all results into a single data frame 
    df_all_runs = pd.concat(df_all_runs, ignore_index=True)

    df_pooled = df_all_runs[df_all_runs["eval_level"] == "pooled"]   
    df_chain = df_all_runs[df_all_runs["eval_level"] == "chain"]


    if var_attr_is_tuple:
        iid_ref_stats_dict = {str(k): v for k, v in iid_ref_stats_dict.items()}

    pooled_results_folder = os.path.join(global_folder, "pooled_results")
    pooled_plots_folder   = os.path.join(global_folder, "pooled_plots")
    png_folder_pooled = os.path.join(png_folder, "pooled_global_plots")
    create_directories(pooled_results_folder, pooled_plots_folder, png_folder_pooled)

    save_extra_scatter = experiment_settings.get("save_extra_scatter", False)
    log_scaled_plots = experiment_settings.get("log_scaled_plots", False)
    
    compute_and_save_global_metrics(
        df_all_runs=df_pooled,
        sampler_colors=sampler_colors,
        varying_attribute=varying_attribute,
        varying_values=varying_values,
        runs=runs,
        num_chains=num_chains,
        config_descr=config_descr + "_pooled",
        global_results_folder=pooled_results_folder,
        global_plots_folder=pooled_plots_folder,
        png_folder=png_folder_pooled,
        iid_ref_stats_dict=iid_ref_stats_dict,
        save_extra_scatter=save_extra_scatter,
        do_mmd=do_mmd,
        do_mmd_rff=do_mmd_rff,
        log_scaled_plots= log_scaled_plots
    )

    chain_results_folder = os.path.join(global_folder, "chain_results")
    chain_plots_folder   = os.path.join(global_folder, "chain_plots")
    png_folder_chain = os.path.join(png_folder, "chain_global_plots")
    create_directories(chain_results_folder, chain_plots_folder, png_folder_chain) 

    compute_and_save_global_metrics(
        df_all_runs=df_chain,
        sampler_colors=sampler_colors,
        varying_attribute=varying_attribute,
        varying_values=varying_values,
        runs=runs,
        num_chains=num_chains,
        config_descr=config_descr + "_chain",
        global_results_folder= chain_results_folder,
        global_plots_folder= chain_plots_folder,
        png_folder=png_folder_chain,
        iid_ref_stats_dict=iid_ref_stats_dict,
        save_extra_scatter= save_extra_scatter,
        do_mmd=do_mmd,
        do_mmd_rff=do_mmd_rff,
        log_scaled_plots= log_scaled_plots
    )

    logger.info(f"===== Config {config_descr} completed successfully. =====")

