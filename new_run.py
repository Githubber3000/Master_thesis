import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import seaborn as sns
import scipy.stats as sp
from scipy.spatial.distance import mahalanobis
from itertools import combinations
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import json
import copy
import yaml
import logging
import warnings
import sys
import os
import shutil 
import subprocess
import traceback
import time
from datetime import datetime
import humanize 
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

logging.getLogger("arviz").setLevel(logging.CRITICAL)

logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)


def set_logging_level(level_name):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(level)

    # remove all existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    handler= logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)
    

def get_git_tag():
        try:
            tag = subprocess.check_output(["git", "describe", "--tags"], stderr=subprocess.DEVNULL).strip().decode()
            return tag
        except subprocess.CalledProcessError:
            return "No tag found"
        
def create_directories(*paths):
    """Creates multiple directories if they don't exist."""
    for path in paths:
        os.makedirs(path)

def load_config_file(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    group_name = data["group_name"]
    configs = data["configs"]

    for cfg in configs:
         if "varying_values" in cfg:
            cfg["varying_values"] = [
                tuple(v) if isinstance(v, list) else v
                for v in cfg["varying_values"]
            ]
    return group_name, configs

def load_experiment_settings(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def get_experiment_paths(group_names, base_dir):
    return [os.path.join(base_dir, f"{name}.yaml") for name in group_names]

def load_default_values(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["defaults"]

def apply_defaults_to_config(config, defaults):
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    return config

def safe_json_dump(obj, path):
    def convert_numpy(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.generic):
            return o.item()
        return o

    with open(path, "w") as f:
        json.dump(obj, f, indent=4, default=convert_numpy)

def ensure_2d(arr):
    """Ensures array shape is (N, d), even if 1D."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    else:
        return arr.reshape(-1, arr.shape[-1])

def save_adjusted_posterior_config(posterior_kwargs, folder, dim_value):
    json_path = os.path.join(folder, f"posterior_config_dim_{dim_value}.json")
    safe_json_dump(posterior_kwargs, json_path)
    

def adjust_dimension_of_kwargs(posterior_type, kwargs_dict_copy, kwargs_dict, target_dim, required_parameters):
    """
    Adjusts only the required vector-like entries in the dictionary to match the given dimension.
    For mixtures, this is applied recursively to each component.

    Parameters:
    - posterior_type: str (e.g. "Normal", "MvNormal", "Mixture")
    - kwargs_dict: dict to be modified in-place
    - target_dim: int, desired length of vector-like parameters
    - required_parameters: dict mapping posterior types to required param keys
    """

    if posterior_type == "Mixture":
        # Recursive call for each component
        component_types = kwargs_dict["component_types"]
        component_params = kwargs_dict["component_params"]
        component_params_copy = kwargs_dict_copy["component_params"]

        for i, comp_type in enumerate(component_types):
            adjust_dimension_of_kwargs(
                posterior_type=comp_type,
                kwargs_dict_copy= component_params_copy[i],
                kwargs_dict=component_params[i],
                target_dim=target_dim,
                required_parameters=required_parameters
            )
        return 

    # Get the required keys for the current posterior type
    required_keys = required_parameters.get(posterior_type, [])

    for key in required_keys:
        # get the original paramter dims 
        value = kwargs_dict_copy.get(key)

        # Skip missing keys
        if value is None:
            continue

        if target_dim >= 2 and not isinstance(value, list):
            raise ValueError(
                f"Parameter '{key}' must be a list when varying dimension ≥ 2 (got scalar: {value})."
            )

        if isinstance(value, list):
            if all(isinstance(v, (int, float)) for v in value):
                # e.g., mu: [1.0, 2.0, 3.0, 4.0] → [1.0, 2.0]
                # Check length before trimming
                if len(value) < target_dim:
                    raise ValueError(f"'{key}' too short: expected ≥{target_dim}, got {len(value)}")
                kwargs_dict[key] = value[:target_dim]
            elif all(isinstance(v, list) and all(isinstance(x, (int, float)) for x in v) for v in value):
                # e.g., cov: 5x5 matrix → 3x3 matrix
                # Check matrix size before trimming
                if len(value) < target_dim or any(len(row) < target_dim for row in value[:target_dim]):
                    raise ValueError(f"'{key}' matrix too small for target_dim={target_dim}")
                trimmed_matrix = [row[:target_dim] for row in value[:target_dim]]
                kwargs_dict[key] = trimmed_matrix


def get_scalar_rhat_and_ess(trace):
    posterior_vars = [v for v in trace.posterior.data_vars if v.startswith("posterior")]
    if not posterior_vars:
        raise ValueError("No posterior variables found.")
    return (
        az.rhat(trace, var_names=posterior_vars).to_array().max().item(),
        az.ess(trace, var_names=posterior_vars).to_array().min().item()
    )

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

    print("high:", high, "low:", low)    
    return low, high, min_mode, max_mode, border

def get_logp_func(weights, components):
    def logp_func(x):
        logps = [pt.log(w) + pm.logp(comp, x) for w, comp in zip(weights, components)]
        return pm.math.logsumexp(pt.stack(logps))
    return logp_func


def get_posterior_dim(posterior_type, params):
    """
    Robustly determines the dimensionality of a posterior from its parameters.
    """
    if posterior_type == "Mixture":
        # Check only the first component (assuming all have same dimension)
        comp_type = params["component_types"][0]
        comp_params = params["component_params"][0]
        return get_posterior_dim(comp_type, comp_params)

    if "mu" in params:
        mu = np.array(params["mu"])
        return mu.shape[0] if mu.ndim > 0 else 1
    elif "loc" in params:
        loc = np.array(params["loc"])
        return loc.shape[0] if loc.ndim > 0 else 1
    elif posterior_type == "Cauchy" and "alpha" in params:
        alpha = np.array(params["alpha"])
        return alpha.shape[0] if alpha.ndim > 0 else 1
    elif posterior_type == "Beta":
        a = np.array(params["a"])
        return a.shape[0] if a.ndim > 0 else 1
    elif posterior_type == "MvNormal" and "mu" in params:
        return len(params["mu"])
    else:
        raise ValueError(f"Cannot determine dimensionality for posterior type '{posterior_type}' with parameters: {params}")
    

def validate_config(config):
    """Checks if the config correctly defines one varying attribute and all other attributes are fixed."""
    
    REQUIRED_ATTRIBUTES = {
    "config_descr",
    "posterior_type",
    "runs",
    "num_samples",
    "num_chains",
    "varying_attribute",
    "varying_values",
    }

    # Posterior-specific required attributes
    POSTERIOR_ATTRIBUTES = {
        "Cauchy": {"alpha", "beta"},
        "Beta": {"a", "b"},
        "Normal": {"mu", "sigma"},
        "SkewNormal": {"mu", "sigma", "alpha"},
        "StudentT": {"nu", "mu", "sigma"},
        "Laplace": {"mu", "b"},
        "SkewStudentT": {"a", "b", "mu", "sigma"},
        "Mixture": {"component_types", "component_params", "weights"},
        "MvNormal": {"mu", "cov"},
        "Custom": {"logp_func"}
    }

    OPTIONAL_ATTRIBUTES = {"base_random_seed", "init_scheme", "varying_component", "dimension_corr"}

    if "config_descr" not in config:
        raise ValueError("Config is missing 'config_descr'.")
    
    config_descr = config["config_descr"]

    if "varying_attribute" not in config:
        raise ValueError(f"Config '{config_descr}' is missing 'varying_attribute'.")
    
    varying_attr = config["varying_attribute"]

    # Ensure all required attributes are present
    missing_attrs = REQUIRED_ATTRIBUTES - config.keys() - {varying_attr}

    if missing_attrs:
        raise ValueError(f"Config '{config_descr}' is missing required attributes: {missing_attrs}.")
    
    posterior_type = config["posterior_type"]

    if posterior_type not in POSTERIOR_ATTRIBUTES:
        raise ValueError(f"Config '{config_descr}' has an invalid 'posterior_type': '{posterior_type}'.")

    if posterior_type == "Mixture" and "varying_component" in config:
        varying_index = config["varying_component"]
        varying_component = config["component_types"][varying_index]
        all_valid_attributes = REQUIRED_ATTRIBUTES.union(POSTERIOR_ATTRIBUTES[posterior_type], POSTERIOR_ATTRIBUTES[varying_component], OPTIONAL_ATTRIBUTES)
        
    else:
        # Ensure varying_attribute is a recognized attribute
        all_valid_attributes = REQUIRED_ATTRIBUTES.union(POSTERIOR_ATTRIBUTES[posterior_type], OPTIONAL_ATTRIBUTES)

    if varying_attr not in all_valid_attributes:
        raise ValueError(f"Config '{config_descr}' has an invalid 'varying_attribute': '{varying_attr}'.")
    
    if varying_attr == "dimension_corr":
        max_dim = max(config["varying_values"])

        for key in ["mu", "sigma", "cov"]: 
            val = config.get(key)

            if val is None:
                continue

            if max_dim >= 2 and not isinstance(val, list):
                raise ValueError(
                    f"Parameter '{key}' in config '{config['config_descr']}' must be a list when varying dimension ≥ 2 (got scalar: {val})."
                )

            if isinstance(val, list) and len(val) < max_dim:
                raise ValueError(f"Parameter '{key}' in config '{config['config_descr']}' is too short for max dimension {max_dim}")

        
    if posterior_type == "Mixture" and varying_attr not in ("num_samples", "num_chains", "init_scheme", "weights", "dimension_corr"):
        if "varying_component" not in config:
            raise ValueError(
                f"Config '{config_descr}' must have 'varying_component' defined "
                f"when varying '{varying_attr}' for a Mixture."
            )
        
    vc = config.get("varying_component")    
    if vc is not None and not (0 <= vc < len(config["component_types"])):
        raise ValueError(
            f"Config '{config_descr}' has invalid 'varying_component' index {vc}, "
            f"but 'component_types' has length {len(config['component_types'])}."
        )
    
    VALID_INIT_SCHEMES = {"equal_per_mode","all_in_middle", "all_near_mode", "thesis_scheme", "None"} 

    if "init_scheme" in config:
        if config["init_scheme"] not in VALID_INIT_SCHEMES and not config["init_scheme"].startswith("all_near_mode_"):
            raise ValueError(
                f"Config '{config_descr}' has invalid 'init_scheme': "
                f"'{config['init_scheme']}'. Must be one of {VALID_INIT_SCHEMES} "
                "or 'all_near_mode_<int>'."
            )


def generate_html_report(experiment_root_folder, experiments, output_path):
    """
    Generates a single HTML report for the entire experiment (all groups and configs).
    """

    template_path = "."  
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template("report.html")

    metrics = ["wasserstein_distance", "mmd_rff", "r_hat", "ess", "runtime"]

    groups_data = []

    for group_name, configs in experiments:
        config_entries = []

        for config in configs:
            config_descr = config["config_descr"]
            png_base_path = os.path.join("report_pngs", group_name, config_descr)

            # Load metadata
            metadata_path = os.path.join(
                experiment_root_folder, "results", group_name, config_descr, f"metadata_config_{config_descr}.json"
            )
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            entry = {
                "config_descr": config_descr,
                "posterior_type": metadata.get("posterior_type"),
                "varying_attribute": metadata.get("varying_attribute"),
                "runs": metadata.get("runs"),
                "git_tag": metadata.get("git_tag"),
                "metrics": metrics,
                "metric_plot_paths": {
                    metric: os.path.join(png_base_path, f"{metric}_global_plot_shaded.png")
                    for metric in metrics
                },
                "glass_plot_paths": {
                    "ws": os.path.join(png_base_path, "glass_delta_ws_dist.png"),
                    "mmd": os.path.join(png_base_path, "glass_delta_mmd.png"),
                }
            }

            config_entries.append(entry)

        groups_data.append({
            "name": group_name,
            "configs": config_entries
        })

    # Render HTML
    html = template.render(
        experiment_name=os.path.basename(experiment_root_folder),
        groups=groups_data,
        metrics=metrics
    )

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Experiment-level HTML report saved to: {output_path}")


def plot_and_save_all_metrics(df_results, sampler_colors, varying_attribute, varying_attribute_for_plot, csv_folder, plots_folder, run_id, config_descr):
    """
    Generates and saves multiple metric plots for different samplers.

    Parameters:
    - df_results: DataFrame containing experiment results.
    - sampler_colors: Dictionary mapping sampler names to colors.
    - varying_attribute: The attribute that varies.
    - varying_attribute_for_plot: The attribute used for plotting.
    - plots_folder: Folder where plots should be saved.
    - run_id: ID of the current run.
    - config_descr: Description of the configuration.
    """
    
    # Define metric labels
    metrics = ["wasserstein_distance","r_hat", "ess", "runtime"]

    # Initialize plots for all metrics
    fig_ax_pairs = {key: plt.subplots(figsize=(10, 6)) for key in metrics}

    # Iterate over samplers and plot all metrics
    for sampler in df_results["sampler"].unique():
        df_sampler = df_results[df_results["sampler"] == sampler]
        csv_filename = os.path.join(csv_folder, f"{sampler}_results.csv")
        df_sampler.to_csv(csv_filename, index=False)

        for metric in metrics:
            fig, ax = fig_ax_pairs[metric]
            ax.plot(df_sampler[varying_attribute_for_plot], df_sampler[metric], 
                    marker="o", linestyle="-", label=sampler, 
                    color=sampler_colors.get(sampler, "black"))

    # Set dynamic axis labels and save plots
    attribute_label = varying_attribute.replace("_", " ").title()

    for metric in metrics:
        fig, ax = fig_ax_pairs[metric]
        finalize_and_save_plot(fig,ax, attribute_label, metric, 
                               f"{metric} for Samplers (config =_{config_descr})",
                               os.path.join(plots_folder, f"{metric}_run_{run_id}.pdf"))
        

def compute_and_save_global_metrics(df_all_runs, sampler_colors, varying_attribute, varying_values, runs, config_descr, global_results_folder, global_plots_folder, png_folder, iid_ref_stats_dict):
    """
    Computes and saves global metric plots (averaged across runs) for different samplers.

    Parameters:
    - df_all_runs: DataFrame containing results from all runs.
    - sampler_colors: Dictionary mapping sampler names to colors.
    - varying_attribute: The attribute that varies.
    - runs: Number of experiment runs.
    - config_descr: Configuration description.
    - global_results_folder: Folder to save CSVs.
    - global_plots_folder: Folder to save plots.
    """

    # Define metrics for aggregation
    metrics = ["wasserstein_distance", "mmd_rff","r_hat", "ess", "runtime"]

    # New figure set (line + fill)
    fig_ax_pairs_shaded = {metric: plt.subplots(figsize=(10, 6)) for metric in metrics}
    fig_g, ax_g = plt.subplots(figsize=(10, 6))  # Glass delta for wasserstein_distance
    fig_g_mmd, ax_g_mmd = plt.subplots(figsize=(10, 6))  # Glass delta for mmd

    global_avg_dfs = {}

    # Load IID reference statistics
    iid_means_dict_swd = {}
    iid_stds_dict_swd = {}
    iid_medians_dict_swd = {}
    iid_q25_dict_swd = {}
    iid_q75_dict_swd = {}
    iid_means_dict_mmd = {}
    iid_stds_dict_mmd = {}
    iid_medians_dict_mmd = {}
    iid_q25_dict_mmd = {}
    iid_q75_dict_mmd = {}

    for key in df_all_runs[varying_attribute].unique():
        k = tuple(key) if isinstance(key, np.ndarray) else key
        iid_entry = iid_ref_stats_dict.get(k)
        if iid_entry is None:
            raise KeyError(f"Missing IID reference stats for varying attribute value: {k}")
        iid_means_dict_swd[k] = iid_entry["mean_swd"]
        iid_stds_dict_swd[k] = iid_entry["std_swd"]
        iid_medians_dict_swd[k] = iid_entry["median_swd"]
        iid_q25_dict_swd[k] = iid_entry["q25_swd"]
        iid_q75_dict_swd[k] = iid_entry["q75_swd"]
        iid_means_dict_mmd[k] = iid_entry["mean_mmd"]
        iid_stds_dict_mmd[k] = iid_entry["std_mmd"]
        iid_medians_dict_mmd[k] = iid_entry["median_mmd"]
        iid_q25_dict_mmd[k] = iid_entry["q25_mmd"]
        iid_q75_dict_mmd[k] = iid_entry["q75_mmd"]


    for metric in metrics:
        fig_shaded, ax_shaded = fig_ax_pairs_shaded[metric]

        # For each sampler, plot its line for this metric
        for sampler in df_all_runs["sampler"].unique():
            df_sampler = df_all_runs[df_all_runs["sampler"] == sampler]
            color = sampler_colors.get(sampler, "black")

            # Pivot: rows = varying_attribute, columns = run_id, values = metric
            df_pivot = df_sampler.pivot_table(
                index=varying_attribute, columns="run_id", values=metric
            )

            if df_pivot.empty or df_pivot.shape[1] == 0:
                print(f"No data for sampler '{sampler}' and metric '{metric}' – skipping.")
                ax_shaded.annotate("⚠️ 'DEMetropolis' r-hat skipped due to invalid values", 
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha="right", va="bottom", fontsize=9, color="red")
                continue

            if metric == "r_hat":
                if df_pivot.isnull().values.any() or  (df_pivot > 1000).any().any():
                    logger.warning(f"Skipping r_hat plot for sampler {sampler} due to extremely high values.")                    
                    ax_shaded.annotate("⚠️ 'DEMetropolis' r-hat skipped due to >1000", 
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha="right", va="bottom", fontsize=9, color="red")
                    continue
                
            # Compute mean+std and median+quantiles 
            means = df_pivot.mean(axis=1)
            stds = df_pivot.std(axis=1)
            medians = df_pivot.median(axis=1)
            q25 = df_pivot.quantile(0.25, axis=1)
            q75 = df_pivot.quantile(0.75, axis=1)

            # Custom ordering based on config (only if needed)
            if isinstance(medians.index[0], str): 
                custom_order = [str(t) for t in varying_values]
                medians = medians.reindex(custom_order)
                q25 = q25.reindex(custom_order)
                q75 = q75.reindex(custom_order)
                means = means.reindex(custom_order)
                stds = stds.reindex(custom_order)

            # Plot median line
            ax_shaded.plot(medians.index, medians, "o-", label=sampler, color=color)

            # Plot uncertainty: interquartile range (q25–q75)
            if len(medians.index) > 1:
                ax_shaded.fill_between(medians.index, q25, q75, color=color, alpha=0.2)
            else:
                lower_err = medians - q25
                upper_err = q75 - medians
                yerr = [lower_err, upper_err]
                ax_shaded.errorbar(medians.index, medians, yerr=yerr, fmt="o", color=color, capsize=5)

            # Save global avg for CSV
            if sampler not in global_avg_dfs:
                global_avg_dfs[sampler] = {}
            global_avg_dfs[sampler][metric] = (medians, q25, q75)


            # Compute glass delta for wasserstein_distance only
            if metric == "wasserstein_distance":
                # Get IID mean and std for this varying attribute value
                iid_means_swd = np.array([iid_means_dict_swd[k] for k in means.index])
                iid_stds_swd = np.array([iid_stds_dict_swd[k] for k in means.index])
            
                # Avoid zero in denominator
                iid_stds_safe = np.where(iid_stds_swd == 0, np.nan, iid_stds_swd)

                # Compute glass delta
                glass_delta = (means.values - iid_means_swd) / iid_stds_safe
        
                global_avg_dfs[sampler]["ws_dist_glass_delta"] = glass_delta

                # Plot glass delta for this sampler
                ax_g.plot(means.index, glass_delta, "o-", label=sampler, color=color)
            
            elif metric == "mmd_rff":
                # Get IID mean and std for this varying attribute value
                iid_means_mmd = np.array([iid_means_dict_mmd[k] for k in medians.index])
                iid_stds_mmd = np.array([iid_stds_dict_mmd[k] for k in medians.index])

                # Avoid zero in denominator
                iid_stds_safe = np.where(iid_stds_mmd == 0, np.nan, iid_stds_mmd)

                # Compute glass delta
                glass_delta_mmd = (means.values - iid_means_mmd) / iid_stds_safe

                global_avg_dfs[sampler]["mmd_rff_glass_delta"] = glass_delta_mmd

                # Plot glass delta for this sampler
                ax_g_mmd.plot(means.index, glass_delta_mmd, "o-", label=sampler, color=color)


        # Only for wasserstein_distance and mmd: Plot IID baseline once
        if metric == "wasserstein_distance":
            
            iid_medians = np.array([iid_medians_dict_swd[k] for k in medians.index])
            iid_q25 = np.array([iid_q25_dict_swd[k] for k in medians.index])
            iid_q75 = np.array([iid_q75_dict_swd[k] for k in medians.index])

            ax_shaded.plot(medians.index, iid_medians, "o--", label="IID Reference", color="black")
            ax_shaded.fill_between(
                medians.index,
                iid_q25,
                iid_q75,
                color="black",
                alpha=0.1,
            )

        elif metric == "mmd_rff":

            iid_medians = np.array([iid_medians_dict_mmd[k] for k in medians.index])
            iid_q25 = np.array([iid_q25_dict_mmd[k] for k in medians.index])
            iid_q75 = np.array([iid_q75_dict_mmd[k] for k in medians.index])

            ax_shaded.plot(medians.index, iid_medians, "o--", label="IID Reference", color="black")
            ax_shaded.fill_between(
                medians.index,
                iid_q25,
                iid_q75,
                color="black",
                alpha=0.1,
            )

    # Save Global Averages per Sampler to CSV
    for sampler, metrics_dict in global_avg_dfs.items():
        # Fill missing metrics with NaNs so CSV is complete
        for m in metrics:
            if m not in metrics_dict:
                nan_series = pd.Series(np.nan, index=metrics_dict["wasserstein_distance"][0].index)
                metrics_dict[m] = (nan_series, nan_series, nan_series)

        df_global_avg = pd.DataFrame({
            varying_attribute: metrics_dict["wasserstein_distance"][0].index,
            **{f"global_median_{metric}": metrics_dict[metric][0].values for metric in metrics},
            **{f"global_q25_{metric}": metrics_dict[metric][1].values for metric in metrics},
            **{f"global_q75_{metric}": metrics_dict[metric][2].values for metric in metrics},
        })

        if "ws_dist_glass_delta" in metrics_dict:
            df_global_avg["ws_dist_glass_delta"] = metrics_dict["ws_dist_glass_delta"]
        if "mmd_rff_glass_delta" in metrics_dict:
            df_global_avg["mmd_rff_glass_delta"] = metrics_dict["mmd_rff_glass_delta"]

        csv_filename = os.path.join(global_results_folder, f"Global_results_{sampler}.csv")
        df_global_avg.to_csv(csv_filename, index=False)

    # Save plots
    attribute_label = varying_attribute.replace("_", " ").title()
    for metric in metrics:
      
        fig_shaded, ax_shaded = fig_ax_pairs_shaded[metric]
        pdf_path = os.path.join(global_plots_folder, f"{metric}_global_plot_shaded.pdf")
        png_path = os.path.join(png_folder, f"{metric}_global_plot_shaded.png")

        finalize_and_save_plot(fig_shaded, ax_shaded, attribute_label, metric,
                               f"Averaged {metric.replace('_', ' ').title()} ({runs} Runs, config = {config_descr})",
                                save_path=pdf_path, save_path_png=png_path)
        

    # Plot Glass's Δ for wasserstein_distance
    pdf_path = os.path.join(global_plots_folder, "glass_delta_ws_dist.pdf")
    png_path = os.path.join(png_folder, "glass_delta_ws_dist.png")
    finalize_and_save_plot(fig_g, ax_g, xlabel=attribute_label, ylabel="Glass's Δ", title=f"Glass's Δ for Wasserstein Distance ({runs} Runs, config = {config_descr})",
    save_path=pdf_path, save_path_png=png_path)

    # Plot Glass's Δ for MMD
    pdf_path = os.path.join(global_plots_folder, "glass_delta_mmd.pdf")
    png_path = os.path.join(png_folder, "glass_delta_mmd.png")
    finalize_and_save_plot(fig_g_mmd, ax_g_mmd, xlabel=attribute_label, ylabel="Glass's Δ", title=f"Glass's Δ for MMD-RFF ({runs} Runs, config = {config_descr})",
                           save_path=pdf_path, save_path_png=png_path)  



def finalize_and_save_plot(fig, ax, xlabel, ylabel, title, save_path, save_path_png=None):
    """
    Finalizes the plot with labels, grid, and saves it to a file.
    
    Parameters:
    - fig: Matplotlib figure
    - ax: Matplotlib axis
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - title: Title of the plot
    - save_path: Path to save the figure.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Sampler")
    ax.grid(True)

    # store as pdf
    fig.savefig(save_path, bbox_inches="tight")

    if save_path_png:
    # store as well as png
        fig.savefig(save_path_png, dpi=150, bbox_inches="tight")

    plt.close(fig)



def plot_histogram(samples, title, save_path=None, posterior_type=None):
    """
    Plots a histogram and KDE of the given samples.

    Parameters:
    - samples: 1D or 2D array of samples.
    - title: Title of the plot.
    - save_path: If provided, saves the figure to this path.
    """
    plt.figure(figsize=(8, 6))

    if samples.ndim == 2:
        # Handle multivariate case
        if samples.shape[1] == 2:
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, label="2D Samples")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.title(title)
            plt.legend()
            plt.grid(True)
            
        elif posterior_type == "MvNormal" and samples.shape[1] > 2:
            logger.info(f"Skipping plotting: Multivariate Normal with dimension {samples.shape[1]}.")
            return
        
    else:
        # Standard 1D histogram + KDE
        plt.hist(samples, bins=50, alpha=0.5, density=True, color='blue', edgecolor='black', label="Histogram")
        sns.kdeplot(samples, color='red', lw=2, label="KDE")
        plt.title(title)
        plt.xlabel("Sample Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def handle_trace_plots(trace, sampler_name, varying_attribute, value, save_path=None, show=False, save_individual=False):
    """
    Handles both displaying and saving trace plots.

    Parameters:
    - trace: the ArviZ InferenceData object
    - sampler_name: name of the sampler (e.g. "HMC")
    - varying_attribute: the name of the varying parameter (e.g. "mu")
    - value: the current value of the varying parameter
    - save_path: path to save the full trace plot (if any)
    - show: if True, show plot in notebook
    - save_individual: if True and dim > 1, save individual dim plots
    """
    posterior_array = trace.posterior["posterior"]
    dim = posterior_array.shape[-1] if posterior_array.ndim == 3 else 1

    if posterior_array.ndim == 3 and dim > 1:
        # Plot combined
        fig = az.plot_trace(trace, compact=True)
        if show:
            plt.suptitle(f"Trace Plot ({sampler_name}, {varying_attribute} = {value})")
            plt.tight_layout()
            plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

        # Plot per dimension
        if save_individual or show:
            for i in range(dim):
                dim_i = posterior_array[..., i]
                fig = az.plot_trace({f"posterior_{i}": dim_i})
                title = f"Trace Plot of posterior[{i}] ({sampler_name}, {varying_attribute} = {value})"
                if show:
                    plt.suptitle(title)
                    plt.tight_layout()
                    plt.show()
                if save_path and save_individual:
                    filename = save_path.replace(".pdf", f"_dim_{i}.pdf")
                    plt.suptitle(title)
                    plt.tight_layout()
                    plt.savefig(filename, bbox_inches="tight")
                    plt.close()

    else:
        fig = az.plot_trace(trace, compact=True)
        if show:
            plt.suptitle(f"Trace Plot ({sampler_name}, {varying_attribute} = {value})")
            plt.tight_layout()
            plt.show()
        if save_path:
            plt.suptitle(f"Trace Plot ({sampler_name}, {varying_attribute} = {value})")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()


def extract_means_from_posterior(posterior_type, posterior_kwargs):
    """
    Generalized function to extract central tendency (mean/loc) for initialization.
    - For Mixture: returns list of all component means.
    - For single-posteriors: returns list with one mean value or vector.
    """
    if posterior_type == "Mixture":
        return extract_means_from_components(posterior_type, posterior_kwargs["component_params"])

    elif "mu" in posterior_kwargs:
        return [posterior_kwargs["mu"]]

    elif "loc" in posterior_kwargs:
        return [posterior_kwargs["loc"]]

    elif posterior_type == "Cauchy" and "alpha" in posterior_kwargs:
        return [posterior_kwargs["alpha"]] 

    elif posterior_type == "Beta":
        a = posterior_kwargs["a"]
        b = posterior_kwargs["b"]
        # Expected value
        return [a / (a + b)]  

    else:
        raise ValueError(f"Cannot extract central location (mu or loc) for posterior type '{posterior_type}'.")


def extract_means_from_components(posterior_type, component_params):
    """
    Extracts central tendency (mu or loc) from each component's parameters.
    """
    means = []
    for params in component_params:
        if "mu" in params:
            means.append(params["mu"])
        elif "loc" in params:
            means.append(params["loc"])

        elif posterior_type == "Cauchy" and "alpha" in params:
            means.append(params["alpha"])
            
        elif posterior_type == "Beta":
            a = params["a"]
            b = params["b"]
             # Expected value
            means.append([a / (a + b)]) 
        else:
            raise ValueError("Component missing a central tendency parameter (mu or loc).")
    return means


def get_initvals(init_scheme, means, num_chains, rng=None, run_id=None, init_folder=None, value=None, unimodal_init_margin = 50):
    """Generates initialization values based on the chosen scheme.""" 

    rng = rng or np.random.default_rng()
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
        parent_folder= os.path.join(init_folder, "chain initvals")
        create_directories(parent_folder)
        chain_info_path = os.path.join(parent_folder, f"init_{value}.json")
        chain_info_plot_path = os.path.join(parent_folder, f"init_{value}.pdf")
        save_sample_info(sample_info=init_info, json_path=chain_info_path, plot_path=chain_info_plot_path, label="Init Values", case = init_info["case"])

    logger.debug(f"Generated initvals: {initvals}")
    return initvals


def save_sample_info(sample_info, json_path, plot_path, label="Samples", case=None):
    """
    General utility to save sample info (e.g., init values, warmup samples) as JSON and plot if dim ≤ 2.
    
    Parameters:
    - sample_info: dict containing
        - "samples": list of dicts like [{"posterior": ...}, ...]
        - "means_array": list of means (e.g. from init or components)
        - "dim": int, dimensionality
        - optionally: "low", "high", "case"
    - json_path: path to save JSON info
    - plot_path: path to save the plot
    - label: label for sample points (e.g., "Init Values", "Warmup Samples")
    - case: override case type (for optional bounding box display)
    """

    safe_json_dump(sample_info, json_path)

    dim = sample_info["dim"]
    means_array = np.array(sample_info.get("means_array", []))

    if label == "Init Values":
        samples = np.array([list(v.values())[0] for v in sample_info["samples"]])
    elif label == "Samples":
        samples = np.array(sample_info["samples"])

    if dim > 2:
        return

    fig, ax = plt.subplots(figsize=(8, 2) if dim == 1 else (8, 6))

    # 1D case
    if dim == 1:
        samples_flat = samples.flatten()
        ax.scatter(samples_flat, np.zeros_like(samples_flat), color='blue', label=label, alpha=0.7)
        means_flat = means_array.flatten()
        ax.scatter(means_flat, np.zeros_like(means_flat), color='red', marker='x', s=100, label='Means')

        if case == "multimodal" or case == "unimodal":
            # Handle scalar or list storage
            low = sample_info["low"]
            high = sample_info["high"]

            ax.axvline(low, color="black", linestyle="--", label="Init Box")
            ax.axvline(high, color="black", linestyle="--")  

        ax.set_yticks([])
        ax.set_xlabel("Value")


    # 2D case
    elif dim == 2:
        ax.scatter(samples[:, 0], samples[:, 1], color='blue', label=label, alpha=0.7)
        ax.scatter(means_array[:, 0], means_array[:, 1], color='red', marker='x', s=100, label='Means')
        
        if case == "multimodal" or case == "unimodal":
            low = np.array(sample_info["low"])
            high = np.array(sample_info["high"])

            rect = plt.Rectangle(low, *(high - low), linewidth=1, edgecolor='black',
                                    facecolor='none', linestyle='--', label='Init Box')
            ax.add_patch(rect)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_aspect("equal")


    if label == "Init Values":
        ax.set_title(f"{label} and Means")
    elif label == "Samples":
        sampler = sample_info.get("sampler", "Unknown")
        case = sample_info.get("case", "Unknown")
        ax.set_title(f"First {case} from {sampler}")
    ax.grid(True)

    if dim == 1:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(right=0.75)  
    else:
        ax.legend()  

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)



def compute_mean_mahalanobis_distance(component_params):
    means = []
    covs = []

    for comp in component_params:
        mu = np.atleast_1d(comp["mu"])
        sigma = comp["sigma"]

        if np.isscalar(sigma):
            cov = np.diag([sigma**2] * mu.shape[0])
        else:
            cov = np.atleast_2d(sigma) ** 2  # Ensure covariance matrix

        means.append(mu)
        covs.append(cov)

    pairwise_distances = []
    for (i, j) in combinations(range(len(means)), 2):
        pooled_cov = (covs[i] + covs[j]) / 2
        try:
            inv_cov = np.linalg.inv(pooled_cov)
            d = mahalanobis(means[i], means[j], inv_cov)
            pairwise_distances.append(d)
        except np.linalg.LinAlgError:
            continue  # skip singular cases

    return np.mean(pairwise_distances) if pairwise_distances else 0.0



def sliced_wasserstein_distance(X, Y, L=100):
    """
    Computes the sliced Wasserstein distance (SWD_p) between two sets of samples.
    
    Parameters:
    - X: numpy array of shape (N, d) -> first sample set
    - Y: numpy array of shape (N, d) -> second sample set
    - L: int, number of random projections
    - p: int, order of Wasserstein distance (default: 1)
    
    Returns:
    - SWD_p: float, the sliced Wasserstein distance
    """

    N, d = X.shape  # Assuming X and Y have the same shape
    S = 0  # Accumulation variable

    for _ in range(L):
        # Sample a random unit vector (projection direction)
        theta = np.random.randn(d)
        theta /= np.linalg.norm(theta)  # Normalize to unit sphere

        # Compute projections
        alpha = X @ theta
        beta = Y @ theta

        # Compute 1D Wasserstein distance
        W_i = sp.wasserstein_distance(alpha, beta)

        # Accumulate
        S += W_i

    # Compute final SWD
    SWD_p = (S / L) 

    return SWD_p

def compute_mmd_rff(X, Y, D=500, sigma=1.0, seed=None):
    """
    Computes the approximate Maximum Mean Discrepancy (MMD) using Random Fourier Features (RFF)
    between two sample sets X and Y.

    Parameters:
    - X: np.ndarray of shape (n, d) – sample set from distribution p(x)
    - Y: np.ndarray of shape (m, d) – sample set from distribution q(x)
    - D: int – number of random Fourier features
    - sigma: float – bandwidth of the Gaussian kernel
    - seed: int or None – random seed for reproducibility

    Returns:
    - mmd_rff: float – approximate MMD value
    """
    rng = np.random.default_rng(seed)

    n, d = X.shape
    m, _ = Y.shape

    # Step 1: Generate random frequencies and offsets
    omega = rng.normal(loc=0.0, scale=1.0 / sigma, size=(D, d))
    b = rng.uniform(0, 2 * np.pi, size=D)

    # Step 2: Compute random Fourier features
    def z(x):
        projection = np.dot(x, omega.T) + b
        return np.sqrt(2.0 / D) * np.cos(projection)

    Z_X = z(X)  # shape (n, D)
    Z_Y = z(Y)  # shape (m, D)

    # Step 3: Calculate mean embeddings
    mu_p = Z_X.mean(axis=0)
    mu_q = Z_Y.mean(axis=0)

    # Step 4: Calculate MMD^2 (Euclidean distance between embeddings)
    mmd_rff = np.linalg.norm(mu_p - mu_q)

    return mmd_rff


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
        "MvNormal": lambda p: rng.multivariate_normal(mean=np.array(p["mu"]), cov=np.array(p["cov"]), size=num_samples),
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
            iid_samples = np.empty((num_samples, posterior_dim))  # Multivariate case
        else:
            iid_samples = np.empty(num_samples)

        for i, (comp_type, comp_params) in enumerate(zip(component_types, component_params)):
            mask = chosen_components == i  # Select samples for this component
            num_selected = mask.sum()
            if num_selected > 0:
                if comp_type in scipy_distributions or comp_type == "SkewStudentT":
                    iid_samples[mask] = generate_iid_samples(posterior_type=comp_type, num_samples=num_selected, rng=rng, **comp_params)

        return iid_samples
    
    else:
        raise ValueError(f"Unsupported posterior type: {posterior_type}")


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
    num_chains=None,
    rng=None,
    group_folder=None,
    required_parameters=None
):
    """
    Generates all IID batches for the given posterior type and varying attribute.

    Parameters:
    - posterior_type: Type of the posterior (e.g., "Mixture", "Normal").
    - posterior_kwargs: Dictionary of parameters for the posterior.
    - varying_attribute: The attribute that varies (e.g., "mu", "sigma").
    - varying_values: List of values for the varying attribute.
    - num_total_iid_batches: Total number of IID batches to generate.
    - num_iid_vs_iid_batches: Number of IID vs IID batches.
    - num_samples: Number of samples per batch.
    
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
        # Loop through all varying values for Mixture posterior
        for value in varying_values:
            
            if varying_attribute == "weights":
                    iid_kwargs["weights"] = value
            elif varying_attribute == "dimension_corr":
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
                varying_attribute=varying_attribute,
                posterior_type=posterior_type
            )

    # Single posterior case
    else:
        for value in varying_values:
            
            if varying_attribute == "dimension_corr":
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
                varying_attribute=varying_attribute,
                posterior_type=posterior_type
            )
   
    return iid_batches_dict, iid_ref_stats_dict


def compute_and_store_iid_stats(
    iid_batches,
    value,
    num_iid_vs_iid_batches,
    iid_ref_stats_dict,
    iid_histogram_folder,
    varying_attribute,
    posterior_type
):

    ref_swd_values = []
    ref_mmd_values = []

    # Pairwise comparison for SWD/MMD stats
    for i in range(0, num_iid_vs_iid_batches, 2): 
        x = ensure_2d(iid_batches[i])
        y = ensure_2d(iid_batches[i + 1])
        swd = sliced_wasserstein_distance(x, y)
        mmd_rff = compute_mmd_rff(x, y, D=500, sigma=1.0, seed=None)
        ref_swd_values.append(swd)
        ref_mmd_values.append(mmd_rff)


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
        "q75_mmd": np.quantile(ref_mmd_values, 0.75)
    }

    plot_histogram(
        samples=iid_batches[0],
        title=f"IID Samples Histogram & KDE ({varying_attribute}={value})",
        save_path=os.path.join(
            iid_histogram_folder, f"iid_hist_kde_{varying_attribute}_{value}.pdf"
        ),
        posterior_type=posterior_type
    )


class PosteriorExample:
    """Base class for different posterior types."""
    
    def __init__(self):
        self.model = None  # Placeholder for the PyMC model
    
    def _define_posterior(self):
        """Subclasses should implement this method to define the posterior."""
        raise NotImplementedError("Subclasses must implement _define_posterior()")

    def run_sampling(self, sampler_name, num_samples=2000, tune=1000, num_chains=2, initvals=None, run_id=None, plot_first_sample=None, init_folder=None, value=None, means=None, posterior_type=None, run_random_seed=None):
        """Runs MCMC sampling using the chosen sampler."""

        with self.model:

            if sampler_name == "SMC":
                trace = pm.sample_smc(num_samples, chains=num_chains, progressbar=False, random_seed=run_random_seed)
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

                if initvals != None:
                    trace = pm.sample(num_samples, tune=tune, step=sampler,initvals=initvals, chains=num_chains, return_inferencedata=True, discard_tuned_samples=discard_tuned_samples, progressbar=False, random_seed=run_random_seed)   
                else:
                    trace = pm.sample(num_samples, tune=tune, step=sampler, chains=num_chains, return_inferencedata=True, discard_tuned_samples=discard_tuned_samples, progressbar=False, random_seed=run_random_seed)

                if run_id == 1 and plot_first_sample:
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
        # Retrieve the distribution class from PyMC
        dist_class = getattr(pm, self.dist_name)   
        dist = dist_class.dist(**self.dist_params)
        logp_func = lambda x: pm.logp(dist, x)

        dim = get_posterior_dim(self.dist_name, self.dist_params)
        shape = (dim,) if dim > 1 else ()

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
                components.append(dist_class.dist(**params)) 
        
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

            # Define the custom distribution using pm.CustomDist
            pm.CustomDist("posterior", logp=self.logp_func)

        return model


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
    unimodal_init_margin=None,
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

    # Number of IID bacthes for the IID vs IID comparison
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
    create_directories(group_folder, init_folder, runs_folder)

    if varying_attribute == "dimension_corr":
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
        iid_batches_dict, iid_ref_stats_dict = generate_all_iid_batches(
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
            required_parameters=required_parameters 
        )       

    # Define fixed colors for each sampler
    sampler_colors = {
        "Metro": "blue",
        "HMC": "red",
        "DEMetro": "green",
        "SMC": "orange",
    }

    # move SMC to the end of the samplers list for efficient building of PyMC models
    samplers = list(experiment_settings["samplers"]) 
    if "SMC" in samplers:
        samplers.remove("SMC")
        samplers.append("SMC")

    plot_first_sample = experiment_settings.get("plot_first_sample", False)
    df_all_runs = []

    # === Run the Experiment ===
    for run_id in range(1, runs + 1):
        logger.info(f"Running {config_descr} - Run {run_id}")

        run_random_seed = int(rng.integers(1_000_000))

        run_folder = os.path.join(runs_folder, f"run_{run_id}")
        csv_folder = os.path.join(run_folder, "result CSVs")
        traces_folder = os.path.join(run_folder, "trace_plots")
        plots_folder = os.path.join(run_folder, "plots_of_run")
        
        create_directories(run_folder, csv_folder, traces_folder, plots_folder)

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

                if varying_attribute == "weights":
                    posterior_kwargs["weights"] = value
                elif varying_attribute == "dimension_corr":
                    adjust_dimension_of_kwargs(posterior_type, posterior_kwargs_original, posterior_kwargs, target_dim=value, required_parameters=required_parameters)
                    if run_id == 1:
                            save_adjusted_posterior_config(
                                posterior_kwargs,
                                folder=regular_posteriors_folder,
                                dim_value=value
                            ) 
                elif varying_attribute == "init_scheme":
                    init_scheme = value
                elif varying_attribute == "num_samples":
                    num_samples = value
                elif varying_attribute == "num_chains":
                    num_chains = value
                else:
                    # Vary only the selected component's parameter
                    posterior_kwargs["component_params"][component_index][varying_attribute] = value

            else:

                if varying_attribute == "dimension_corr":
                    adjust_dimension_of_kwargs(posterior_type, posterior_kwargs_original, posterior_kwargs, target_dim=value, required_parameters=required_parameters)
                    if run_id == 1:
                            save_adjusted_posterior_config(
                                posterior_kwargs,
                                folder=regular_posteriors_folder,
                                dim_value=value
                            )                     
                elif varying_attribute == "init_scheme":
                    init_scheme = value
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
                base_posterior = MixturePosterior(**posterior_kwargs)
            elif posterior_type == "Custom":
                logp_func = posterior_kwargs["logp_func"]
                base_posterior = CustomPosterior(logp_func=logp_func)
            else:
                base_posterior = SinglePosterior(dist_name=posterior_type, dist_params=posterior_kwargs)

            means = None
            initvals = None
  
            if init_scheme is not None:
                    means = extract_means_from_posterior(posterior_type, posterior_kwargs)
                    initvals = get_initvals(init_scheme, means, num_chains, rng, run_id, init_value_folder, value, unimodal_init_margin=unimodal_init_margin)
        
            # Get IID samples for the current varying value
            if posterior_type != "Custom" and varying_attribute not in ["init_scheme", "num_chains"]:
                iid_batches = iid_batches_dict[value]
            elif posterior_type == "Custom":
                iid_batches = None

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
                        low, high,_,_,_  = get_uniform_prior_bounds(means_array=means_array, expansion_factor=0.25)   
                        posterior = MixturePosterior(**posterior_kwargs, use_smc=True, low=low, high=high)             

                    else :
                        # compute higher and lower bound for init prior
                        low, high, _,_,_ = get_uniform_prior_bounds(means_array=means_array, expansion_factor=0.25, unimodal_init_margin=unimodal_init_margin)
                        posterior = SinglePosterior(dist_name=posterior_type, dist_params=posterior_kwargs, use_smc=True, low=low, high=high)

                if posterior_type == "Mixture" and component_index is not None:
                        logger.info(f"Running {sampler_name} with {varying_attribute} = {value} (Component {component_index})")
                else:
                    logger.info(f"Running {sampler_name} with {varying_attribute} = {value}")

                # **Measure Computation Time**
                start_time = time.time()
                trace = posterior.run_sampling(
                    sampler_name, num_samples=samples_per_chain, num_chains=num_chains, initvals = initvals, run_id=run_id, plot_first_sample=plot_first_sample,
                    init_folder= init_value_folder, value=value, means=means, posterior_type=posterior_type, run_random_seed=run_random_seed)
                end_time = time.time()
                runtime = end_time - start_time
                
                # Plot trace plots in notebook if requested
                if experiment_settings.get("plot_traces_in_notebook", False):
                    handle_trace_plots(
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
                    handle_trace_plots(
                        trace=trace,
                        sampler_name=sampler_name,
                        varying_attribute=varying_attribute,
                        value=value,
                        show=False,
                        save_path= os.path.join(var_attr_folder, f"{sampler_name}_trace_plot.pdf"),
                        save_individual=experiment_settings.get("save_individual_traceplots_per_dim", False),
                    )
                
                # Save trace to NetCDF file if requested
                if experiment_settings.get("save_traces", False):
                    trace_filename = os.path.join(var_attr_folder, f"{sampler_name}_trace.nc")
                    az.to_netcdf(trace, trace_filename)

                posterior_samples = trace.posterior["posterior"].values

                # Ensure posterior_samples always has shape (N, dims)
                if posterior_samples.ndim == 2:
                    posterior_samples = posterior_samples.reshape(-1, 1) 
                else:
                    posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[-1])
 
                # Only compute Wasserstein distance if we have iid_samples
                if posterior_type != "Custom":
                    # 2*runs have already been used for iid vs iid comparison
                    fresh_iid_index = num_iid_vs_iid_batches + run_id-1
                    iid_batch = ensure_2d(iid_batches[fresh_iid_index]) 
                    dim = get_posterior_dim(posterior_type, posterior_kwargs)
                    # print dimension with logging level DEBUG
                    if get_posterior_dim(posterior_type, posterior_kwargs) > 1:
                        mcmc_vs_iid_swd = sliced_wasserstein_distance(posterior_samples, iid_batch, L=50)
                    else:
                        mcmc_vs_iid_swd = sliced_wasserstein_distance(posterior_samples, iid_batch, L=1)
                    mmd_rff_value = compute_mmd_rff(posterior_samples, iid_batch, D=500, sigma=1.0, seed=run_random_seed)
                else:
                    mcmc_vs_iid_swd = np.nan
                    mmd_rff_value = np.nan

                # Compute R-hat and ESS
                r_hat, ess = get_scalar_rhat_and_ess(trace)

                #print(f"R-hat for sampler {sampler_name}: {r_hat}")
                #print(f"ESS for sampler {sampler_name}: {ess}")

                results.append({
                    "run_id": run_id,
                    varying_attribute: value,
                    "sampler": sampler_name,
                    "wasserstein_distance": mcmc_vs_iid_swd,
                    "mmd_rff": mmd_rff_value,
                    "r_hat": r_hat,
                    "ess": ess,
                    "runtime": runtime
                })


        # Convert results to DataFrame and save
        df_results = pd.DataFrame(results)

        var_attr_is_tuple = False

        # Handle tuple-based attributes consistently
        if isinstance(df_results[varying_attribute].iloc[0], tuple):
            var_attr_is_tuple = True
            df_results[varying_attribute] = df_results[varying_attribute].apply(str)
            varying_attribute_for_plot = varying_attribute
        else:
            varying_attribute_for_plot = varying_attribute

        df_results = df_results.sort_values(varying_attribute_for_plot, ascending=True)

        if experiment_settings.get("save_plots_and_csv_per_run", False):
            plot_and_save_all_metrics(
                df_results=df_results,
                sampler_colors=sampler_colors,
                varying_attribute=varying_attribute,
                varying_attribute_for_plot=varying_attribute_for_plot,
                csv_folder=csv_folder,
                plots_folder=plots_folder,
                run_id=run_id,
                config_descr=config_descr
            )

        df_all_runs.append(df_results)

        # Now increments the TQDM progress bar if it's provided
        if progress_bar is not None:
            progress_bar.update(1)

    logger.info("All runs completed successfully.")

    # ===== GLOBAL RESULTS FOLDER =====
    global_folder = os.path.join(group_folder, "global_results")
    global_results_folder = os.path.join(global_folder, "result CSVs")
    global_plots_folder = os.path.join(global_folder, "plots")
    create_directories(global_folder, global_results_folder, global_plots_folder)
    
    # Combine all results into a single data frame 
    df_all_runs = pd.concat(df_all_runs, ignore_index=True)

    if var_attr_is_tuple:
        iid_ref_stats_dict = {str(k): v for k, v in iid_ref_stats_dict.items()}

    compute_and_save_global_metrics(
        df_all_runs=df_all_runs,
        sampler_colors=sampler_colors,
        varying_attribute=varying_attribute,
        varying_values=varying_values,
        runs=runs,
        config_descr=config_descr,
        global_results_folder=global_results_folder,
        global_plots_folder=global_plots_folder,
        png_folder=png_folder,
        iid_ref_stats_dict=iid_ref_stats_dict
    )

    logger.info(f"===== Config {config_descr} completed successfully. =====")

experiment_name = "new_run"
config_names = ["Multimodal", "Dimension", "Correlated", "Skewness", "Kurtosis"]

# Define the root directory for all experiments
experiment_root_folder = os.path.join("experiments", f"exp_{experiment_name}")
results_folder = os.path.join(experiment_root_folder, "results")
configs_folder = os.path.join(experiment_root_folder, "configs")
report_pngs_folder = os.path.join(experiment_root_folder, "report_pngs")

# Check if the folder already exists
if os.path.exists(experiment_root_folder):
    user_input = input(
        f"Folder '{experiment_root_folder}' already exists and will be overwritten.\n"
        "Do you want to continue? (yes/no): "
    ).strip().lower()

    if user_input not in ["yes", "y"]:
        print("Operation aborted. No files were deleted.")
        sys.exit(0)

    shutil.rmtree(results_folder)
    shutil.rmtree(report_pngs_folder)
else:
    create_directories(experiment_root_folder)
 
create_directories(results_folder)
create_directories(report_pngs_folder)

# Copy experiment_template folders into experiment_root
for subfolder in ["default_vals", "settings"]:
    src = os.path.join("experiment_template", subfolder)
    dst = os.path.join(experiment_root_folder, subfolder)
    if not os.path.exists(dst):
        shutil.copytree(src, dst)

if not os.path.exists(configs_folder):
    os.makedirs(configs_folder)

# Copy only the configs that are defined in config_names
for config_name in config_names:
    src = os.path.join("experiment_template", "configs", f"{config_name}.yaml")
    dst = os.path.join(experiment_root_folder, "configs", f"{config_name}.yaml")
    if os.path.exists(src):
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    else:
        print(f"Warning: Config file {src} not found!")


experiment_paths = get_experiment_paths(config_names, base_dir=os.path.join(experiment_root_folder, "configs"))
settings_path = os.path.join(experiment_root_folder, "settings", "experiment_settings.yaml")
defaults_path = os.path.join(experiment_root_folder, "default_vals", "attribute_default_vals.yaml")

experiment_settings = load_experiment_settings(settings_path)
defaults = load_default_values(defaults_path)

experiments = []
for path in experiment_paths:
    group_name, config_list = load_config_file(path)
    config_list = [apply_defaults_to_config(cfg, defaults) for cfg in config_list]
    experiments.append((group_name, config_list))

total_configs = sum(len(config_list) for _, config_list in experiments)
runs_per_config = defaults.get("runs", 1)
total_runs = sum(config["runs"] for _,exp_group in experiments for config in exp_group)

print(f"number of configs: {total_configs}")
print(f"number of runs per config: {runs_per_config}")
print(f"total number of runs: {total_runs}")

# Validate all configurations before running the experiments
for group_name, exp_group in experiments:
    for config in exp_group:
        validate_config(config)

print("All configurations are valid. Starting experiments with the following settings and defaults:")
print("\nExperiment Settings:")
print(json.dumps(experiment_settings, indent=2))
print("\nDefaults:")
print(json.dumps(defaults, indent=2), "\n")


start_time = time.time()
start_dt = datetime.now()

failed_configs = []

with tqdm(total=total_runs, desc="Total experiment progress") as pbar:
    for group_name, exp_group in experiments:
        for config in exp_group:
            try:
                # png folder for html report for each group and config
                png_folder = os.path.join(experiment_root_folder, "report_pngs", group_name, config["config_descr"])
                create_directories(png_folder)

                run_experiment(
                    results_folder,
                    png_folder,
                    experiment_settings,
                    posterior_type=config["posterior_type"],
                    config_descr=config["config_descr"],
                    runs=config["runs"],
                    varying_attribute=config["varying_attribute"],
                    varying_values=config["varying_values"],
                    init_scheme="varies" if config["varying_attribute"] == "init_scheme" else config.get("init_scheme"),
                    num_samples="varies" if config["varying_attribute"] == "num_samples" else config["num_samples"],
                    num_chains="varies" if config["varying_attribute"] == "num_chains" else config["num_chains"],
                    base_random_seed=config.get("base_random_seed"),
                    unimodal_init_margin=config.get("unimodal_init_margin"),
                    group_name=group_name,
                    progress_bar=pbar, 
                    # Pass remaining keys as posterior_kwargs
                    **{k: v for k, v in config.items() if k not in [
                        "config_descr", "runs", "varying_attribute", "varying_values", 
                        "num_samples", "num_chains", "init_scheme", 
                        "base_random_seed", "posterior_type", "unimodal_init_margin"
                    ]} 
                )
            except Exception as e:
                print(f"Error in config '{config['config_descr']}': {e}")
                traceback.print_exc()
                failed_configs.append((config['config_descr'], str(e)))
                

end_time = time.time()
end_dt = datetime.now()
duration = end_time - start_time
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = round(duration % 60, 1)

generate_html_report(
        experiment_root_folder=experiment_root_folder,
        experiments=experiments,
        output_path=os.path.join(experiment_root_folder, f"exp_{experiment_name}_report.html")
    )

def get_folder_size(path='.'):
    """Compute total size of all files in directory."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

# Prepare the summary text
size_bytes = get_folder_size(experiment_root_folder)

summary_lines = [
    "\n============================",
    "Experiment Summary",
    "============================",
    f"Started at:                   {start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
    f"Finished at:                  {end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
    f"Total duration:               {hours}h {minutes}m {seconds}s",
    f"Output folder:                {experiment_root_folder}",
    f"Output folder size:           {humanize.naturalsize(size_bytes)}",
    f"Total configurations:         {total_configs}",
    f"Successful configuration:     {total_configs - len(failed_configs)}",
    f"Failed configurations:        {len(failed_configs)}"
]

if failed_configs:
    summary_lines.append("\n Failed Configurations:")
    for cfg, msg in failed_configs:
        summary_lines.append(f" - {cfg}: {msg}")

# Print to console
print("\n".join(summary_lines))

# Also save to summary.txt
summary_path = os.path.join(results_folder, "summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

print(f"Summary saved to: {summary_path}")
