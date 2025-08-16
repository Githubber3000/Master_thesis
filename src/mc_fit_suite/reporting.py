from __future__ import annotations
import json, os
import logging
import copy
from jinja2 import Environment, FileSystemLoader 
from glob import glob
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from itertools import combinations
import matplotlib.gridspec as gridspec
import pprint
import pandas as pd
import arviz as az

from .utils   import extract_varying_value_from_json, safe_json_dump, build_trace_groups, load_pairwise_sampler_stats, load_selected_global_stats

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family":           "serif",
    "axes.titlesize":        14,   
    "axes.labelsize":        22,  
    "xtick.labelsize":       18,    
    "ytick.labelsize":       18,
    "legend.fontsize":       8,    
    "legend.title_fontsize": 9,
    "lines.linewidth":       1.5,
    "lines.markersize":      6,
})


def generate_html_report(experiment_root_folder, report_pngs_folder, experiments, output_path, do_mmd, do_mmd_rff):
    """
    Generates a single HTML report for the entire experiment (all groups and configs).
    """

    template_path = "."  
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template("report.html")

    def rel(p): 
        return os.path.relpath(p, start=experiment_root_folder)
    
    def collect_metric_pngs(base, metrics):
        out = {}
        for m in metrics:
            full_p = os.path.join(base, f"{m}_global_plot_shaded.png")
            if os.path.exists(full_p):
                out[m] = rel(full_p)
        return out
    
    def collect_scatter_pngs(base, metrics):
        out = {}
        for m in metrics:
            full_p = os.path.join(base, f"{m}_global_plot_scatter.png")
            if os.path.exists(full_p):
                out[m] = rel(full_p)
        return out
    
    def collect_scatter_htmls(base, metrics):
        out = {}
        for m in metrics:
            full_p = os.path.join(base, f"{m}_global_scatter_interactive.html")
            if os.path.exists(full_p):
                out[m] = rel(full_p)
        return out

    def collect_glass_pngs(base, glass_keys): 
        out = {}
        for k in glass_keys:
            full_p = os.path.join(base, f"glass_delta_{k}.png")
            if os.path.exists(full_p):
                out[k] = rel(full_p)
        return out

    MASTER_ORDER = [
        "mean_rmse",
        "var_rmse",
        "wasserstein_distance",
        "mmd",
        "mmd_rff",
        "runtime",
        "ess",
        "ess_per_sec",
        "r_hat",
        "mode_transitions"
    ]

    metrics = [
      m for m in MASTER_ORDER
      if (m != "mmd"     or do_mmd)
     and (m != "mmd_rff" or do_mmd_rff)
    ]

    keys = ["ws", "mean_rmse", "var_rmse"]                      
    if do_mmd:     keys.append("mmd")  
    if do_mmd_rff: keys.append("mmd_rff")

    groups_data = []

    for group_name, configs in experiments:
        config_entries = []

        for config in configs:

            config_descr = config["config_descr"]

            result_folder_chain  = os.path.join(experiment_root_folder, "results", group_name, config_descr, "global_results", "chain_results")
            result_folder_pooled = os.path.join(experiment_root_folder, "results", group_name, config_descr, "global_results", "pooled_results")


            global_stats_chain  = load_selected_global_stats(result_folder_chain, keys, "stats")
            global_stats_pooled = load_selected_global_stats(result_folder_pooled, keys, "stats")


            # Load pairwise sampler comparison stats
            pairwise_stats_chain  = {
                "ws": load_pairwise_sampler_stats(result_folder_chain, "ws"),
                "mean_rmse": load_pairwise_sampler_stats(result_folder_chain, "mean_rmse"),
                "var_rmse": load_pairwise_sampler_stats(result_folder_chain, "var_rmse"),
                "mmd": load_pairwise_sampler_stats(result_folder_chain, "mmd") if do_mmd else {},
                "mmd_rff": load_pairwise_sampler_stats(result_folder_chain, "mmd_rff") if do_mmd_rff else {}
            }
            pairwise_stats_pooled = {
                "ws": load_pairwise_sampler_stats(result_folder_pooled, "ws"),
                "mean_rmse": load_pairwise_sampler_stats(result_folder_pooled, "mean_rmse"),
                "var_rmse": load_pairwise_sampler_stats(result_folder_pooled, "var_rmse"),
                "mmd": load_pairwise_sampler_stats(result_folder_pooled, "mmd") if do_mmd else {},
                "mmd_rff": load_pairwise_sampler_stats(result_folder_pooled, "mmd_rff") if do_mmd_rff else {}
            }

            delta_data_chain = load_selected_global_stats(result_folder_chain, keys, "delta")
            delta_data_pooled = load_selected_global_stats(result_folder_pooled, keys, "delta")

            png_base = os.path.join(experiment_root_folder, "results", "z_html_pngs", group_name, config_descr)
            pooled_png_base = os.path.join(png_base, "pooled_global_plots")
            chain_png_base = os.path.join(png_base, "chain_global_plots")

            # Glob all KDE plots for this config
            kde_path = os.path.join(report_pngs_folder, group_name, config_descr,"IID_KDE_and_Histograms", "iid_hist_kde_*.png")
            kde_plots = sorted(glob(kde_path), key=extract_varying_value_from_json)

            # Turn absolute paths into relative paths for <img src=...> in HTML
            rel_kde_paths = [rel(p) for p in kde_plots]

            pooled_init_path = os.path.join(report_pngs_folder, group_name, config_descr, "pooled_init", "init_*.png")
            pooled_init_plots = sorted(glob(pooled_init_path), key=extract_varying_value_from_json) 

            rel_pooled_init_paths = [rel(p) for p in pooled_init_plots]

            chain_init_path = os.path.join(report_pngs_folder, group_name, config_descr, "chain_init", "init_*.png")
            chain_init_plots = sorted(glob(chain_init_path), key=extract_varying_value_from_json)

            rel_chain_init_paths = [rel(p) for p in chain_init_plots]    

            pooled_trace_path = os.path.join(report_pngs_folder, group_name, config_descr, "trace_plots", "pooled_*.png")
            pooled_trace_plots = sorted(glob(pooled_trace_path), key=extract_varying_value_from_json)

            chain_trace_path = os.path.join(report_pngs_folder, group_name, config_descr, "trace_plots", "chain_*.png")
            chain_trace_plots = sorted(glob(chain_trace_path), key=extract_varying_value_from_json)

            trace_groups = build_trace_groups(pooled_trace_plots, chain_trace_plots, rel, type="trace")

            pooled_pairwise_scatter_path = os.path.join(report_pngs_folder, group_name, config_descr, "pairwise_scatter", "pooled_*.html")
            pooled_pairwise_scatter_plots = sorted(glob(pooled_pairwise_scatter_path), key=extract_varying_value_from_json)

            chain_pairwise_scatter_path = os.path.join(report_pngs_folder, group_name, config_descr, "pairwise_scatter", "chain_*.html")
            chain_pairwise_scatter_plots = sorted(glob(chain_pairwise_scatter_path), key=extract_varying_value_from_json)

            scatter_groups = build_trace_groups(pooled_pairwise_scatter_plots, chain_pairwise_scatter_plots, rel, type="pairwise_scatter")

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

                # batch
                "metric_plot_paths_pooled" : collect_metric_pngs(pooled_png_base, metrics),
                "glass_plot_paths_pooled"  : collect_glass_pngs(pooled_png_base, keys),
                "scatter_plot_paths_pooled": collect_scatter_pngs(pooled_png_base, metrics),
                "scatter_html_paths_pooled": collect_scatter_htmls(pooled_png_base, metrics),

                # chain 
                "metric_plot_paths_chain" : collect_metric_pngs(chain_png_base, metrics),
                "glass_plot_paths_chain"  : collect_glass_pngs(chain_png_base, keys),
                "scatter_plot_paths_chain": collect_scatter_pngs(chain_png_base, metrics),
                "scatter_html_paths_chain": collect_scatter_htmls(chain_png_base, metrics),

                # KDE and init plots
                "iid_kde_plot_paths": rel_kde_paths,
                "pooled_init_plot_paths": rel_pooled_init_paths,
                "chain_init_plot_paths": rel_chain_init_paths,
                "kde_init_triples": list(zip(rel_kde_paths, rel_pooled_init_paths, rel_chain_init_paths)),

                # sampler stats
                "sampler_stats_chain": global_stats_chain,
                "sampler_stats_pooled": global_stats_pooled,
                "pairwise_stats_chain": pairwise_stats_chain,
                "pairwise_stats_pooled": pairwise_stats_pooled,

                # delta data
                "delta_data_chain": delta_data_chain,
                "delta_data_pooled": delta_data_pooled,

                "trace_groups": trace_groups,
                "scatter_groups": scatter_groups
            }

            config_entries.append(entry)

        groups_data.append({
            "name": group_name,
            "configs": config_entries
        })

    html = template.render(
        experiment_name=os.path.basename(experiment_root_folder),
        groups=groups_data,
        metrics=metrics
    )

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Experiment-level HTML report saved to: {output_path}")



def plot_and_save_all_metrics(df_results, sampler_colors, varying_attribute, varying_attribute_for_plot, csv_folder, plots_folder, run_id, config_descr, do_mmd, do_mmd_rff):
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
    metrics = [ "mean_rmse", "var_rmse",  "wasserstein_distance", "runtime", "ess", "ess_per_sec", "r_hat", "mode_transitions"]

    if do_mmd:
        metrics.append("mmd")
    if do_mmd_rff:
        metrics.append("mmd_rff")

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

    # Define custom mapping for specific attributes
    attribute_name_map = {
        "mu": "Mode Distance" if not config_descr.lower().startswith("base") else "Base Case (2D Gaussian)",
        "mm": "Mode Distance",
        "nu": "Degrees of Freedom"
    }

    # Get label: use mapping if available, else fallback to generic formatting
    attribute_label = attribute_name_map.get(
        varying_attribute,
        varying_attribute.replace("_", " ").title()
    )

    metric_name_map = {
        "wasserstein_distance": "SWD",
        "mmd_rff": "MMD-RFF",
        "mmd": "MMD",
        "mean_rmse": "Mean RMSE",
        "var_rmse": "Variance RMSE",
        "runtime": "Runtime (s)",
        "ess": "ESS",
        "ess_per_sec": "ESS/sec",
        "r_hat": "R-hat",
        "mode_transitions": "Mode Transitions"
    }


    for metric in metrics:
        # Get display label for metric
        metric_label = metric_name_map.get(
            metric,
            metric.replace('_', ' ').title()
        )

        fig, ax = fig_ax_pairs[metric]
        finalize_and_save_plot(fig,ax, attribute_label, metric_label, 
                               os.path.join(plots_folder, f"{metric}_run_{run_id}.pdf"))

        # f"{metric} for Samplers (config =_{config_descr})",  old title 


def compute_delta_and_stats(metric_name, means, values, sampler, sampler_metric_values, iid_means_dict, iid_stds_dict, iid_runs_dict, ax_g, color, global_avg_dfs, varying_attribute, runs):
    # Glass Δ
    iid_mean = pd.Series([iid_means_dict[k] for k in means.index], index=means.index)
    iid_std = pd.Series([iid_stds_dict[k] for k in means.index], index=means.index)
    glass_delta = (means - iid_mean) / iid_std.replace(0, np.nan)

    global_avg_dfs[sampler][f"{metric_name}_glass_delta"] = glass_delta
    global_avg_dfs[sampler][f"{metric_name}_mcmc_mean"] = means
    global_avg_dfs[sampler][f"{metric_name}_iid_mean"] = iid_mean
    global_avg_dfs[sampler][f"{metric_name}_iid_std"] = iid_std

    ax_g.plot(means.index, glass_delta, "o-", label=sampler, color=color)

    # MCMC vs IID
    iid_distances = np.array([iid_runs_dict[k] for k in means.index])
    mcmc_distances = np.array(values)

    tt_stats_mc_iid = {}
    tt_pvals_mc_iid = {}

    for varying_value, iid_distance, mcmc_distance in zip(means.index, iid_distances, mcmc_distances):
        stat, p_value = ttest_ind(mcmc_distance, iid_distance, equal_var=False, nan_policy="omit")
        tt_stats_mc_iid[varying_value] = stat
        tt_pvals_mc_iid[varying_value] = p_value

    global_avg_dfs[sampler][f"{metric_name}_t_stat"] = pd.Series(tt_stats_mc_iid)
    global_avg_dfs[sampler][f"{metric_name}_p_value"] = pd.Series(tt_pvals_mc_iid)

    # MCMC vs MCMC
    for sampler_a, sampler_b in combinations(sampler_metric_values.keys(), 2):
        key = f"{sampler_a}_vs_{sampler_b}"
        if key not in global_avg_dfs:
            global_avg_dfs[key] = {}

        values_a = sampler_metric_values[sampler_a]["values"]
        values_b = sampler_metric_values[sampler_b]["values"]
        index = sampler_metric_values[sampler_a]["index"]

        assert index.equals(sampler_metric_values[sampler_b]["index"]), f"Index mismatch for {key}"

        tt_stats_mc_mc = {}
        tt_pvals_mc_mc = {}
        paired_d = {}

        for i, varying_value in enumerate(index):
            a_vals = values_a[i]
            b_vals = values_b[i]
            stat, p_value = ttest_rel(a_vals, b_vals, nan_policy="omit")
            paired_d[varying_value] = stat / np.sqrt(runs)

            tt_stats_mc_mc[varying_value] = stat
            tt_pvals_mc_mc[varying_value] = p_value

        df_pairwise = pd.DataFrame({
            f"{varying_attribute}": index,
            f"{metric_name}_t_stat": pd.Series(tt_stats_mc_mc, index=index),
            f"{metric_name}_p_value": pd.Series(tt_pvals_mc_mc, index=index),
            f"{metric_name}_paired_cohens_d": pd.Series(paired_d, index=index),
        })

        global_avg_dfs[key][f"{metric_name}_pairwise"] = df_pairwise


def plot_iid_baseline(
    metric_name,
    medians,
    means,
    ax_shaded,
    ax_mean,
    iid_medians_dict,
    iid_q25_dict,
    iid_q75_dict,
    iid_means_dict,
    iid_stds_dict
):
    iid_medians = np.array([iid_medians_dict[k] for k in medians.index])
    iid_q25     = np.array([iid_q25_dict[k] for k in medians.index])
    iid_q75     = np.array([iid_q75_dict[k] for k in medians.index])
    iid_means   = np.array([iid_means_dict[k] for k in means.index])
    iid_stds    = np.array([iid_stds_dict[k] for k in means.index])



    # --- Median/IQR panel ---
    if len(medians.index) > 1:
        ax_shaded.plot(medians.index, iid_medians, "--", color="black")
        ax_shaded.fill_between(medians.index, iid_q25, iid_q75, color="gray", alpha=0.1)
        ax_mean.plot(means.index, iid_means, "--", color="black")
        ax_mean.fill_between(means.index, iid_means + iid_stds, iid_means - iid_stds, color="gray", alpha=0.1)
    else:
        lower_err = iid_medians - iid_q25
        upper_err = iid_q75 - iid_medians
        yerr = np.vstack([lower_err, upper_err])
        ax_shaded.errorbar(medians.index, iid_medians, yerr=yerr, fmt="o", color="black", capsize=5)
        ax_mean.errorbar(means.index, iid_means, yerr=iid_stds, fmt="o", color="black", capsize=5)


    # Plot IID median line + IQR fill
    # ax_shaded.plot(medians.index, iid_medians, "--", color="black", marker="o" if len(iid_medians) == 1 else None)
    # ax_shaded.fill_between(medians.index, iid_q25, iid_q75, color="gray", alpha=0.1)

    # Plot IID mean line + std fill
    # ax_mean.plot(means.index, iid_means, "--", color="black", marker="o" if len(iid_means) == 1 else None)
    # ax_mean.fill_between(means.index, iid_means + iid_stds, iid_means - iid_stds, color="gray", alpha=0.1)


def compute_and_save_global_metrics(df_all_runs, sampler_colors, varying_attribute, varying_values, runs, num_chains, config_descr, global_results_folder, global_plots_folder, png_folder, iid_ref_stats_dict, save_extra_scatter, do_mmd, do_mmd_rff, log_scaled_plots):
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
    metrics = ["mean_rmse", "var_rmse", "wasserstein_distance","runtime", "ess", "ess_per_sec", "r_hat", "mode_transitions"]

    if do_mmd:
        metrics.append("mmd")
    if do_mmd_rff:
        metrics.append("mmd_rff")

    if config_descr.lower().startswith("base"):
        is_base_case = True
    else:
        is_base_case = False

    # Define custom mapping for specific attributes
    attribute_name_map = {
        "mu": "Mode Distance" if not is_base_case else "Base Case (2D Gaussian)",
        "mm": "Mode Distance",
        "nu": "Degrees of Freedom"
    }

    # Get label: use mapping if available, else fallback to generic formatting
    attribute_label = attribute_name_map.get(
        varying_attribute,
        varying_attribute.replace("_", " ").title()
    )

    metric_name_map = {
        "wasserstein_distance": "SWD",
        "mmd_rff": "MMD-RFF",
        "mmd": "MMD",
        "mean_rmse": "Mean RMSE",
        "var_rmse": "Variance RMSE",
        "runtime": "Runtime (s)",
        "ess": "ESS",
        "ess_per_sec": "ESS/sec",
        "r_hat": "R-hat",
        "mode_transitions": "Mode Transitions"
    }

    # Figure for shaded plots (median + IQR)
    fig_ax_pairs_shaded = {m: plt.subplots(figsize=(10, 6)) for m in metrics}
    # Figure for mean of samplers and mean + sd of IID baseline
    fig_ax_pairs_mean = {m: plt.subplots(figsize=(10, 6)) for m in metrics}
    # Figure for Glass delta of wasserstein_distance
    fig_g, ax_g = plt.subplots(figsize=(10, 6))
    fig_g_mean, ax_g_mean = plt.subplots(figsize=(10, 6)) 
    fig_g_var, ax_g_var = plt.subplots(figsize=(10, 6))  

    if do_mmd:
        fig_g_mmd, ax_g_mmd = plt.subplots(figsize=(10, 6)) 
    if do_mmd_rff:
        fig_g_mmd_rff, ax_g_mmd_rff = plt.subplots(figsize=(10, 6)) 


    global_avg_dfs = {}
    scatter_data = {}

    # Load IID reference statistics
    iid_means_dict_swd = {}
    iid_stds_dict_swd = {}
    iid_medians_dict_swd = {}
    iid_q25_dict_swd = {}
    iid_q75_dict_swd = {}
    iid_runs_dict_swd = {} 

    iid_means_dict_mean_rmse = {}
    iid_stds_dict_mean_rmse = {}
    iid_medians_dict_mean_rmse = {}
    iid_q25_dict_mean_rmse = {}
    iid_q75_dict_mean_rmse = {}
    iid_runs_dict_mean_rmse = {}

    iid_means_dict_var_rmse = {}
    iid_stds_dict_var_rmse = {}
    iid_medians_dict_var_rmse = {}
    iid_q25_dict_var_rmse = {}
    iid_q75_dict_var_rmse = {}
    iid_runs_dict_var_rmse = {}

    if do_mmd:
        iid_means_dict_mmd = {}
        iid_stds_dict_mmd = {}
        iid_medians_dict_mmd = {}
        iid_q25_dict_mmd = {}
        iid_q75_dict_mmd = {}
        iid_runs_dict_mmd = {}

    if do_mmd_rff:
        iid_means_dict_mmd_rff = {}
        iid_stds_dict_mmd_rff = {}
        iid_medians_dict_mmd_rff = {}
        iid_q25_dict_mmd_rff = {}
        iid_q75_dict_mmd_rff = {}
        iid_runs_dict_mmd_rff = {}


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
        iid_runs_dict_swd[k] = iid_entry["runs_swd"]

        iid_means_dict_mean_rmse[k] = iid_entry["mean_mean_rmse"]
        iid_stds_dict_mean_rmse[k] = iid_entry["std_mean_rmse"]
        iid_medians_dict_mean_rmse[k] = iid_entry["median_mean_rmse"]
        iid_q25_dict_mean_rmse[k] = iid_entry["q25_mean_rmse"]
        iid_q75_dict_mean_rmse[k] = iid_entry["q75_mean_rmse"]
        iid_runs_dict_mean_rmse[k] = iid_entry["runs_mean_rmse"]

        iid_means_dict_var_rmse[k] = iid_entry["mean_var_rmse"]
        iid_stds_dict_var_rmse[k] = iid_entry["std_var_rmse"]
        iid_medians_dict_var_rmse[k] = iid_entry["median_var_rmse"]
        iid_q25_dict_var_rmse[k] = iid_entry["q25_var_rmse"]
        iid_q75_dict_var_rmse[k] = iid_entry["q75_var_rmse"]
        iid_runs_dict_var_rmse[k] = iid_entry["runs_var_rmse"]

        if do_mmd:
            iid_means_dict_mmd[k] = iid_entry["mean_mmd"]
            iid_stds_dict_mmd[k] = iid_entry["std_mmd"]
            iid_medians_dict_mmd[k] = iid_entry["median_mmd"]
            iid_q25_dict_mmd[k] = iid_entry["q25_mmd"]
            iid_q75_dict_mmd[k] = iid_entry["q75_mmd"]
            iid_runs_dict_mmd[k] = iid_entry["runs_mmd"]

        if do_mmd_rff:
            iid_means_dict_mmd_rff[k] = iid_entry["mean_mmd_rff"]
            iid_stds_dict_mmd_rff[k] = iid_entry["std_mmd_rff"]
            iid_medians_dict_mmd_rff[k] = iid_entry["median_mmd_rff"]
            iid_q25_dict_mmd_rff[k] = iid_entry["q25_mmd_rff"]
            iid_q75_dict_mmd_rff[k] = iid_entry["q75_mmd_rff"]
            iid_runs_dict_mmd_rff[k] = iid_entry["runs_mmd_rff"]


    for metric in metrics:

        sampler_metric_values = {}

        # Get display label for metric
        metric_label = metric_name_map.get(
            metric,
            metric.replace('_', ' ').title()
        )

        fig_shaded, ax_shaded = fig_ax_pairs_shaded[metric]
        fig_mean,   ax_mean   = fig_ax_pairs_mean[metric] 

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
                # ax_shaded.annotate("'DEMetropolis' r-hat skipped due to invalid values", 
                #        xy=(0.98, 0.02), xycoords='axes fraction',
                #        ha="right", va="bottom", fontsize=9, color="red")
                continue

            if metric == "r_hat":
                if df_pivot.isnull().values.any() or  (df_pivot > 1000).any().any():
                    logger.warning(f"Skipping r_hat plot for sampler {sampler} due to extremely high values.")                    
                    ax_shaded.annotate("'DEMetropolis' r-hat skipped due to >1000", 
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha="right", va="bottom", fontsize=9, color="red")
                    continue
            
            
            # Custom ordering based on config (only if needed)
            if isinstance(df_pivot.index[0], str):
                # Reorder the index to match the varying values
                custom_order = [str(t) for t in varying_values]
                df_pivot = df_pivot.reindex(custom_order)

            # # reverse order for tail wei
            # if varying_attribute == "nu":
            #     custom_order = sorted(varying_values, reverse=True)
            #     df_pivot = df_pivot.reindex(custom_order)   

            # Compute mean+std and median+quantiles
            means = df_pivot.mean(axis=1)
            stds = df_pivot.std(axis=1, ddof=1) 
            medians = df_pivot.median(axis=1)
            q25 = df_pivot.quantile(0.25, axis=1)
            q75 = df_pivot.quantile(0.75, axis=1)
            values = df_pivot.values
            
            sampler_metric_values[sampler] = {
                "values": df_pivot.values,
                "index": df_pivot.index
            }

            if not is_base_case:
                # Plot median line
                ax_shaded.plot(medians.index, medians, "o-", label=sampler, color=color)

                # Plot mean line
                ax_mean.plot(means.index, means, "o-", label=sampler, color=color)

            # Plot uncertainty: interquartile range (q25–q75)
            if len(medians.index) > 1:
                ax_shaded.fill_between(medians.index, q25, q75, color=color, alpha=0.2)
                ax_mean.fill_between(means.index, means + stds, means - stds, color=color, alpha=0.2)
            else:
                # lower_err = medians - q25
                # upper_err = q75 - medians
                # yerr = [lower_err, upper_err]
                # ax_shaded.errorbar(medians.index, medians, yerr=yerr, fmt="o", color=color, capsize=5)
                # ax_mean.errorbar(means.index, means, yerr=stds, fmt="o", color=color, capsize=5)


                names = list(sampler_colors.keys())
                i, n = names.index(sampler), len(names)
                width = 0.16                    
                off = (i - (n - 1) / 2) * width

                # center x (numeric); fall back to 0.0 if index is not numeric
                try:
                    x0 = float(np.asarray(medians.index)[0])
                except Exception:
                    x0 = 0.0

                # scalars for error bars
                med  = float(medians.iloc[0])
                ql   = float(q25.iloc[0])
                qu   = float(q75.iloc[0])
                mean_ = float(means.iloc[0])
                std_  = float(stds.iloc[0])

                # IQR around median (shaded panel)
                ax_shaded.errorbar([x0 + off], [med],
                                yerr=[[med - ql], [qu - med]],
                                fmt="o", color=color, capsize=5, zorder=2)

                # mean ± std (mean panel)
                ax_mean.errorbar([x0 + off], [mean_],
                                yerr=[[std_], [std_]],
                                fmt="o", color=color, capsize=5, zorder=2)

                ax_mean.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
                ax_shaded.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

            if save_extra_scatter:
                # x = v.attr value repeated per run
                xs, ys = np.broadcast_to(
                    df_pivot.index.to_numpy()[:, None], df_pivot.shape
                ).ravel(), df_pivot.to_numpy().ravel()

                scatter_data.setdefault(metric, []).append({
                    "sampler": sampler,
                    "xs": xs, "ys": ys, "color": color,
                    "medians": medians, "q25": q25, "q75": q75, "means": means, "stds": stds
                })

            # Save global avg for CSV
            if sampler not in global_avg_dfs:
                global_avg_dfs[sampler] = {}
            global_avg_dfs[sampler][metric] = (medians, q25, q75)


            metrics_config = {
                "wasserstein_distance": {
                    "name": "ws",
                    "mean": iid_means_dict_swd,
                    "std": iid_stds_dict_swd,
                    "runs": iid_runs_dict_swd,
                    "ax": ax_g,
                    "medians": iid_medians_dict_swd,
                    "q25": iid_q25_dict_swd,
                    "q75": iid_q75_dict_swd
                },
                "mean_rmse": {
                    "name": "mean_rmse",
                    "mean": iid_means_dict_mean_rmse,
                    "std": iid_stds_dict_mean_rmse,
                    "runs": iid_runs_dict_mean_rmse,
                    "ax": ax_g_mean,
                    "medians": iid_medians_dict_mean_rmse,
                    "q25": iid_q25_dict_mean_rmse,
                    "q75": iid_q75_dict_mean_rmse
                },
                "var_rmse": {
                    "name": "var_rmse",
                    "mean": iid_means_dict_var_rmse,
                    "std": iid_stds_dict_var_rmse,
                    "runs": iid_runs_dict_var_rmse,
                    "ax": ax_g_var,
                    "medians": iid_medians_dict_var_rmse,
                    "q25": iid_q25_dict_var_rmse,
                    "q75": iid_q75_dict_var_rmse
                }
            }

            if do_mmd:
                metrics_config["mmd"] = {
                    "name": "mmd",
                    "mean": iid_means_dict_mmd,
                    "std": iid_stds_dict_mmd,
                    "runs": iid_runs_dict_mmd,
                    "ax": ax_g_mmd,
                    "medians": iid_medians_dict_mmd,
                    "q25": iid_q25_dict_mmd,
                    "q75": iid_q75_dict_mmd
                }
            
            if do_mmd_rff:
                metrics_config["mmd_rff"] = {
                    "name": "mmd_rff",
                    "mean": iid_means_dict_mmd_rff,
                    "std": iid_stds_dict_mmd_rff,
                    "runs": iid_runs_dict_mmd_rff,
                    "ax": ax_g_mmd_rff,
                    "medians": iid_medians_dict_mmd_rff,
                    "q25": iid_q25_dict_mmd_rff,
                    "q75": iid_q75_dict_mmd_rff
                }

            if metric in metrics_config:
                cfg = metrics_config[metric]

                compute_delta_and_stats(
                    metric_name=cfg["name"],
                    means=means,
                    values=values,
                    sampler=sampler,
                    sampler_metric_values=sampler_metric_values,
                    iid_means_dict=cfg["mean"],
                    iid_stds_dict=cfg["std"],
                    iid_runs_dict=cfg["runs"],
                    ax_g=cfg["ax"],
                    color=color,
                    global_avg_dfs=global_avg_dfs,
                    varying_attribute=varying_attribute,
                    runs=runs
                )

                plot_iid_baseline(
                    metric_name=metric,
                    medians=medians,
                    means=means,
                    ax_shaded=ax_shaded,
                    ax_mean=ax_mean,
                    iid_medians_dict=cfg["medians"],
                    iid_q25_dict=cfg["q25"],
                    iid_q75_dict=cfg["q75"],
                    iid_means_dict=cfg["mean"],
                    iid_stds_dict=cfg["std"]
                )


        #  scatter plot for this metric
        if save_extra_scatter and metric in scatter_data:
            fig = go.Figure()

            for entry in scatter_data[metric]:
                sampler = entry["sampler"]

                median_legend  = f"{sampler}-median"      
                mean_legend = f"{sampler}-mean" 

                fig.add_trace(go.Scatter(
                    x=entry["xs"], y=entry["ys"],
                    mode="markers",
                    name=f"{sampler} pts",
                    marker=dict(color=entry["color"], opacity=0.6, size=6),
                    visible="legendonly"
                ))

                # median line
                fig.add_trace(go.Scatter(
                    x=entry["medians"].index,
                    y=entry["medians"],
                    mode="lines+markers",
                    name=f"{sampler} median ",
                    marker=dict(symbol="diamond", size=8, color=entry["color"]),
                    legendgroup=median_legend
                ))

                # IQR fill
                fig.add_trace(go.Scatter(
                    x=list(entry["medians"].index) + list(entry["medians"].index)[::-1],
                    y=list(entry["q75"]) + list(entry["q25"])[::-1],
                    fill="toself",
                    fillcolor=entry["color"],       
                    opacity=0.2,   
                    line=dict(width=0),
                    name=f"{sampler} IQR",
                    legendgroup=median_legend,
                    showlegend=False 
                ))

                # mean line
                fig.add_trace(go.Scatter(
                    x=entry["medians"].index,
                    y=entry["means"],
                    mode="lines+markers",
                    name=f"{sampler} mean",
                    line=dict(dash="dash", color=entry["color"], width=2),
                    marker=dict(symbol="triangle-up", size=8, color=entry["color"]),
                    legendgroup=mean_legend,
                    visible="legendonly"
                ))

                # std fill
                fig.add_trace(go.Scatter(
                    x=list(entry["medians"].index) + list(entry["medians"].index)[::-1],
                    y=list(entry["means"] + entry["stds"]) + list(entry["means"] - entry["stds"])[::-1],
                    fill="toself",
                    fillcolor=entry["color"],       
                    opacity=0.1,   
                    line=dict(width=0),
                    name=f"{sampler} ±1 SD",
                    legendgroup=mean_legend,
                    showlegend=False,
                    visible="legendonly" 
                ))

            # grab the x-axis values
            xvals = list(scatter_data[metric][0]["medians"].index)
            
            if metric in metrics_config:
                cfg = metrics_config[metric]  # this is safe if `metric` is always valid

                iid_median = [cfg["medians"][k] for k in xvals]
                iid_q25    = [cfg["q25"][k]     for k in xvals]
                iid_q75    = [cfg["q75"][k]     for k in xvals]
                iid_mean   = [cfg["mean"][k]    for k in xvals]
                iid_std    = [cfg["std"][k]     for k in xvals]
                iid_runs   = [cfg["runs"][k]    for k in xvals]

                # Flatten xs & ys of IID runs
                xs_iid, ys_iid = [], []
                for x, arr in zip(xvals, iid_runs):
                    xs_iid.extend([x] * len(arr))  
                    ys_iid.extend(arr)               

                fig.add_trace(go.Scatter(
                    x=xs_iid,
                    y=ys_iid,
                    mode="markers",
                    name="IID pts",
                    marker=dict(color="black", opacity=0.6, size=6),
                    visible="legendonly"
                ))

                # IID median line
                fig.add_trace(go.Scatter(
                    x=xvals,
                    y=iid_median,
                    mode="lines+markers",
                    name= f"IID median",
                    legendgroup="IID_median",                 
                    line=dict(color="black", width=2), 
                    marker=dict(symbol="diamond", size=8, color="black"),
                ))

                # IID IQR fill
                fig.add_trace(go.Scatter(
                    x=xvals + xvals[::-1],
                    y=iid_q75 + iid_q25[::-1],
                    fill="toself",
                    fillcolor="rgba(100,100,100,0.3)",       
                    line=dict(width=0),               
                    name="IID IQR",
                    legendgroup="IID_median",
                    showlegend=False 
                ))

                # IID mean line
                fig.add_trace(go.Scatter(
                    x=xvals,
                    y=iid_mean,  
                    mode="lines+markers",
                    name="IID mean",
                    legendgroup="IID_mean",                 
                    line=dict(dash="dash", color="black", width=2), 
                    marker=dict(symbol="triangle-up-open", size=8, color="black"),
                    visible="legendonly"
                ))

                # IID std fill
                fig.add_trace(go.Scatter(
                    x=xvals + xvals[::-1],
                    y=[m + s for m, s in zip(iid_mean, iid_std)] + [m - s for m, s in zip(iid_mean, iid_std)][::-1],
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.25)",       
                    line=dict(width=0),             
                    name="IID ±1 SD",
                    legendgroup="IID_mean",
                    showlegend=False,
                    visible="legendonly"
                ))

            fig.update_layout(
                title=f"All runs {metric.replace('_',' ').title()} ({runs} runs, {config_descr})",
                xaxis_title= attribute_label,
                yaxis_title= metric_label,
                legend=dict(
                    itemclick="toggle",                     
                    itemdoubleclick="toggleothers"          
                ),
                width=1100,
                height=1100 * 9 / 16,     
                margin=dict(l=40, r=40, t=50, b=40)
            )

            median_mask_on  = [
                (" median" in t.name and "mean" not in t.name) or ("IQR"  in t.name)
                for t in fig.data
            ]

            median_mask_off = ["legendonly" if v else None for v in median_mask_on]

            mean_mask_on    = [
                (" mean" in t.name) or ("IID ±1 SD" in t.name)    
                for t in fig.data
            ]

            mean_mask_off   = ["legendonly" if v else None for v in mean_mask_on]

            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    direction="left",
                    x=0.99, y=1.15,
                    showactive=False,
                    xanchor="right",       
                    yanchor="top",
                    pad=dict(l=12, r=12, t=4, b=4),
                    bgcolor="rgba(19, 143, 214, 0.38)",
                    bordercolor="darkgrey",
                    borderwidth=1,
                    font=dict(size=12),


                    buttons=[
                        dict(
                            label="md",
                            method="update",
                            args=[{"visible": median_mask_on}],
                            args2=[{"visible": median_mask_off}]
         
                        ),
                        dict(
                            label="me",
                            method="update",
                            args=[{"visible": mean_mask_on}],
                            args2=[{"visible": mean_mask_off}]
                        )
                    ]
                )]
            )

            # Save out as a standalone HTML
            html_path = os.path.join(
                png_folder,
                f"{metric}_global_scatter_interactive.html"
            )

            fig.write_html(html_path, include_plotlyjs="cdn", config=dict(responsive=True), full_html=True)



    # Save Global Averages per Sampler to CSV
    for sampler, metrics_dict in global_avg_dfs.items():
        
        wrote_pairwise = False

        for pairwise_key in ["ws_pairwise", "mmd_pairwise", "mmd_rff_pairwise", "mean_rmse_pairwise", "var_rmse_pairwise"]:
            if pairwise_key in metrics_dict:
                # Handle pairwise case: export the full df_pairwise
                df_pairwise = metrics_dict[pairwise_key]
                csv_filename = os.path.join(global_results_folder, f"Pairwise_results_{sampler}_{pairwise_key.replace('_pairwise', '')}.csv")
                df_pairwise.to_csv(csv_filename, index=False)
                wrote_pairwise = True
        
        # If only pairwise results exist, skip the global metric CSV
        if wrote_pairwise:
            continue

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
            "ws_mcmc_mean":  metrics_dict["ws_mcmc_mean"].values,
            "ws_iid_mean":   metrics_dict["ws_iid_mean"].values,
            "ws_iid_std":    metrics_dict["ws_iid_std"].values,
            "mean_rmse_mcmc_mean": metrics_dict["mean_rmse_mcmc_mean"].values,
            "mean_rmse_iid_mean":  metrics_dict["mean_rmse_iid_mean"].values,
            "mean_rmse_iid_std":   metrics_dict["mean_rmse_iid_std"].values,
            "var_rmse_mcmc_mean": metrics_dict["var_rmse_mcmc_mean"].values,
            "var_rmse_iid_mean":  metrics_dict["var_rmse_iid_mean"].values,
            "var_rmse_iid_std":   metrics_dict["var_rmse_iid_std"].values,

            **({"mmd_mcmc_mean": metrics_dict["mmd_mcmc_mean"].values,
                "mmd_iid_mean":  metrics_dict["mmd_iid_mean"].values,
                "mmd_iid_std":   metrics_dict["mmd_iid_std"].values}
            if do_mmd else {}),

            **({"mmd_rff_mcmc_mean": metrics_dict["mmd_rff_mcmc_mean"].values,
                "mmd_rff_iid_mean":  metrics_dict["mmd_rff_iid_mean"].values,
                "mmd_rff_iid_std":   metrics_dict["mmd_rff_iid_std"].values}
            if do_mmd_rff else {})
        })

        if "ws_glass_delta" in metrics_dict:
            df_global_avg["ws_glass_delta"] = metrics_dict["ws_glass_delta"].values
            df_global_avg["ws_t_stat"] = metrics_dict["ws_t_stat"].values
            df_global_avg["ws_p_value"] = metrics_dict["ws_p_value"].values

        if "var_rmse_glass_delta" in metrics_dict:
            df_global_avg["var_rmse_glass_delta"] = metrics_dict["var_rmse_glass_delta"].values
            df_global_avg["var_rmse_t_stat"] = metrics_dict["var_rmse_t_stat"].values
            df_global_avg["var_rmse_p_value"] = metrics_dict["var_rmse_p_value"].values

        if "mean_rmse_glass_delta" in metrics_dict:
            df_global_avg["mean_rmse_glass_delta"] = metrics_dict["mean_rmse_glass_delta"].values
            df_global_avg["mean_rmse_t_stat"] = metrics_dict["mean_rmse_t_stat"].values
            df_global_avg["mean_rmse_p_value"] = metrics_dict["mean_rmse_p_value"].values

        if "mmd_glass_delta" in metrics_dict:
            df_global_avg["mmd_glass_delta"] = metrics_dict["mmd_glass_delta"].values
            df_global_avg["mmd_t_stat"] = metrics_dict["mmd_t_stat"].values
            df_global_avg["mmd_p_value"] = metrics_dict["mmd_p_value"].values

        if "mmd_rff_glass_delta" in metrics_dict:
            df_global_avg["mmd_rff_glass_delta"] = metrics_dict["mmd_rff_glass_delta"].values
            df_global_avg["mmd_rff_t_stat"] = metrics_dict["mmd_rff_t_stat"].values
            df_global_avg["mmd_rff_p_value"] = metrics_dict["mmd_rff_p_value"].values


        csv_filename = os.path.join(global_results_folder, f"Global_results_{sampler}.csv")
        df_global_avg.to_csv(csv_filename, index=False)

    # Save plots
    for metric in metrics:

        # Get display label for metric
        metric_label = metric_name_map.get(
            metric,
            metric.replace('_', ' ').title()
        )

        #title_md = (f"Averaged {metric.replace('_', ' ').title()} "
        #        f"({runs} Runs, config = {config_descr})")   
        fig_shaded, ax_shaded = fig_ax_pairs_shaded[metric]
        pdf_path_md = os.path.join(global_plots_folder, f"{metric}_global_plot_shaded.pdf")
        png_path_md = os.path.join(png_folder, f"{metric}_global_plot_shaded.png")

        if metric == "mode_transitions":
            finalize_and_save_plot(fig_shaded, ax_shaded, attribute_label, metric_label, save_path=pdf_path_md, save_path_png=png_path_md)
        else:
            finalize_and_save_plot(fig_shaded, ax_shaded, attribute_label, metric_label, save_path=pdf_path_md, save_path_png=png_path_md, log_scaled_plots= log_scaled_plots)

        #title_me = (f"Mean {metric.replace('_', ' ').title()} "
        #              f"({runs} Runs, config = {config_descr})")
        fig_mean, ax_mean = fig_ax_pairs_mean[metric]
        pdf_path_me = os.path.join(global_plots_folder, f"{metric}_global_plot_mean.pdf")
        png_path_me = os.path.join(png_folder, f"{metric}_global_plot_mean.png")

        if metric == "mode_transitions":
            finalize_and_save_plot(fig_mean, ax_mean, attribute_label, metric_label, save_path=pdf_path_me, save_path_png=png_path_me)
        else:
            finalize_and_save_plot(fig_mean, ax_mean, attribute_label, metric_label, save_path=pdf_path_me, save_path_png=png_path_me, log_scaled_plots=log_scaled_plots)

    # Plot Glass's Δ for wasserstein_distance
    pdf_path = os.path.join(global_plots_folder, "glass_delta_ws.pdf")
    png_path = os.path.join(png_folder, "glass_delta_ws.png")
    #title_ws= f"Glass's Δ for Wasserstein Distance ({runs} Runs, config = {config_descr})"

    finalize_and_save_plot(fig_g, ax_g, xlabel=attribute_label, ylabel="Glass's Δ",
                            save_path=pdf_path, save_path_png=png_path)

    # Plot Glass's Δ for mean_rmse
    pdf_path_mean_rmse = os.path.join(global_plots_folder, "glass_delta_mean_rmse.pdf")
    png_path_mean_rmse = os.path.join(png_folder, "glass_delta_mean_rmse.png")
    #title_mean_rmse = f"Glass's Δ for Mean RMSE ({runs} Runs, config = {config_descr})"

    finalize_and_save_plot(fig_g_mean, ax_g_mean, xlabel=attribute_label, ylabel="Glass's Δ",
                            save_path=pdf_path_mean_rmse, save_path_png=png_path_mean_rmse)

    # Plot Glass's Δ for var_rmse
    pdf_path_var_rmse = os.path.join(global_plots_folder, "glass_delta_var_rmse.pdf")
    png_path_var_rmse = os.path.join(png_folder, "glass_delta_var_rmse.png")
    #title_var_rmse = f"Glass's Δ for Var RMSE ({runs} Runs, config = {config_descr})"

    finalize_and_save_plot(fig_g_var, ax_g_var, xlabel=attribute_label, ylabel="Glass's Δ",
                            save_path=pdf_path_var_rmse, save_path_png=png_path_var_rmse)

    if do_mmd:
        # Plot Glass's Δ for MMD
        pdf_path = os.path.join(global_plots_folder, "glass_delta_mmd.pdf")
        png_path = os.path.join(png_folder, "glass_delta_mmd.png")
        #title_mmd = f"Glass's Δ for MMD ({runs} Runs, config = {config_descr})"

        finalize_and_save_plot(fig_g_mmd, ax_g_mmd, xlabel=attribute_label, ylabel="Glass's Δ",
                            save_path=pdf_path, save_path_png=png_path)
    if do_mmd_rff:
        # Plot Glass's Δ for MMD-RFF
        pdf_path = os.path.join(global_plots_folder, "glass_delta_mmd_rff.pdf")
        png_path = os.path.join(png_folder, "glass_delta_mmd_rff.png")
        #title_mmd_rff = f"Glass's Δ for MMD-RFF ({runs} Runs, config = {config_descr})"

        finalize_and_save_plot(fig_g_mmd_rff, ax_g_mmd_rff, xlabel=attribute_label, ylabel="Glass's Δ",
                            save_path=pdf_path, save_path_png=png_path, log_scaled_plots=log_scaled_plots)




def finalize_and_save_plot(fig, ax, xlabel, ylabel, save_path, save_path_png=None, log_scaled_plots=False):
    """
    Finalizes the plot with labels, grid, and saves it to a file.
    
    Parameters:
    - fig: Matplotlib figure
    - ax: Matplotlib axis
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - save_path: Path to save the figure.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if log_scaled_plots:
        base, ext = os.path.splitext(save_path)
        save_path_log = f"{base}_log{ext}"
        ax.set_yscale('log')

        if save_path:
            fig.savefig(save_path_log, bbox_inches="tight")

    if save_path_png:
        fig.savefig(save_path_png, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_histogram(samples, title, save_path=None, save_path_png=None, posterior_type=None, value=None, xlim=(-20, 40)):
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
            
        elif (posterior_type == "MvNormal" or posterior_type == "MvStudentT") and samples.shape[1] > 2:
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

        if xlim is not None:
            plt.xlim(xlim)

    if save_path:

        plt.savefig(save_path, bbox_inches="tight")
        plt.savefig(save_path_png, bbox_inches="tight")

        metadata_path = save_path_png.replace(".png", ".json")
        with open(metadata_path, "w") as f:
            json.dump({"varying_value": value}, f)

        plt.close()
    else:
        plt.show()



def describe_idata(idata, title):
    print(f"\n=== {title} ===")
    for group in idata._groups:
        if getattr(idata, group) is None:
            continue
        ds = getattr(idata, group)
        print(f"{group}:")
        for var in ds.data_vars:
            dims = ds[var].dims
            shape = ds[var].shape
            print(f"  {var} dims={dims} shape={shape}")
    print()



def plot_all_dims_combined(mcmc_trace, iid_samples, var_name="posterior"):
    """
    Compare the marginal of all dimensions combined in one histogram+KDE.
    """

    num_dims = mcmc_trace.posterior[var_name].shape[-1]
    height_per_dim = 0.35
    base_height = 4
    fig_height = min(20, base_height + num_dims * height_per_dim)

    fig = plt.figure(figsize=(20, fig_height), constrained_layout=True)
    outer_spec = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 1.5], figure=fig)

    # Left: forest plot
    ax0 = fig.add_subplot(outer_spec[0, 0])
    # Middle: ridge plot
    ax1 = fig.add_subplot(outer_spec[0, 1])

    agg_height = min(2 + np.sqrt(num_dims), 10)
    inner_spec = gridspec.GridSpecFromSubplotSpec(
        nrows=3,
        ncols=1,
        subplot_spec=outer_spec[0, 2],
        height_ratios=[1, agg_height, 1]
    )

    ax2 = fig.add_subplot(inner_spec[1, 0])

    if iid_samples.ndim == 2:
        iid_samples = iid_samples[np.newaxis, ...] 

    # Wrap in InferenceData
    idata_iid = az.from_dict(posterior={var_name: iid_samples})

    az.plot_forest(
        [idata_iid, mcmc_trace],
        model_names=["IID", "MCMC"],
        var_names=[var_name],
        combined=True,
        kind="forestplot",
        quartiles=True,
        hdi_prob=0.95,
        ax=ax0,
        colors=["steelblue", "crimson"]
    )
    ax0.set_title("Forest per Dimension")

    az.plot_forest(
            [idata_iid, mcmc_trace],
            model_names=["IID", "MCMC"],
            var_names= [var_name],
            combined=True,
            kind="ridgeplot",
            ax=ax1,
            colors=["steelblue", "crimson"]
    )

    ax1.set_title("Ridge per Dimension")

    # flatten *all* axes into one 1‐D array
    post = mcmc_trace.posterior[var_name]             
    mcmc_flat = post.values.flatten()
    iid_flat = iid_samples.flatten()

    # wrap flattened samples into new InferenceData objects
    idata_mcmc_flat = az.from_dict(posterior={"": mcmc_flat[np.newaxis, :]})
    idata_iid_flat = az.from_dict(posterior={"": iid_flat[np.newaxis, :]})

    # plot ridgeplot of aggregated marginals (right panel)
    az.plot_forest(
        [idata_iid_flat, idata_mcmc_flat],
        model_names=["IID", "MCMC"],
        var_names= [""],
        combined=True,
        kind="ridgeplot",
        ax=ax2,
        colors=["steelblue", "crimson"]
)

    ax2.set_title("aggregated marginals")

    return fig


def handle_trace_plots(eval_level, trace, iid_samples, sampler_name, varying_attribute, value, save_path=None, png_path=None, show=False, save_individual=False):
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
    fig = plot_all_dims_combined(trace, iid_samples)

    plt.suptitle(f"Marginal Comparison for {eval_level} ({sampler_name}, {varying_attribute} = {value})")
    plt.tight_layout()

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if png_path:
        plt.savefig(png_path, bbox_inches="tight")
        metadata_path = png_path.replace(".png", ".json")
        meta = {
            "varying_value": value,       # e.g. 0.10, 0.25, …
            "eval_level":   eval_level,   # "pooled" or "chain"
            "sampler":      sampler_name  # "HMC", "NUTS", …
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f)
    plt.close()

    # Plot per dimension
    if posterior_array.ndim == 3 and dim > 1 and save_individual:
        for i in range(dim):
            dim_i = posterior_array[..., i]
            fig = az.plot_trace({f"posterior_{i}": dim_i}, compact=True)
            title = f"Trace Plot of posterior[{i}] {eval_level} ({sampler_name}, {varying_attribute} = {value})"

            plt.suptitle(title)
            plt.tight_layout()

            if show:
                plt.show()

            if save_path:
                filename = save_path.replace(".pdf", f"_dim_{i}.pdf")
                plt.savefig(filename, bbox_inches="tight")

            plt.close()



def plot_pairwise_scatter(
    eval_level,
    mcmc_trace,
    iid_samples,
    sampler_name,
    varying_attribute,
    value,
    var_name="posterior",
    marker_size=6,
    marker_opacity=0.6,
    html_path=None,
):
    
    dims= (0, 1)  # default to first two dimensions
    
    #  extract and flatten MCMC samples
    post = mcmc_trace.posterior[var_name].values
    # post.shape == (n_chains, n_draws, D)
    n_chains, n_draws, D = post.shape
    if max(dims) >= D:
        raise ValueError(f"Requested dims {dims}, but posterior has only {D} dimensions.")
    
    mcmc_flat = post.reshape(n_chains * n_draws, D)
    x_mcmc = mcmc_flat[:, dims[0]]
    y_mcmc = mcmc_flat[:, dims[1]]

    max_pts = 1000
    if len(x_mcmc) > max_pts:
        idx = np.random.choice(len(x_mcmc), max_pts, replace=False)
        x_mcmc, y_mcmc = x_mcmc[idx], y_mcmc[idx]

    #  flatten IID samples
    iid_arr = np.asarray(iid_samples)
    if iid_arr.ndim != 2 or iid_arr.shape[1] <= max(dims):
        raise ValueError(
            f"IID samples should be 2D with >= {max(dims)+1} columns; got shape {iid_arr.shape}."
        )
    iid_flat = iid_arr.reshape(-1, iid_arr.shape[1])
    x_iid = iid_flat[:, dims[0]]
    y_iid = iid_flat[:, dims[1]]

    if len(x_iid) > max_pts:
        idx = np.random.choice(len(x_iid), max_pts, replace=False)
        x_iid, y_iid = x_iid[idx], y_iid[idx]

    # build Plotly figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_mcmc,
            y=y_mcmc,
            mode="markers",
            name="MCMC",
            marker=dict(color="crimson", opacity=marker_opacity, size=marker_size),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_iid,
            y=y_iid,
            mode="markers",
            name="IID",
            marker=dict(color="steelblue", opacity=marker_opacity, size=marker_size),
        )
    )

    fig.update_layout(
        title=f" {eval_level} ({sampler_name}, {varying_attribute} = {value}  : dim {dims[0]} vs {dims[1]}",
        xaxis_title=f"Dimension {dims[0]}",
        yaxis_title=f"Dimension {dims[1]}",
        legend=dict(
            itemclick="toggle",          # click to show/hide one sampler
            itemdoubleclick="toggleothers"  # dbl-click to isolate one
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        template="simple_white",
    )

    # save standalone HTML
    if html_path:
        fig.write_html(
            html_path,
            include_plotlyjs="cdn",
            full_html=True,
            config={"responsive": True},
        )
        
        metadata_path = html_path.replace(".html", ".json")
        meta = {
            "varying_value": value,       # e.g. 0.10, 0.25, …
            "eval_level":   eval_level,   # "pooled" or "chain"
            "sampler":      sampler_name  # "HMC", "NUTS", …
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f)
    plt.close()

    

def save_sample_info(sample_info, json_path, plot_path, png_path=None, varying_attribute=None, value=None, label="Samples", case=None):
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
        ax.set_title(f"{label} & Means ({varying_attribute} = {value})")
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
    if png_path:
        plt.savefig(png_path, bbox_inches="tight")
        metadata_path = png_path.replace(".png", ".json")
        with open(metadata_path, "w") as f:
            json.dump({"varying_value": value}, f)
    plt.close(fig)


