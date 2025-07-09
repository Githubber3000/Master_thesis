from __future__ import annotations
import json, os
import logging
from jinja2 import Environment, FileSystemLoader 
from glob import glob
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az

from .utils   import extract_varying_value_from_json, safe_json_dump

logger = logging.getLogger(__name__)

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
      "wasserstein_distance",
      "mmd",
      "mmd_rff",
      "r_hat",
      "ess",
      "runtime",
    ]
    metrics = [
      m for m in MASTER_ORDER
      if (m != "mmd"     or do_mmd)
     and (m != "mmd_rff" or do_mmd_rff)
    ]

    # Which Glass’s Δ plots to look for:
    glass_keys = ["ws"]                      
    if do_mmd:     glass_keys.append("mmd")  
    if do_mmd_rff: glass_keys.append("mmd_rff")

    groups_data = []

    for group_name, configs in experiments:
        config_entries = []

        for config in configs:
            config_descr = config["config_descr"]
            
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
                "glass_plot_paths_pooled"  : collect_glass_pngs(pooled_png_base, glass_keys),
                "scatter_plot_paths_pooled": collect_scatter_pngs(pooled_png_base, metrics),
                "scatter_html_paths_pooled": collect_scatter_htmls(pooled_png_base, metrics),

                # chain  (may be None)
                "metric_plot_paths_chain" : collect_metric_pngs(chain_png_base, metrics),
                "glass_plot_paths_chain"  : collect_glass_pngs(chain_png_base, glass_keys),
                "scatter_plot_paths_chain": collect_scatter_pngs(chain_png_base, metrics),
                "scatter_html_paths_chain": collect_scatter_htmls(chain_png_base, metrics),

                "iid_kde_plot_paths": rel_kde_paths,
                "pooled_init_plot_paths": rel_pooled_init_paths,
                "chain_init_plot_paths": rel_chain_init_paths,
                "kde_init_triples": list(zip(rel_kde_paths, rel_pooled_init_paths, rel_chain_init_paths)),
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
    metrics = ["wasserstein_distance", "r_hat", "ess", "runtime"]

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

    # Set dynamic axis labels and save plots
    attribute_label = varying_attribute.replace("_", " ").title()

    for metric in metrics:
        fig, ax = fig_ax_pairs[metric]
        finalize_and_save_plot(fig,ax, attribute_label, metric, 
                               f"{metric} for Samplers (config =_{config_descr})",
                               os.path.join(plots_folder, f"{metric}_run_{run_id}.pdf"))
        

def compute_and_save_global_metrics(df_all_runs, sampler_colors, varying_attribute, varying_values, runs, num_chains, config_descr, global_results_folder, global_plots_folder, png_folder, iid_ref_stats_dict, save_extra_scatter, do_mmd, do_mmd_rff):
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
    metrics = ["wasserstein_distance","r_hat", "ess", "runtime"]

    if do_mmd:
        metrics.append("mmd")
    if do_mmd_rff:
        metrics.append("mmd_rff")

    attribute_label = varying_attribute.replace("_", " ").title()

    # New figure set (line + fill)
    fig_ax_pairs_shaded = {metric: plt.subplots(figsize=(10, 6)) for metric in metrics}
    fig_g, ax_g = plt.subplots(figsize=(10, 6))  # Glass delta for wasserstein_distance

    if do_mmd:
        fig_g_mmd, ax_g_mmd = plt.subplots(figsize=(10, 6))  # Glass delta for mmd
    if do_mmd_rff:
        fig_g_mmd_rff, ax_g_mmd_rff = plt.subplots(figsize=(10, 6)) # Glass delta for mmd_rff


    global_avg_dfs = {}
    scatter_data = {}

    # Load IID reference statistics
    iid_means_dict_swd = {}
    iid_stds_dict_swd = {}
    iid_medians_dict_swd = {}
    iid_q25_dict_swd = {}
    iid_q75_dict_swd = {}

    if do_mmd:
        iid_means_dict_mmd = {}
        iid_stds_dict_mmd = {}
        iid_medians_dict_mmd = {}
        iid_q25_dict_mmd = {}
        iid_q75_dict_mmd = {}

    if do_mmd_rff:
        iid_means_dict_mmd_rff = {}
        iid_stds_dict_mmd_rff = {}
        iid_medians_dict_mmd_rff = {}
        iid_q25_dict_mmd_rff = {}
        iid_q75_dict_mmd_rff = {}

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

        if do_mmd:
            iid_means_dict_mmd[k] = iid_entry["mean_mmd"]
            iid_stds_dict_mmd[k] = iid_entry["std_mmd"]
            iid_medians_dict_mmd[k] = iid_entry["median_mmd"]
            iid_q25_dict_mmd[k] = iid_entry["q25_mmd"]
            iid_q75_dict_mmd[k] = iid_entry["q75_mmd"]

        if do_mmd_rff:
            iid_means_dict_mmd_rff[k] = iid_entry["mean_mmd_rff"]
            iid_stds_dict_mmd_rff[k] = iid_entry["std_mmd_rff"]
            iid_medians_dict_mmd_rff[k] = iid_entry["median_mmd_rff"]
            iid_q25_dict_mmd_rff[k] = iid_entry["q25_mmd_rff"]
            iid_q75_dict_mmd_rff[k] = iid_entry["q75_mmd_rff"]


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
                ax_shaded.annotate("'DEMetropolis' r-hat skipped due to invalid values", 
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha="right", va="bottom", fontsize=9, color="red")
                continue

            if metric == "r_hat":
                if df_pivot.isnull().values.any() or  (df_pivot > 1000).any().any():
                    logger.warning(f"Skipping r_hat plot for sampler {sampler} due to extremely high values.")                    
                    ax_shaded.annotate("'DEMetropolis' r-hat skipped due to >1000", 
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha="right", va="bottom", fontsize=9, color="red")
                    continue
            
            
            # Compute mean+std and median+quantiles 
            means = df_pivot.mean(axis=1)
            medians = df_pivot.median(axis=1)
            q25 = df_pivot.quantile(0.25, axis=1)
            q75 = df_pivot.quantile(0.75, axis=1)

            # Custom ordering based on config (only if needed)
            if isinstance(medians.index[0], str):
                custom_order = [str(t) for t in varying_values]
                means = means.reindex(custom_order) 
                medians = medians.reindex(custom_order)
                q25 = q25.reindex(custom_order)
                q75 = q75.reindex(custom_order)
                
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


            if save_extra_scatter:
                # x = v.attr value repeated per run
                xs, ys = np.broadcast_to(
                    df_pivot.index.to_numpy()[:, None], df_pivot.shape
                ).ravel(), df_pivot.to_numpy().ravel()

                # save data for a per-sampler scatter figure later
                scatter_data.setdefault(metric, []).append(
                    (sampler, xs, ys, color)
                )

            # Save global avg for CSV
            if sampler not in global_avg_dfs:
                global_avg_dfs[sampler] = {}
            global_avg_dfs[sampler][metric] = (medians, q25, q75)

            # Compute glass delta for wasserstein_distance only
            if metric == "wasserstein_distance":
                # Get IID mean and std for this varying attribute value

                iid_mean_swd = pd.Series(
                    [iid_means_dict_swd[k] for k in means.index],
                    index=means.index,                   
                    name="iid_mean_swd"
                )
                iid_std_swd = pd.Series(
                    [iid_stds_dict_swd[k] for k in means.index],
                    index=means.index,
                    name="iid_std_swd"
                )

                # Glass Δ
                glass_delta = (means - iid_mean_swd) / iid_std_swd.replace(0, np.nan)
        
                global_avg_dfs[sampler]["ws_dist_glass_delta"] = glass_delta
                global_avg_dfs[sampler]["ws_dist_mcmc_mean"]   = means          
                global_avg_dfs[sampler]["ws_dist_iid_mean"]    = iid_mean_swd   
                global_avg_dfs[sampler]["ws_dist_iid_std"]     = iid_std_swd 

                # Plot glass delta for this sampler
                ax_g.plot(means.index, glass_delta, "o-", label=sampler, color=color)
            
            elif metric == "mmd":
                # Get IID mean and std for this varying attribute value


                iid_mean_mmd = pd.Series(
                    [iid_means_dict_mmd[k] for k in means.index],
                    index=means.index,                    
                    name="iid_mean_mmd"
                )

                iid_std_mmd = pd.Series(
                    [iid_stds_dict_mmd[k] for k in means.index],
                    index=means.index,
                    name="iid_std_mmd"
                )

                # Glass Δ
                glass_delta = (means - iid_mean_mmd) / iid_std_mmd.replace(0, np.nan)

                global_avg_dfs[sampler]["mmd_glass_delta"] = glass_delta
                global_avg_dfs[sampler]["mmd_mcmc_mean"] = means
                global_avg_dfs[sampler]["mmd_iid_mean"]  = iid_mean_mmd
                global_avg_dfs[sampler]["mmd_iid_std"]   = iid_std_mmd

                # Plot glass delta for this sampler
                ax_g_mmd.plot(means.index, glass_delta, "o-", label=sampler, color=color)

            elif metric == "mmd_rff":
                # Get IID mean and std for this varying attribute value

                iid_mean_mmd_rff = pd.Series(
                    [iid_means_dict_mmd_rff[k] for k in means.index],
                    index=means.index,                    
                    name="iid_mean_mmd_rff"
                )

                iid_std_mmd_rff = pd.Series(
                    [iid_stds_dict_mmd_rff[k] for k in means.index],
                    index=means.index,
                    name="iid_std_mmd_rff"
                )

                # Glass Δ
                glass_delta = (means - iid_mean_mmd_rff) / iid_std_mmd_rff.replace(0, np.nan)

                global_avg_dfs[sampler]["mmd_rff_glass_delta"] = glass_delta
                global_avg_dfs[sampler]["mmd_rff_mcmc_mean"] = means
                global_avg_dfs[sampler]["mmd_rff_iid_mean"]  = iid_mean_mmd_rff
                global_avg_dfs[sampler]["mmd_rff_iid_std"]   = iid_std_mmd_rff

                # Plot glass delta for this sampler
                ax_g_mmd_rff.plot(means.index, glass_delta, "o-", label=sampler, color=color)


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

        elif metric == "mmd":

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

        elif metric == "mmd_rff":

            iid_medians = np.array([iid_medians_dict_mmd_rff[k] for k in medians.index])
            iid_q25 = np.array([iid_q25_dict_mmd_rff[k] for k in medians.index])
            iid_q75 = np.array([iid_q75_dict_mmd_rff[k] for k in medians.index])

            ax_shaded.plot(medians.index, iid_medians, "o--", label="IID Reference", color="black")
            ax_shaded.fill_between(
                medians.index,
                iid_q25,
                iid_q75,
                color="black",
                alpha=0.1,
            )


        #  scatter plot for this metric
        if save_extra_scatter and metric in scatter_data:
            fig = go.Figure()

            for sampler, xs, ys, color in scatter_data[metric]:
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=sampler,
                    marker=dict(color=color, opacity=0.6, size=6)
                ))

            fig.update_layout(
                title=f"All runs {metric.replace('_',' ').title()} ({runs} runs, {config_descr})",
                xaxis_title=attribute_label,
                yaxis_title=metric.replace('_',' ').title(),
                legend=dict(
                    itemclick="toggle",                     # click to show/hide one sampler
                    itemdoubleclick="toggleothers"          # dbl-click to isolate one
                ),
                width=None,
                height=None,
                margin=dict(l=40, r=20, t=50, b=40)
            )

            # Save out as a standalone HTML
            html_path = os.path.join(
                png_folder,
                f"{metric}_global_scatter_interactive.html"
            )

            fig.write_html(html_path, include_plotlyjs="cdn", config=dict(responsive=True), full_html=True)



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
            "ws_mcmc_mean":  metrics_dict["ws_dist_mcmc_mean"].values,
            "ws_iid_mean":   metrics_dict["ws_dist_iid_mean"].values,
            "ws_iid_std":    metrics_dict["ws_dist_iid_std"].values,
            
            **({"mmd_mcmc_mean": metrics_dict["mmd_mcmc_mean"].values,
                "mmd_iid_mean":  metrics_dict["mmd_iid_mean"].values,
                "mmd_iid_std":   metrics_dict["mmd_iid_std"].values}
            if do_mmd else {}),

            **({"mmd_rff_mcmc_mean": metrics_dict["mmd_rff_mcmc_mean"].values,
                "mmd_rff_iid_mean":  metrics_dict["mmd_rff_iid_mean"].values,
                "mmd_rff_iid_std":   metrics_dict["mmd_rff_iid_std"].values}
            if do_mmd_rff else {})
        })

        if "ws_dist_glass_delta" in metrics_dict:
            df_global_avg["ws_dist_glass_delta"] = metrics_dict["ws_dist_glass_delta"].values
        if "mmd_glass_delta" in metrics_dict:
            df_global_avg["mmd_glass_delta"] = metrics_dict["mmd_glass_delta"].values
        if "mmd_rff_glass_delta" in metrics_dict:
            df_global_avg["mmd_rff_glass_delta"] = metrics_dict["mmd_rff_glass_delta"].values

        csv_filename = os.path.join(global_results_folder, f"Global_results_{sampler}.csv")
        df_global_avg.to_csv(csv_filename, index=False)

    # Save plots
    for metric in metrics:

        title = (f"Averaged {metric.replace('_', ' ').title()} "
                f"({runs} Runs, config = {config_descr})")   
        fig_shaded, ax_shaded = fig_ax_pairs_shaded[metric]
        pdf_path = os.path.join(global_plots_folder, f"{metric}_global_plot_shaded.pdf")
        png_path = os.path.join(png_folder, f"{metric}_global_plot_shaded.png")

        finalize_and_save_plot(fig_shaded, ax_shaded, attribute_label, metric,
                               title, save_path=pdf_path, save_path_png=png_path)
        

    # Plot Glass's Δ for wasserstein_distance
    pdf_path = os.path.join(global_plots_folder, "glass_delta_ws_dist.pdf")
    png_path = os.path.join(png_folder, "glass_delta_ws.png")
    title_ws= f"Glass's Δ for Wasserstein Distance ({runs} Runs, config = {config_descr})"

    finalize_and_save_plot(fig_g, ax_g, xlabel=attribute_label, ylabel="Glass's Δ", title=title_ws,
                            save_path=pdf_path, save_path_png=png_path)

    if do_mmd:
        # Plot Glass's Δ for MMD
        pdf_path = os.path.join(global_plots_folder, "glass_delta_mmd.pdf")
        png_path = os.path.join(png_folder, "glass_delta_mmd.png")
        title_mmd = f"Glass's Δ for MMD ({runs} Runs, config = {config_descr})"

        finalize_and_save_plot(fig_g_mmd, ax_g_mmd, xlabel=attribute_label, ylabel="Glass's Δ", title=title_mmd,
                            save_path=pdf_path, save_path_png=png_path)
    if do_mmd_rff:
        # Plot Glass's Δ for MMD-RFF
        pdf_path = os.path.join(global_plots_folder, "glass_delta_mmd_rff.pdf")
        png_path = os.path.join(png_folder, "glass_delta_mmd_rff.png")
        title_mmd_rff = f"Glass's Δ for MMD-RFF ({runs} Runs, config = {config_descr})"

        finalize_and_save_plot(fig_g_mmd_rff, ax_g_mmd_rff, xlabel=attribute_label, ylabel="Glass's Δ", title=title_mmd_rff,
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


