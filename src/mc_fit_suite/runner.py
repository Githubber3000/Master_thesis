from __future__ import annotations
import os, shutil, traceback, humanize
import sys
import time
from tqdm import tqdm
from datetime import datetime
from .core       import run_experiment
from .reporting  import generate_html_report
from .utils      import (create_directories, get_folder_size)
from .config     import (load_config_file, load_default_values,
                         load_experiment_settings, apply_defaults_to_config,
                         get_experiment_paths, validate_config)



def run_full_experiment(                 
        experiment_name,
        config_names,
):
    """High-level driver that prepares folders, runs all configs,
    and writes the HTML + summary TXT."""

    template_dir = "experiment_template"
    output_root = "experiments"
    exp_root   = os.path.join(output_root, f"exp_{experiment_name}")
    results    = os.path.join(exp_root, "results")
    cfg_dir    = os.path.join(exp_root, "configs")
    png_root   = os.path.join(results, "z_html_pngs")

     # Create / clean directory tree
    if os.path.exists(exp_root):
        shutil.rmtree(results)
    else:
        create_directories(exp_root)

    create_directories(results)
    create_directories(png_root)

    # Copy experiment_template folders into experiment_root
    for subfolder in ["default_vals", "settings"]:
        src = os.path.join(template_dir, subfolder)
        dst = os.path.join(exp_root, subfolder)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    # Copy only the configs that are defined in config_names
    for config_name in config_names:
        src = os.path.join(template_dir, "configs", f"{config_name}.yaml")
        dst = os.path.join(exp_root, "configs", f"{config_name}.yaml")
        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copy(src, dst)
        else:
            print(f"Warning: Config file {src} not found!")

    # Load YAMLs & defaults
    settings_path = os.path.join(exp_root, "settings", "experiment_settings.yaml")
    defaults_path = os.path.join(exp_root, "default_vals", "attribute_default_vals.yaml")
    exp_paths = get_experiment_paths(config_names, base_dir=os.path.join(exp_root, "configs"))

    experiment_settings = load_experiment_settings(settings_path)
    defaults = load_default_values(defaults_path)

    experiments = []
    for path in exp_paths:
        group_name, config_list = load_config_file(path)
        config_list = [apply_defaults_to_config(cfg, defaults) for cfg in config_list]
        experiments.append((group_name, config_list))

    total_configs = sum(len(config_list) for _, config_list in experiments)
    total_posterior_variants = sum(len(config["varying_values"]) for _, config_list in experiments for config in config_list)
    runs_per_config = defaults.get("runs", 1)
    total_runs = sum(config["runs"] * len(config["varying_values"]) for _, exp_group in experiments for config in exp_group)


    print(f"Configurations: {total_configs}")
    print(f"Posterior Variants: {total_posterior_variants}")
    print(f"Default Runs per Posterior: {runs_per_config}")
    print(f"Total Runs: {total_runs}")

    # Validate all configurations before running the experiments
    for group_name, exp_group in experiments:
        for config in exp_group:
            validate_config(config)

    print("All configurations are valid. Starting experiments")
   
    start_time = time.time()
    start_dt = datetime.now()

    failed_configs = []

    with tqdm(total=total_runs, desc="Total experiment progress", file=sys.stdout, dynamic_ncols=True) as pbar:
        for group_name, exp_group in experiments:
            for config in exp_group:
                try:
                    # png folder for html report for each group and config
                    config_png_folder = os.path.join(results, "z_html_pngs", group_name, config["config_descr"])
                    create_directories(config_png_folder)

                    run_experiment(
                        results,
                        config_png_folder,
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
                        group_name=group_name,
                        progress_bar=pbar, 
                        # Pass remaining keys as posterior_kwargs
                        **{k: v for k, v in config.items() if k not in [
                            "config_descr", "runs", "varying_attribute", "varying_values", 
                            "num_samples", "num_chains", "init_scheme", 
                            "base_random_seed", "posterior_type"
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
            experiment_root_folder=exp_root,
            report_pngs_folder=png_root,
            experiments=experiments,
            output_path=os.path.join(exp_root, f"exp_{experiment_name}_report.html"),
            do_mmd= experiment_settings.get("do_mmd", False),
            do_mmd_rff= experiment_settings.get("do_mmd_rff", False),
        )

    size_bytes = get_folder_size(exp_root)

    summary_lines = [
        "\n============================",
        "Experiment Summary",
        "============================",
        f"Started at:                   {start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Finished at:                  {end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total duration:               {hours}h {minutes}m {seconds}s",
        f"Output folder:                {exp_root}",
        f"Output folder size:           {humanize.naturalsize(size_bytes)}",
        f"Total configurations:         {total_configs}",
        f"Successful configuration:     {total_configs - len(failed_configs)}",
        f"Failed configurations:        {len(failed_configs)}"
    ]

    if failed_configs:
        # summary_lines.append("\n Failed Configurations:")
        for cfg, msg in failed_configs:
            summary_lines.append(f" - {cfg}: {msg}")

    # Print to console
    print("\n".join(summary_lines))

    # Also save to summary.txt
    summary_path = os.path.join(results, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"Summary saved to: {summary_path}")
