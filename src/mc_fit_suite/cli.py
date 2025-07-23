from __future__ import annotations

import warnings, logging
import typer
import json
from pathlib import Path
from .runner import run_full_experiment 
from .config import load_experiment_settings, load_default_values

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

logging.getLogger("arviz").setLevel(logging.CRITICAL)
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

app = typer.Typer(help="MC-Fit Suite: create or re-run experiments")

@app.callback(invoke_without_command=True)
def main(    
    rerun: bool = typer.Option(
        False,
        "-r", "--rerun",
        help="Rerun an experiment you’ve already created",
        is_flag=True,
    )
):
    """
    Create a brand-new experiment, or re-run one you've already done (maybe with adapted settings in the experiment).
    """
    base = Path("experiments")
    template = Path("experiment_template")

    #  Gather existing experiment names
    existing_exps = sorted(p.name.removeprefix("exp_")
                      for p in base.glob("exp_*") if p.is_dir())
    
    if rerun:

        if not existing_exps:
            typer.echo("No existing experiments to rerun.")
            raise typer.Exit(1)
        typer.echo("Existing experiments:")
        for i, name in enumerate(existing_exps, start=1):
            typer.echo(f"  {i:>2}) {name}")
        choice = typer.prompt("Select experiment number to rerun")
        try:
            idx = int(choice) - 1
            experiment_name = existing_exps[idx]
        except:
            typer.echo("Invalid selection.")
            raise typer.Exit(1)
        

    else:

        experiment_name = typer.prompt("Experiment name").strip()
        if not experiment_name:
            typer.echo("You must enter a name.")
            raise typer.Exit(1)
        
        exp_path = base / f"exp_{experiment_name}"
        # keep asking until they give a non-existing one
        while exp_path.exists():
            typer.echo(f"Experiment '{experiment_name}' already exists.")
            experiment_name = typer.prompt(
                "Pick a new experiment name (must be different)"
            ).strip()
            if not experiment_name:
                typer.echo("You must enter a name.")
                raise typer.Exit(1)
            exp_path = base / f"exp_{experiment_name}"


    # Decide which configs to run (union of existing + template)
    exp_cfg_dir      = base / f"exp_{experiment_name}" / "configs"
    template_cfg_dir = template / "configs"

    # gather existing configs (if any)
    configs_in_exp = sorted(p.stem for p in exp_cfg_dir.glob("*.yaml")) if exp_cfg_dir.exists() else []

    # gather template stubs
    configs_in_templ = sorted(p.stem for p in template_cfg_dir.glob("*.yaml"))

    # union, preserving sort
    configs = sorted(dict.fromkeys(configs_in_exp + configs_in_templ))

    typer.echo("\nAvailable configs:")
    for i, name in enumerate(configs, start=1):
        marker = " *(from current experiment)*" if name in configs_in_exp else ""
        typer.echo(f"  {i:>2}) {name}{marker}")

    default = ""
    # default to all existing if rerun
    if rerun and configs_in_exp:
        default = " ".join(str(configs.index(n) + 1) for n in configs_in_exp)

    choice_str = typer.prompt(
        f"\nSelect config numbers to run for {experiment_name} (e.g. 1 3 5)",
        default=default
    ).strip()

    try:
        tokens       = choice_str.replace(",", " ").split()
        idxs         = [int(tok) - 1 for tok in tokens]
        config_names = [configs[i] for i in idxs]
    except Exception:
        typer.echo("Invalid selection.")
        raise typer.Exit(1)



    # Load & show the two YAMLs, then confirm
   
    if rerun:
        settings_path = base / f"exp_{experiment_name}" / "settings" / "experiment_settings.yaml"
        defaults_path = base / f"exp_{experiment_name}" / "default_vals" / "attribute_default_vals.yaml"
    else:
        settings_path = template / "settings"     / "experiment_settings.yaml"
        defaults_path = template / "default_vals" / "attribute_default_vals.yaml"

    experiment_settings = load_experiment_settings(settings_path)
    defaults            = load_default_values(defaults_path)

    typer.echo("\nExperiment settings:")
    typer.echo(typer.style(json.dumps(experiment_settings, indent=2), fg="yellow"))
    typer.echo("\nDefault values:")
    typer.echo(typer.style(json.dumps(defaults, indent=2), fg="yellow"))

    if not typer.confirm(f"\nProceed to run {experiment_name} with these settings?", default=True):
        typer.echo("Aborted by user.")
        raise typer.Exit(1)

    if rerun:
        typer.echo(f"\nRe-running '{experiment_name}' on configs: {config_names!r}\n")
    else:
        typer.echo(f"\nCreating new experiment '{experiment_name}' with configs: {config_names!r}\n")

    run_full_experiment(
        experiment_name=experiment_name,
        config_names=config_names,
    )

if __name__ == "__main__":
    app()