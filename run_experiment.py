import argparse
from pathlib import Path
from mc_fit_suite.runner import run_full_experiment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name",    required=True, help="experiment name")
    p.add_argument("--configs", required=True, nargs="+",
                   help="YAML config files (without .yaml)")
    args = p.parse_args()

    run_full_experiment(
        experiment_name=args.name,
        config_names=args.configs,
    )

if __name__ == "__main__":
    main()
