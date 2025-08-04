from __future__ import annotations
import yaml, os
from .utils import safe_json_dump
import numpy as np


def load_config_file(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    group_name = data["group_name"]
    configs = data["configs"]
 
    for cfg in configs:

        if cfg.get("posterior_type") == "Circle":
            num_components = cfg["component_number"]
            radius = cfg["radius"]
            cov = cfg.get("cov", [[1.0, 0.0], [0.0, 1.0]])
            weight_type = cfg.get("weights", "equal")
            component_type = cfg.get("component_type", "MvNormal")

            component_params, _, weights = adjust_circle_layout(num_components, component_type, radius, cov, weight_type)

            cfg["posterior_type"] = "Mixture"
            cfg["component_params"] = component_params
            cfg["component_types"] = [component_type] * num_components
            cfg["weights"] = weights

            cfg["radius"] = radius
            cfg["component_number"] = num_components
            cfg["component_type"] = component_type
            cfg["cov"] = cov
            cfg["weight_type"] = weight_type 

            print(f"Transformed Circle config '{cfg['config_descr']}' into Mixture with {len(cfg['component_params'])} components")
            print("First few component means:")
            for i, cp in enumerate(cfg["component_params"][:3]):
                print(f"  Component {i}: mu = {cp['mu']}, cov = {cp['cov']}")
        
        #Top-level convert mu/loc (for SinglePosterior)
        for key in ("mu", "loc"):
            if key in cfg:
                cfg[key] = parse_mu_entry(cfg[key])

        # Top-level convert cov/scale (for SinglePosterior)
        for key in ("cov", "scale"):
            if key in cfg:
                cfg[key] = parse_cov_entry(cfg[key])
                
        # cov/scale inside component_params (for MixturePosterior)
        if "component_params" in cfg:
            for component in cfg["component_params"]:
                for key in ("cov", "scale"):
                    if key in component:
                        component[key] = parse_cov_entry(component[key])

        if "component_params" in cfg:
            for component in cfg["component_params"]:
                for key in ("mu", "loc"):
                    if key in component:
                        component[key] = parse_mu_entry(component[key])

        if "varying_values" in cfg:
            # If we’re varying mu/loc, parse each spec string into a vector
            if cfg["varying_attribute"] in ("mu", "loc"):
                cfg["varying_values"] = [
                    parse_mu_entry(v) for v in cfg["varying_values"]
                ]

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


def adjust_mode_means(component_params, d, r, direction=None):

    mu1 = np.zeros(d)
    mu2 = r * np.array(direction)

    print(f"Adjusting mode means for {len(component_params)} components: mu1 = {mu1}, mu2 = {mu2}")

    if "mu" in component_params[0]:
        component_params[0]["mu"] = mu1
        component_params[1]["mu"] = mu2
    elif "loc" in component_params[0]:
        component_params[0]["loc"] = mu1
        component_params[1]["loc"] = mu2



def adjust_dimension_of_kwargs(posterior_type, kwargs_dict_copy, kwargs_dict, target_dim, required_parameters):
    """
    Adjusts only the required vector-like entries in the dictionary to match the given dimension.
    For mixtures, this is applied recursively to each component.

    Parameters:
    - posterior_type: str (e.g., "Normal", "MvNormal", "Mixture")
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

    # Exclude parameters that do not depend on dimension
    dimension_invariant_keys = {"nu"} 
    adaptable_keys = [k for k in required_keys if k not in dimension_invariant_keys]

    for key in adaptable_keys:
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



def adjust_circle_layout(num_components, component_type, radius, cov, weight_type):
    mus = [
        [
            radius * np.cos(2 * np.pi * k / num_components),
            radius * np.sin(2 * np.pi * k / num_components),
        ]
        for k in range(num_components)
    ]

    if weight_type == "equal":
        weights = [1.0 / num_components] * num_components

    component_params = [{"mu": mu, "cov": cov} for mu in mus]

    component_types = [component_type] * num_components

    return component_params, component_types, weights


def save_adjusted_posterior_config(posterior_kwargs, folder, dim_value):
    json_path = os.path.join(folder, f"posterior_config_dim_{dim_value}.json")
    safe_json_dump(posterior_kwargs, json_path)


def parse_mu_entry(mu_entry):
    """
    Parse a mu entry specifier into a list of floats.

    Supported specifiers:
    - "ZERO_<n>":    returns an n-dimensional zero vector
    - "FIRST_<v>_<n>": returns an n-dimensional vector with all zeros
                       except the first element set to v

    If mu_entry is not a str, it is returned unchanged.
    """

    if not isinstance(mu_entry, str):
        return mu_entry

    if mu_entry.startswith("ZERO_"):
        n = int(mu_entry.split("_")[1])
        return [0.0] * n
    
    if mu_entry.startswith("FIRST_"):
        _, value_str, dim_str = mu_entry.split("_", 2)
        v = float(value_str)
        n = int(dim_str)
        vec = [0.0] * n
        vec[0] = v
        return vec

    raise ValueError(f"Unknown mu specifier: {mu_entry}")


def parse_cov_entry(cov_entry):

    if not isinstance(cov_entry, str):
        return cov_entry
    
    if cov_entry.startswith("ID_"):
        n = int(cov_entry.split("_")[1])
        return np.eye(n).tolist()
    
    if cov_entry.startswith("HIGH_"):
            n = int(cov_entry.split("_")[1])
            rho = 0.9
            cov = np.full((n, n), rho)
            np.fill_diagonal(cov, 1.0)
            return cov.tolist()

    raise ValueError(f"Unknown covariance specifier: {cov_entry}")


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
        "MvStudentT": {"nu", "mu", "scale"},
        "Custom": {"logp_func"}
    }

    OPTIONAL_ATTRIBUTES = {"base_random_seed", "init_scheme", "varying_component", "dimension", "correlation", "circle_radius", "circle_modes", "mm"}

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
    
    if varying_attr == "dimension":
        max_dim = max(config["varying_values"])

        for key in ["mu", "sigma", "cov", "scale"]: 
            val = config.get(key)

            if val is None:
                continue

            if max_dim >= 2 and not isinstance(val, list):
                raise ValueError(
                    f"Parameter '{key}' in config '{config['config_descr']}' must be a list when varying dimension ≥ 2 (got scalar: {val})."
                )

            if isinstance(val, list) and len(val) < max_dim:
                raise ValueError(f"Parameter '{key}' in config '{config['config_descr']}' is too short for max dimension {max_dim}")
  
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