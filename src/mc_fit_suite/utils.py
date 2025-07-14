from __future__ import annotations
import logging, os, json
import numpy as np
import subprocess
import sys

logger = logging.getLogger(__name__)

def set_logging_level(level_name):
    """
    Configure the root logger once for the whole package / script.
    """

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
        
def get_folder_size(path='.'):
    """Compute total size of all files in directory."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total
        
def create_directories(*paths):
    """Creates multiple directories if they don't exist."""
    for path in paths:
        os.makedirs(path)

def rel_to_root(path, root):
    """Return *path* relative to *root* so that HTML <img> links keep working
    no matter where the whole experiment folder is moved."""

    return os.path.relpath(path, start=root)

def extract_varying_value_from_json(any_plot_path):

    base, _ext = os.path.splitext(any_plot_path)
    json_path = base + ".json"

    with open(json_path, "r") as f:
        data = json.load(f)

        v = data.get("varying_value")

        if isinstance(v, (int, float)):
            return v
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return v[0]
        return tuple(v) if isinstance(v, (list, tuple)) else str(v)
    

def load_meta(any_plot_path):

    base, _ext = os.path.splitext(any_plot_path)
    json_path = base + ".json"

    with open(json_path) as f:
        return json.load(f)


def build_trace_groups(pooled_plots, chain_plots, rel, type="trace"):
    """
    Return a list of groups, each containing one varying_value and
    a list of sampler blocks (with pooled + chain PNGs, if present).
    """

    sampler_order = ["HMC", "Metro", "SMC", "DEMetro_Z"]  

    pooled = {}  
    chain  = {}

    # --- populate pooled dict ----------------------------------------
    for p in pooled_plots:
        meta = load_meta(p)                      
        raw_value   = meta["varying_value"]
        value = tuple(raw_value) if isinstance(raw_value, (list, tuple)) else raw_value
        sampler = meta["sampler"]

        if value not in pooled:                   
            pooled[value] = {}
        pooled[value][sampler] = rel(p)

    # --- populate chain dict -----------------------------------------
    for p in chain_plots:
        meta = load_meta(p)
        raw_value   = meta["varying_value"]
        value = tuple(raw_value) if isinstance(raw_value, (list, tuple)) else raw_value
        sampler = meta["sampler"]

        if value not in chain:
            chain[value] = {}
        chain[value][sampler] = rel(p)

    # --- merge into ordered list of groups ---------------------------
    groups = []
    for value in sorted(pooled):   # all distinct values, sorted
        samplers_block = []
        for s in sampler_order:                      # fixed sampler order
            if s in pooled[value]:
                samplers_block.append({
                    "sampler":   s,
                    f"pooled_{type}": pooled.get(value, {}).get(s),
                    f"chain_{type}":  chain.get(value, {}).get(s)
                })
        groups.append({"varying_value": value, "samplers": samplers_block})

    return groups


def build_correlation_cov_matrix(dim, rho):
    cov = np.full((dim, dim), rho)
    np.fill_diagonal(cov, 1.0)
    return cov.tolist()

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
    elif (posterior_type == "MvNormal" or posterior_type== "MvStudentT")and "mu" in params:
        return len(params["mu"])
    else:
        raise ValueError(f"Cannot determine dimensionality for posterior type '{posterior_type}' with parameters: {params}")
    
    

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



