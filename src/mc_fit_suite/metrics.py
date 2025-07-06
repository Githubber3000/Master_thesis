from __future__ import annotations
import numpy as np
from scipy import stats as sp
import arviz as az


def get_scalar_rhat_and_ess(trace):
    posterior_vars = [v for v in trace.posterior.data_vars if v.startswith("posterior")]
    if not posterior_vars:
        raise ValueError("No posterior variables found.")
    return (
        az.rhat(trace, var_names=posterior_vars).to_array().max().item(),
        az.ess(trace, var_names=posterior_vars).to_array().min().item()
    )


def sliced_wasserstein_distance(X, Y, L=None, rng=None):
    """
    Computes the sliced Wasserstein distance (SWD_p) between two sets of samples.
    
    Parameters:
    - X: numpy array of shape (N, d) -> first sample set
    - Y: numpy array of shape (N, d) -> second sample set
    - L: int, number of random projections
    
    Returns:
    - SWD_p: float, the sliced Wasserstein distance
    """

    rng = rng or np.random.default_rng()

     # Assuming X and Y have the same shape
    N, d = X.shape 
    # Accumulation variable
    S = 0  

    for _ in range(L):
        # Sample a random unit vector (projection direction)
        theta = rng.standard_normal(d)
        norm = np.linalg.norm(theta)
        if norm == 0:
            continue
        # Normalize to unit sphere
        theta /= norm  

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

def compute_mmd_rff(X, Y, D=None, sigma=1.0, rng=None):
    """
    Computes the approximate Maximum Mean Discrepancy (MMD) using Random Fourier Features (RFF)
    between two sample sets X and Y.

    Parameters:
    - X: np.ndarray of shape (n, d) – sample set from distribution p(x)
    - Y: np.ndarray of shape (m, d) – sample set from distribution q(x)
    - D: int – number of random Fourier features
    - sigma: float – bandwidth of the Gaussian kernel
    - rng: np.random.Generator, optional – random number generator for reproducibility

    Returns:
    - mmd_rff: float – approximate MMD value
    """

    rng = rng or np.random.default_rng()

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
