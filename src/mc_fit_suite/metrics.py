from __future__ import annotations
import numpy as np
from scipy import stats as sp
import arviz as az


def get_scalar_rhat_and_ess(trace, compute_rhat=True):
    posterior_vars = [v for v in trace.posterior.data_vars if v.startswith("posterior")]
    if not posterior_vars:
        raise ValueError("No posterior variables found.")
    
    return (
        az.rhat(trace, var_names=posterior_vars).to_array().max().item() if compute_rhat else np.nan,
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



def median_heuristic(X, Y, rng=None):
    """
    Compute the median heuristic for the RBF bandwidth sigma.
    
    Parameters:
    - X: np.ndarray of shape (n, d)
    - Y: np.ndarray of shape (m, d)
    - rng: np.random.Generator, optional
    
    Returns:
    - sigma: float, median of pairwise distances
    """
    rng = rng or np.random.default_rng()

    Z = np.vstack([X, Y])
    N = Z.shape[0]

    # If the number of samples is too large, randomly sample 1000 points
    if N > 1000:
        idx = rng.choice(N, size=1000, replace=False)
        Z = Z[idx]
        N = 1000
    
    diffs = Z[:, None, :] - Z[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)      
    upper = dist[np.triu_indices(N, k=1)]

    return np.median(upper)


def compute_mmd_rff(X, Y, D=500, rng=None):
    """
    Computes the approximate Maximum Mean Discrepancy (MMD) using Random Fourier Features (RFF)
    between two sample sets X and Y.

    Parameters:
    - X: np.ndarray of shape (n, d) – sample set from distribution p(x)
    - Y: np.ndarray of shape (m, d) – sample set from distribution q(x)
    - D: int – number of random Fourier features
    - rng: np.random.Generator, optional – random number generator for reproducibility

    Returns:
    - mmd_rff: float – approximate MMD value
    """

    rng = rng or np.random.default_rng()
    
    sigma = median_heuristic(X, Y, rng=rng)

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

    # Step 4: Calculate MMD (Euclidean distance between embeddings)
    mmd_rff = np.linalg.norm(mu_p - mu_q)

    return mmd_rff


def compute_mmd(X, Y, rng=None):
    """
    Computes the Maximum Mean Discrepancy (MMD) 
    between two sample sets X and Y.

    Parameters:
    - X: np.ndarray of shape (n, d) – sample set from distribution p(x)
    - Y: np.ndarray of shape (m, d) – sample set from distribution q(x)
    - rng: np.random.Generator, optional – random number generator for reproducibility

    Returns:
    - mmd: float – MMD value
    """

    rng = rng or np.random.default_rng()
    
    sigma = median_heuristic(X, Y, rng=rng)

    n, d = X.shape
    m, _ = Y.shape

    # Step 1: Compute Kernel Matrices
    # K_xx[i,j] = exp(-||x_i - x_j||^2 / (2*sigma^2))
    XX_sq_dists = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    K_xx = np.exp(-XX_sq_dists / (2 * sigma**2))

    # K_yy[j,k] = exp(-||y_j - y_k||^2 / (2*sigma^2))
    YY_sq_dists = np.sum((Y[:, None, :] - Y[None, :, :])**2, axis=2)
    K_yy = np.exp(-YY_sq_dists / (2 * sigma**2))

    # K_xy[i,j] = exp(-||x_i - y_j||^2 / (2*sigma^2))
    XY_sq_dists = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)
    K_xy = np.exp(-XY_sq_dists / (2 * sigma**2))

    # Step 2: Calculate MMD^2 
    mmd2 = (
        K_xx.sum() / (n * n)
        + K_yy.sum() / (m * m)
        - 2 * K_xy.sum() / (n * m)
    )

    return np.sqrt(mmd2)  


def compute_summary_discrepancies(mcmc_samples, iid_samples):
    """
    Computes RMSE of mean and variance per dimension between MCMC and IID samples.
    
    Returns:
        mean_rmse: RMSE between per-dimension means
        var_rmse: RMSE between per-dimension variances
    """
    mcmc_mean = np.mean(mcmc_samples, axis=0)
    iid_mean = np.mean(iid_samples, axis=0)
    mean_rmse = np.sqrt(np.mean((mcmc_mean - iid_mean) ** 2))

    mcmc_var = np.var(mcmc_samples, axis=0)
    iid_var = np.var(iid_samples, axis=0)
    var_rmse = np.sqrt(np.mean((mcmc_var - iid_var) ** 2))

    return mean_rmse, var_rmse



def count_mode_transitions(trace):
    """
    Count cross-mode transitions for a symmetric bimodal posterior split at x1=0.
    Fixed settings: mode_margin = ±1.0, confirm_steps = 3 consecutive samples.
    """
    MODE_MARGIN   = 1.0   # distance from 0 required to be considered inside a mode
    CONFIRM_STEPS = 3     # need this many consecutive samples in the new mode to confirm

    x = np.asarray(trace)

    s = x if x.ndim == 1 else x[:, 0]  # use first coordinate as separating axis

    # mode_labels: -1 = left mode, 0 = between modes (margin), +1 = right mode
    mode_labels = np.zeros(len(s), dtype=int)
    mode_labels[s >=  MODE_MARGIN] =  1
    mode_labels[s <= -MODE_MARGIN] = -1

    print("min(s):", np.min(s), "max(s):", np.max(s))
    print("unique labels:", np.unique(mode_labels, return_counts=True))

    current_mode = None
    candidate_mode = None
    candidate_streak = 0
    transitions = 0

    for sample_mode in mode_labels:
        if current_mode is None:
            if sample_mode != 0:
                current_mode = sample_mode
            continue

        if sample_mode == 0 or sample_mode == current_mode:
            # still in same mode or in-between: cancel any pending switch
            candidate_mode = None
            candidate_streak = 0
            continue

        # sample_mode is the opposite mode
        if candidate_mode == sample_mode:
            candidate_streak += 1
        else:
            candidate_mode = sample_mode
            candidate_streak = 1

        if candidate_streak >= CONFIRM_STEPS:
            transitions += 1
            current_mode = sample_mode
            candidate_mode = None
            candidate_streak = 0

    return transitions

