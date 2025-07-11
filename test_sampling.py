"""
Test script for MCMC and nested sampling
"""

import numpy as np
import matplotlib.pyplot as plt
from model import BBNModel, ln_posterior, PRIORS
import emcee
import multiprocessing as mp
from emcee.autocorr import integrated_time
import pandas as pd


def get_autocorr_time(chain):
    """
    Calculate autocorrelation time for determining thinning factor.
    
    Parameters
    ----------
    chain : array
        MCMC chain
        
    Returns
    -------
    n_effective : int
        Number of effective samples
    """
    try:
        tau = integrated_time(chain, c=5, tol=0)
        if tau is None:
            return len(chain) // 10
        if np.isscalar(tau):
            return max(1, int(len(chain) / float(tau)))
        else:
            tau_max = float(np.max(tau))
            return max(1, int(len(chain) / tau_max))
    except Exception as e:
        print(f"Warning: Could not calculate autocorrelation time: {e}")
        return len(chain) // 10  # Default to 10% of samples


def simple_thinning(samples, n_select, method='random'):
    """
    Simple thinning methods for sample selection.
    
    Parameters
    ----------
    samples : array-like
        Full set of samples
    n_select : int
        Number of samples to select
    method : str
        Thinning method: 'random', 'systematic', 'autocorr'
        
    Returns
    -------
    selected_indices : array
        Indices of selected samples
    selected_samples : array
        Selected samples
    """
    samples = np.asarray(samples).flatten()
    n_total = len(samples)
    
    if n_select >= n_total:
        return np.arange(n_total), samples
    
    if method == 'random':
        # Random selection
        indices = np.random.choice(n_total, n_select, replace=False)
        indices.sort()  # Keep original order
    elif method == 'systematic':
        # Systematic sampling
        step = n_total // n_select
        indices = np.arange(0, n_total, step)[:n_select]
    elif method == 'autocorr':
        # Based on autocorrelation time
        tau_eff = get_autocorr_time(samples)
        step = max(1, tau_eff // 2)
        indices = np.arange(0, n_total, step)[:n_select]
    else:
        raise ValueError(f"Unknown thinning method: {method}")
    
    selected_samples = samples[indices]
    return indices, selected_samples


def run_mcmc_sampling(disp, nwalkers=7, nsteps=1000, seed=None):
    """
    Run MCMC sampling for a given dispersion type.
    
    Parameters
    ----------
    disp : int
        Dispersion type (1, 2, or 3)
    nwalkers : int
        Number of walkers for emcee
    nsteps : int
        Number of steps for each walker
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    chain : numpy.ndarray
        Flattened chain of samples
    log_likelihoods : numpy.ndarray
        Log-likelihood values for each sample
    """
    if seed is not None:
        np.random.seed(seed)
    
    model = BBNModel()
    rng = np.random.default_rng()
    
    lo, hi = PRIORS[disp]
    p0 = rng.uniform(lo, hi, size=(nwalkers, 1))
    
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, 1, ln_posterior, args=(model, disp), pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=True)
    
    chain = sampler.get_chain(flat=True)
    if chain is not None:
        # Get log-likelihoods
        log_prob = sampler.get_log_prob(flat=True)
        return chain[:, 0], log_prob
    else:
        return np.array([]), np.array([])


def test_single_dispersion(disp=1, nwalkers=8, nsteps=1000, npoints=10, seed=42):
    """
    Test both sampling methods on a single dispersion type with reduced parameters.
    """
    print(f"Testing dispersion {disp} with reduced parameters...")
    print("=" * 50)
    
    # Test MCMC
    print("Running MCMC...")
    mcmc_chain, log_likelihoods = run_mcmc_sampling(disp, nwalkers, nsteps, seed)
    print(f"MCMC: {len(mcmc_chain)} samples")
    
    if len(mcmc_chain) == 0:
        print("Warning: MCMC chain is empty!")
        return mcmc_chain, log_likelihoods
    
    # Apply burn-in
    burn_in = int(0.10 * len(mcmc_chain))
    chain_post = mcmc_chain[burn_in:]
    log_likelihoods_post = log_likelihoods[burn_in:]
    
    # Calculate autocorrelation time to determine thinning factor
    print("Calculating autocorrelation time...")
    n_effective = get_autocorr_time(chain_post)
    n_thin = min(n_effective, len(chain_post) // 10)  # Use 10% of samples or effective samples
    print(f"Autocorrelation suggests {n_effective} effective samples")
    print(f"Will select {n_thin} samples for thinning")
    
    # Apply thinning
    print("Applying thinning...")
    selected_indices, chain_thinned = simple_thinning(
        chain_post, n_thin, method='autocorr'
    )
    log_likelihoods_thinned = log_likelihoods_post[selected_indices]
    
    print(f"Thinning: {len(chain_post)} → {len(chain_thinned)} samples")
    
    # Save thinned samples as CSV
    print("Saving thinned data to CSV...")
    df_thinned = pd.DataFrame({
        'param': chain_thinned,
        'log_likelihood': log_likelihoods_thinned,
        'sample_index': selected_indices + burn_in  # Original indices in full chain
    })
    
    csv_filename = f"dispersion_{disp}_thinned.csv"
    df_thinned.to_csv(csv_filename, index=False)
    print(f"Saved {len(df_thinned)} thinned samples to {csv_filename}")
    
    # Also save full chain for reference
    df_full = pd.DataFrame({
        'param': chain_post,
        'log_likelihood': log_likelihoods_post,
        'sample_index': np.arange(burn_in, len(mcmc_chain))
    })
    
    csv_full_filename = f"dispersion_{disp}_full.csv"
    df_full.to_csv(csv_full_filename, index=False)
    print(f"Saved {len(df_full)} full post-burnin samples to {csv_full_filename}")
    
    return chain_thinned, log_likelihoods_thinned


if __name__ == "__main__":
    
    # for i in (2,3):
    #     test_single_dispersion(disp=i) 

    test_single_dispersion(disp=3)