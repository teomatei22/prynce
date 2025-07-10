"""
Test script for MCMC and nested sampling
"""

import numpy as np
import matplotlib.pyplot as plt
from model import BBNModel, ln_posterior, PRIORS
import emcee
from emcee.autocorr import integrated_time
import multiprocessing as mp


def calculate_autocorr_thinning(chain, min_thin=1, max_thin=None):
    """
    Calculate appropriate thinning based on autocorrelation time.
    
    Parameters
    ----------
    chain : array
        MCMC chain (can be 1D or 2D)
    min_thin : int
        Minimum thinning factor
    max_thin : int, optional
        Maximum thinning factor (if None, no limit)
        
    Returns
    -------
    thin_factor : int
        Recommended thinning factor
    tau : float
        Integrated autocorrelation time
    """
    # Ensure chain is 2D for emcee
    if chain.ndim == 1:
        chain_2d = chain.reshape(-1, 1)
    else:
        chain_2d = chain
    
    try:
        # Calculate integrated autocorrelation time
        tau = integrated_time(chain_2d, c=5.0, quiet=True)
        
        # For 1D chain, tau is a scalar; for multi-dimensional, take max
        if np.isscalar(tau):
            tau_max = tau
        else:
            tau_max = np.max(tau)
        
        # Recommended thinning: 2 * autocorrelation time
        # This ensures samples are approximately independent
        thin_factor = max(min_thin, int(2 * tau_max))
        
        # Apply maximum thinning limit if specified
        if max_thin is not None:
            thin_factor = min(thin_factor, max_thin)
            
        return thin_factor, tau_max
        
    except Exception as e:
        print(f"Warning: Could not calculate autocorrelation time: {e}")
        print("Using default thinning factor of 1")
        return min_thin, np.nan


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
    
    chain = sampler.get_chain(flat=True)[:, 0]
    return chain


def test_single_dispersion(disp=1, nwalkers=8, nsteps=1000, npoints=10, seed=42):
    """
    Test both sampling methods on a single dispersion type with reduced parameters.
    """
    print(f"Testing dispersion {disp} with reduced parameters...")
    print("=" * 50)
    
    # Test MCMC
    print("Running MCMC...")
    mcmc_chain = run_mcmc_sampling(disp, nwalkers, nsteps, seed)
    print(f"MCMC: {len(mcmc_chain)} samples")
    burn_in = int(0.10 * len(mcmc_chain))
    chain_post = mcmc_chain[burn_in:]

    # Calculate autocorrelation-based thinning
    print("Calculating autocorrelation time...")
    thin_factor, tau = calculate_autocorr_thinning(chain_post, min_thin=1, max_thin=100)
    print(f"Autocorrelation time: {tau:.2f}")
    print(f"Recommended thinning factor: {thin_factor}")
    
    # Apply thinning based on autocorrelation time
    if thin_factor > 1:
        indices = np.arange(0, len(chain_post), thin_factor)
        chain_thinned = chain_post[indices]
        print(f"Applied autocorrelation-based thinning: {len(chain_post)} → {len(chain_thinned)} samples")
    else:
        print("No thinning applied (thin_factor = 1)")    
    
    return mcmc_chain 


if __name__ == "__main__":
    
    for i in (1,2,3):
        test_single_dispersion(disp=i) 

    # test_single_dispersion(disp=3, nwalkers=8, nsteps=3000, seed=42)