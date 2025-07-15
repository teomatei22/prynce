"""
Test script for MCMC and nested sampling
"""

import numpy as np
import matplotlib.pyplot as plt
from model import DispersionModel, ln_posterior, PRIORS
import emcee
import multiprocessing as mp
from emcee.autocorr import integrated_time
from joblib import Parallel, delayed
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
        return len(chain) // 10



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


    model = DispersionModel()

    lo, hi = PRIORS[disp]
    p0 = np.random.uniform(lo, hi, size=(nwalkers, 1))

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, 1, ln_posterior, args=(model, disp), pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    chain = sampler.get_chain(flat=True)
    if chain is not None:
        log_prob = sampler.get_log_prob(flat=True)
        return chain[:, 0], log_prob
    else:
        return np.array([]), np.array([])


def test_single_dispersion(disp=1, nwalkers=8, nsteps=1000, npoints=10, seed=42):
    """
    Test both sampling methods on a single dispersion type with reduced parameters.
    """

    run_mcmc_sampling(disp, nwalkers, nsteps, seed)

if __name__ == "__main__":
    for i in [3]:
        test_single_dispersion(disp=i, nwalkers=16, nsteps=1500)
