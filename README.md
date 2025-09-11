
# PRyNCe: PRyMordial Non-Commutativity Evaluation

A flexible framework for studying how non-standard cosmological models impact Big Bang Nucleosynthesis (BBN) through MCMC sampling.

## Overview

**PRyNCe** builds on the PRyM nucleosynthesis engine, adding new physics via modified energy densities from three non-standard dispersion relations. It uses Bayesian inference to estimate how these models affect light element abundances.

## Folder Structure


```
PRyNCe/
├── model.py              # Core BBN model with dispersion relations
├── test\_sampling.py      # MCMC sampling logic
├── plot\_analysis.py      # Postprocessing and visualization
├── dispersion\_*.csv      # MCMC output files
├── analysis\_*.png        # Plots
├── PRyM/                 # BBN engine
├── PRyMrates/            # Reaction rates
└── README.md
```

### `test_sampling.py` – MCMC Sampler

- Uses `emcee` for affine-invariant ensemble sampling
- Supports parallelization and reproducibility
- Includes diagnostics: autocorrelation time, thinning, burn-in

---

### `plot_analysis.py` – Postprocessing

- Loads CSV results
- Calculates model selection stats (AIC, BIC, χ²)
- Plots:
  - Histograms with Gaussian fits
  - Corner plots (with `getdist`)
  - Abundance trends per element

---

## Installation

### Requirements

- Python ≥ 3.8
- PRyMordial: specifically commit bf24c3d 

### Install Dependencies

```bash
pip install numpy scipy matplotlib pandas emcee joblib
pip install getdist  # Optional: for nicer plots
````

### Setup

```bash
git clone <repository-url>
cd PRyNCe

python -m venv prynce_env
source prynce_env/bin/activate  # Windows: prynce_env\Scripts\activate
```

---

## Output Files

### `dispersion_*.csv` Columns:

* `Neff`, `Omegabh2`, `Yp(CMB)`, `Yp`, `DoH`, `He3`, `Li7`
* `param`: Model-specific parameter
* `log_likelihood`: Log-likelihood of each sample

### Plot Files:

* Parameter distributions: `analysis_disp*_parameter.png`
* Corner plots: `analysis_disp*_corner.png`
* Abundance trends: `analysis_disp*_abundances.png`
