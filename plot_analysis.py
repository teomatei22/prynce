import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.stats import chi2

from thin import get_autocorr_time, thin

# Try to import getdist, but provide fallback if not available
try:
    import getdist as gd
    from getdist import plots
    GETDIST_AVAILABLE = True
except ImportError:
    GETDIST_AVAILABLE = False
    print("Warning: getdist not available. Install with: pip install getdist")
    print("Using matplotlib fallback for plotting.")

YP_MIN = 0.24
YP_MAX = 0.25

def load_dispersion_data(filename):
    """
    Load dispersion data from CSV file.
    
    Parameters
    ----------
    filename : str
        Path to the CSV file
        
    Returns
    -------
    data : pandas.DataFrame
        DataFrame with columns: Neff, Omegabh2, 1/Omegabh2, Yp(CMB), Yp, DoH, He3, Li7, param, log_likelihood
    """
    try:
        data = pd.read_csv(filename, sep=' ', names=['Neff', 'Omegabh2', '1/Omegabh2', 'Yp(CMB)', 'Yp', 'DoH', 'He3', 'Li7', 'param', 'log_likelihood'])
        
        # Convert all columns to float to handle scientific notation properly
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        print(f"Loaded {len(data)} samples from {filename}")

        # Filter rows where Yp is between YP_MIN and YP_MAX
        # if 'Yp' in data.columns and 'YP_MIN' in globals() and 'YP_MAX' in globals():
        #     data = data[(data['Yp'] >= YP_MIN) & (data['Yp'] <= YP_MAX)]


        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None



def calculate_model_selection_criteria(samples, log_likelihoods, n_params=1):
    """
    Calculate AIC, BIC, and chi2 values from thinned samples.
    
    Parameters
    ----------
    samples : array
        Parameter samples
    log_likelihoods : array
        Log-likelihood values
    n_params : int
        Number of model parameters
        
    Returns
    -------
    dict : Dictionary with AIC, BIC, chi2 values
    """


    n_samples = len(samples)
    
    # Get the maximum log-likelihood
    max_log_likelihood = np.max(log_likelihoods)
    
    # Calculate chi2 
    chi2_value = -2 * max_log_likelihood
    
    # Calculate AIC
    aic = 2 * n_params - 2 * max_log_likelihood
    
    # Calculate BIC
    bic = np.log(n_samples) * n_params - 2 * max_log_likelihood
    
    return {
        'AIC': aic,
        'BIC': bic,
        'chi2': chi2_value,
        'max_log_likelihood': max_log_likelihood,
        'n_samples': n_samples,
        'n_params': n_params
    }

def plot_parameter_distribution(samples, param_label, disp_label, filename):
    """
    Plot the parameter distribution using matplotlib.
    
    Parameters
    ----------
    samples : array
        Parameter samples
    param_label : str
        Label for the parameter (e.g., r'$\lambda$')
    disp_label : str
        Label for the dispersion type
    filename : str
        Base filename for saving plots
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trace plot
    ax1.plot(samples, alpha=0.6, lw=0.5)
    ax1.set_title(f"Parameter Trace Plot ({disp_label})")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel(f"Parameter Value ({param_label})")
    ax1.grid(True, alpha=0.3)
    
    # Histogram with Gaussian fit
    mu, std = np.mean(samples), np.std(samples)
    ax2.hist(samples, bins=50, density=True, alpha=0.7, 
             label=f"Posterior samples\nμ={mu:.2e}\nσ={std:.1e}")
    
    # Gaussian fit
    x = np.linspace(np.min(samples), np.max(samples), 500)
    gaussian_pdf = norm.pdf(x, mu, std)
    ax2.plot(x, gaussian_pdf, 'r--', lw=2, label='Gaussian fit')
    
    ax2.set_title(f"Parameter Distribution ({disp_label})")
    ax2.set_xlabel(f"Parameter Value ({param_label})")
    ax2.set_ylabel("Probability Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename}_parameter.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Parameter statistics: μ = {mu:.4e}, σ = {std:.1e}")

def plot_abundance_corner(data, disp_label, filename):
    """
    Create corner plot for abundance correlations using matplotlib.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with abundance data
    disp_label : str
        Label for the dispersion type
    filename : str
        Base filename for saving plots
    """
    # Select abundance columns
    abundance_data = data[['Yp', 'DoH', 'He3', 'Li7', 'Neff', 'Omegabh2']].values
    
    # Labels for the corner plot
    labels = [r'$Y_p$', r'$D/H$', r'$^3He/H$', r'$^7Li/H$', r'$N_eff$', r'$\Omega_\nu h^2$']
    
    # Create correlation matrix
    corr_matrix = np.corrcoef(abundance_data.T)
    
    # Create corner plot using matplotlib
    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    
    for i in range(6):
        for j in range(6):
            if i == j:
                # Diagonal: histogram
                axes[i, j].hist(abundance_data[:, i], bins=30, density=True, alpha=0.7)
                axes[i, j].set_title(labels[i])
            else:
                # Off-diagonal: scatter plot
                axes[i, j].scatter(abundance_data[:, j], abundance_data[:, i], alpha=0.5, s=1)
                axes[i, j].set_xlabel(labels[j])
                axes[i, j].set_ylabel(labels[i])
                # Add correlation coefficient
                axes[i, j].text(0.05, 0.95, f'ρ={corr_matrix[i, j]:.2f}', 
                               transform=axes[i, j].transAxes, verticalalignment='top')
    
    fig.suptitle(f"Abundance Correlations ({disp_label})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{filename}_corner.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print correlation matrix
    print(f"\nCorrelation matrix for {disp_label}:")
    print("           Yp      D/H    ³He/H   ⁷Li/H")
    for i, label in enumerate(labels):
        print(f"{label:8s} {corr_matrix[i,0]:6.3f} {corr_matrix[i,1]:6.3f} {corr_matrix[i,2]:6.3f} {corr_matrix[i,3]:6.3f}")

def plot_abundance_distributions(data, disp_label, filename):
    """
    Plot individual abundance distributions using matplotlib.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with abundance data
    disp_label : str
        Label for the dispersion type
    filename : str
        Base filename for saving plots
    """
    abundances = ['Yp', 'DoH', 'He3', 'Li7']
    labels = [r'$Y_p$', r'$D/H$', r'$^3He/H$', r'$^7Li/H$']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (abund, label) in enumerate(zip(abundances, labels)):
        values = data[abund].values
        
        # Histogram
        axes[i].hist(values, bins=30, density=True, alpha=0.7, 
                    label=f"μ={np.mean(values):.4f}\nσ={np.std(values):.4f}")
        
        # Gaussian fit
        mu, std = np.mean(values), np.std(values)
        x = np.linspace(np.min(values), np.max(values), 500)
        gaussian_pdf = norm.pdf(x, mu, std)
        axes[i].plot(x, gaussian_pdf, 'r--', lw=2, label='Gaussian fit')
        
        axes[i].set_title(f"{label} Distribution ({disp_label})")
        axes[i].set_xlabel(f"{label} Value")
        axes[i].set_ylabel("Probability Density")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename}_abundances.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the analysis."""
    
    # Dispersion type labels
    disp_labels = {
        1: ("Dispersion 1", r"$\lambda$"),
        2: ("Dispersion 2", r"$\beta$"),
        3: ("Dispersion 3", r"$\alpha$")
    }
    
    # Process each dispersion type
    for disp in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Analyzing {disp_labels[disp][0]}")
        print(f"{'='*60}")
        
        # Try to load thinned data first
        data = load_dispersion_data(f"dispersion_{disp}.csv")
        autocorr = get_autocorr_time(data)
        samples, log_likelihoods = data["param"], data["log_likelihood"] #thin(data["param"], data["log_likelihood"], n_thin=40*autocorr,kn=4, burnin=10)
        
        if samples is not None and log_likelihoods is not None:
            # Use thinned data
            print("Using thinned samples for analysis")
            
            # Calculate model selection criteria

            criteria = calculate_model_selection_criteria(samples, log_likelihoods, n_params=1)
            
            print(f"\nModel Selection Criteria:")
            print(f"AIC: {criteria['AIC']:.8f}")
            print(f"BIC: {criteria['BIC']:.8f}")
            print(f"χ²: {criteria['chi2']:.8f}")
            print(f"Max Log-Likelihood: {criteria['max_log_likelihood']:.8f}")
            print(f"Number of samples: {criteria['n_samples']}")
            
            # Create plots
            base_filename = f"analysis_disp{disp}_thinned"
            
            # Parameter distribution
            plot_parameter_distribution(samples, disp_labels[disp][1], disp_labels[disp][0], base_filename)
            

if __name__ == "__main__":
    main() 