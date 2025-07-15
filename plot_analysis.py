import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.stats import chi2


"""
Check if getdist is installed. If not, use matplotlib for the corner plots. 
"""
try:
    import getdist as gd
    from getdist import plots

    GETDIST_AVAILABLE = True
except ImportError:
    GETDIST_AVAILABLE = False



def load_dispersion_data(filename):
    """
    Load dispersion data from CSV file with proper data cleaning and filtering.
    
    This function reads dispersion data from a space-separated CSV file, converts
    all columns to numeric format, removes NaN values, and optionally filters
    data based on Yp values if YP_MIN and YP_MAX are defined.
    
    Parameters
    ----------
    filename : str
        Path to the CSV file containing dispersion data
        
    Returns
    -------
    data : pandas.DataFrame or None
        DataFrame with columns: Neff, Omegabh2, 1/Omegabh2, Yp(CMB), Yp, DoH, He3, Li7, param, log_likelihood.
        Returns None if file cannot be loaded or processed.
        
    Notes
    -----
    The function expects a space-separated file with specific column names.
    All columns are converted to numeric format to handle scientific notation.
    NaN values are automatically removed from the dataset.
    """
    try:
        data = pd.read_csv(
            filename,
            sep=" ",
            names=[
                "Neff",
                "Omegabh2",
                "1/Omegabh2",
                "Yp(CMB)",
                "Yp",
                "DoH",
                "He3",
                "Li7",
                "param",
                "log_likelihood",
            ],
        )

        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        

        data = data.dropna()
        data = data[int(len(data)*0.1):]

        print(f"Loaded {len(data)} samples from {filename}")

        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def calculate_model_selection_criteria(samples, log_likelihoods, n_params=1):
    """
    Calculate model selection criteria including AIC, BIC, and chi-squared values.
    
    This function computes various statistical measures for model comparison and
    selection based on the maximum log-likelihood and sample size.
    
    Parameters
    ----------
    samples : array-like
        Parameter samples from the posterior distribution
    log_likelihoods : array-like
        Log-likelihood values corresponding to each sample
    n_params : int, default=1
        Number of parameters in the model
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'AIC': Akaike Information Criterion
        - 'BIC': Bayesian Information Criterion  
        - 'chi2': Chi-squared statistic
        - 'max_log_likelihood': Maximum log-likelihood value
        - 'n_samples': Number of samples
        - 'n_params': Number of parameters
        
    Notes
    -----
    AIC = 2*n_params - 2*max_log_likelihood
    BIC = log(n_samples)*n_params - 2*max_log_likelihood
    chi2 = -2*max_log_likelihood
    """
    n_samples = len(samples)

    max_log_likelihood = np.max(log_likelihoods)

    chi2_value = -2 * max_log_likelihood

    aic = 2 * n_params - 2 * max_log_likelihood

    bic = np.log(n_samples) * n_params - 2 * max_log_likelihood

    return {
        "AIC": aic,
        "BIC": bic,
        "chi2": chi2_value,
        "max_log_likelihood": max_log_likelihood,
        "n_samples": n_samples,
        "n_params": n_params,
    }


def plot_parameter_distribution(samples, param_label, disp_label, filename):
    """
    Create parameter distribution plots including trace plot and histogram with Gaussian fit.
    
    This function generates a comprehensive visualization of parameter samples including
    a trace plot showing the sampling history and a histogram with fitted Gaussian
    distribution for statistical analysis.
    
    Parameters
    ----------
    samples : array-like
        Parameter samples to be plotted
    param_label : str
        Label for the parameter (e.g., r'$\lambda$' for LaTeX formatting)
    disp_label : str
        Label for the dispersion type (e.g., "Dispersion 1")
    filename : str
        Base filename for saving the plot (without extension)
        
    Notes
    -----
    The function creates a two-panel figure:
    - Left panel: Trace plot showing parameter values vs sample index
    - Right panel: Histogram with fitted Gaussian distribution
    The plot is saved as PNG with 150 DPI resolution.
    """
    fig, (ax2) = plt.subplots(1, 1, figsize=(7,6))


    mu, std = np.mean(samples), np.std(samples)
    ax2.hist(
        samples,
        bins=24,
        density=True,
        alpha=0.8,
        color='steelblue',
        edgecolor='navy',
        linewidth=1.2,
        label=f"Posterior samples\nμ={mu:.2e}\nσ={std:.1e}",
    )

    x = np.linspace(np.min(samples), np.max(samples), 500)
    gaussian_pdf = norm.pdf(x, mu, std)
    ax2.plot(x, gaussian_pdf, "r--", lw=2, label="Gaussian fit")

    ax2.set_title(f"Parameter Distribution ({disp_label})")
    ax2.set_xlabel(f"Parameter Value ({param_label})")
    ax2.set_ylabel("Probability Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{filename}_parameter.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Parameter statistics: μ = {mu:.4e}, σ = {std:.1e}")


def plot_abundance_corner(data, disp_label, filename):
    """
    Create a comprehensive corner plot showing correlations between abundance parameters using getdist.
    
    This function generates a corner plot (triangle plot) showing the
    relationships between all abundance parameters (Yp, D/H, ³He/H, ⁷Li/H, Neff, Ωνh²)
    using the getdist package for publication-quality visualization.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing abundance data with columns: Yp, DoH, He3, Li7, Neff, Omegabh2
    disp_label : str
        Label for the dispersion type
    filename : str
        Base filename for saving the plot (without extension)
        
    Notes
    -----
    The corner plot shows:
    - Diagonal: 1D marginalized posteriors
    - Off-diagonal: 2D marginalized contours and correlation coefficients
    - All plots use consistent formatting and labels
    """
    
    from getdist import plots, MCSamples

    abundance_data = data[["Yp(CMB)", "DoH", "He3", "Li7"]].values

    
    

    labels = [
        r"$Y_p$",
        r"$D/H$",
        r"$^3He/H$",
        r"$^7Li/H$",
    ]

    labels = [lbl[1:-1] for lbl in labels]
   

    samples = MCSamples(samples=abundance_data, names=["Yp(CMB)", "DoH", "He3", "Li7"], labels=labels)

    
    g = plots.get_subplot_plotter()
    g.triangle_plot([samples], filled=True, contour_colors=[
        ["red", "green", "blue"][int(disp_label[-1]) - 1],
    ],diag1d_kwargs={'smooth': 0.1})

    
    g.fig.suptitle(f"Abundances ({disp_label})", fontsize=14)

    
    g.export(f"{filename}_corner.eps")

    
    corr_matrix = np.corrcoef(abundance_data.T)
    print(f"\nCorrelation matrix for {disp_label}:")
    print("           Yp      D/H    ³He/H   ⁷Li/H")
    for i, label in enumerate(labels):
        print(
            f"{label:8s} {corr_matrix[i, 0]:6.3f} {corr_matrix[i, 1]:6.3f} {corr_matrix[i, 2]:6.3f} {corr_matrix[i, 3]:6.3f}"
        )


def plot_abundance_distributions(data, disp_label, filename):
    """
    Create individual distribution plots for each abundance parameter.
    
    This function generates a 2x2 subplot layout showing the distribution of each
    abundance parameter (Yp, D/H, ³He/H, ⁷Li/H) with histograms and fitted Gaussian
    distributions for statistical analysis.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing abundance data with columns: Yp, DoH, He3, Li7
    disp_label : str
        Label for the dispersion type
    filename : str
        Base filename for saving the plot (without extension)
        
    Notes
    -----
    Each subplot shows:
    - Histogram of the abundance parameter
    - Fitted Gaussian distribution (red dashed line)
    - Mean and standard deviation in the legend
    - Proper LaTeX formatting for parameter labels
    """
    abundances = ["Yp", "DoH", "He3", "Li7"]
    labels = [r"$Y_p$", r"$D/H$", r"$^3He/H$", r"$^7Li/H$"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (abund, label) in enumerate(zip(abundances, labels)):
        values = data[abund].values

        axes[i].hist(
            values,
            bins=30,
            density=True,
            alpha=0.8,
            color='cornflowerblue',
            edgecolor='royalblue',
            linewidth=1.2,
            label=f"μ={np.mean(values):.4f}\nσ={np.std(values):.4f}",
        )

        mu, std = np.mean(values), np.std(values)
        x = np.linspace(np.min(values), np.max(values), 500)
        gaussian_pdf = norm.pdf(x, mu, std)
        axes[i].plot(x, gaussian_pdf, "r--", lw=2, label="Gaussian fit")

        axes[i].set_title(f"{label} Distribution ({disp_label})")
        axes[i].set_xlabel(f"{label} Value")
        axes[i].set_ylabel("Probability Density")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{filename}_abundances.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """
    Main function to run the complete analysis pipeline for all dispersion types.
    
    This function orchestrates the entire analysis workflow:
    1. Loads dispersion data for each dispersion type (1, 2, 3)
    2. Extracts parameter samples and log-likelihoods
    3. Calculates model selection criteria (AIC, BIC, χ²)
    4. Generates comprehensive visualization plots
    5. Saves all results with appropriate filenames
    
    The function processes three dispersion types:
    - Dispersion 1: λ parameter
    - Dispersion 2: β parameter  
    - Dispersion 3: α parameter
    
    For each dispersion type, it creates:
    - Parameter distribution plots (trace + histogram)
    - Model selection statistics
    - Abundance correlation plots (if abundance data available)
    
    Notes
    -----
    All plots are saved with descriptive filenames including the dispersion type.
    The function handles missing data gracefully and provides informative output.
    """
    disp_labels = {
        1: ("Dispersion 1", r"$\lambda$"),
        2: ("Dispersion 2", r"$\beta$"),
        3: ("Dispersion 3", r"$\alpha$"),
    }

    for disp in [1,2,3]:
        print(f"\n{'=' * 60}")
        print(f"Analyzing {disp_labels[disp][0]}")
        print(f"{'=' * 60}")

        data = load_dispersion_data(f"dispersion_{disp}.csv")
        if data is None:
            continue
            
        samples, log_likelihoods = data["param"], data["log_likelihood"]

        if samples is not None and log_likelihoods is not None:

            criteria = calculate_model_selection_criteria(
                samples, log_likelihoods, n_params=1
            )

            print(f"\nModel Selection Criteria:")
            print(f"AIC: {criteria['AIC']:.8f}")
            print(f"BIC: {criteria['BIC']:.8f}")
            print(f"χ²: {criteria['chi2']:.8f}")
            print(f"Max Log-Likelihood: {criteria['max_log_likelihood']:.8f}")

            print(f"Yp(CMB) mean: {data['Yp(CMB)'].mean()}, std: { data['Yp(CMB)'].std()}")
            print(f"D/H mean: {data['DoH'].mean()}, D/H std: {data['DoH'].std()}")
            print(f"3He/H mean: {data['He3'].mean()}, 3He/H std: {data['He3'].std()}")
            print(f"7Li/H mean: {data['Li7'].mean()}, 7Li/H std: {data['Li7'].std()}")
            print(f"N_eff mean: {data['Neff'].mean()}, N_eff std: {data['Neff'].std()}")
            print(f"Omega mean: {data['Omegabh2'].mean()}, Omega std: {data['Omegabh2'].std()}")


            base_filename = f"analysis_disp{disp}"

            plot_parameter_distribution(
                samples, disp_labels[disp][1], disp_labels[disp][0], base_filename
            )

            if all(col in data.columns for col in ["Yp(CMB)", "DoH", "He3", "Li7", "Neff", "Omegabh2"]):
                plot_abundance_corner(data, disp_labels[disp][0], base_filename)
                plot_abundance_distributions(data, disp_labels[disp][0], base_filename)

                plt.scatter(data["Yp(CMB)"], data["param"])
                plt.savefig(f"analysis_disp{disp}.png", dpi=150, bbox_inches="tight")
                plt.close()
            

if __name__ == "__main__":
    main()
