import argparse
import os
import re
import numpy as np
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plots will be skipped")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, normal fitting will be skipped")


@dataclass
class DataConfig:
    data_dir: str
    min_samples_per_chain: int
    max_chains: int
    random_seed: int


def compute_split_rhat(chains: np.ndarray) -> np.ndarray:
    """
    Compute split-chain Gelman-Rubin R-hat per parameter.

    Parameters
    ----------
    chains : np.ndarray
        Array of shape (nchains, nsteps, ndim), AFTER burn-in and thinning.

    Returns
    -------
    np.ndarray
        R-hat values of shape (ndim,).
    """
    if chains.ndim != 3:
        raise ValueError("chains must be (nchains, nsteps, ndim)")

    nchains, nsteps, ndim = chains.shape
    if nsteps < 4:
        raise ValueError("Too few steps after burn-in to compute R-hat (need >= 4)")

    half = nsteps // 2
    if half < 2:
        raise ValueError("Too few steps per split-chain (need >= 2)")

    split_chains = np.concatenate([chains[:, :half, :], chains[:, -half:, :]], axis=0)
    m = split_chains.shape[0]
    n = split_chains.shape[1]

    chain_means = split_chains.mean(axis=1)
    chain_vars = split_chains.var(axis=1, ddof=1)
    grand_means = chain_means.mean(axis=0)

    B = n * ((chain_means - grand_means) ** 2).sum(axis=0) / (m - 1)
    W = chain_vars.mean(axis=0)

    var_hat = ((n - 1) / n) * W + (B / n)
    rhat = np.sqrt(var_hat / W)
    return rhat


def get_display_names(model_name: str) -> List[str]:
    """Get the display names for parameters of a given model."""
    # All dispersion models have 1 parameter
    return ["disp_param"]


def get_prior_bounds(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get the prior bounds used in the MCMC runs from model.py."""
    # From model.py PRIORS dictionary
    prior_bounds_map = {
        "dispersion_1": (1e-10, 9e-3),      # DISP_LAM
        "dispersion_2": (1e-9, 1e-3),       # DISP_BET  
        "dispersion_3": (1e-10, 2e-3),      # DISP_ALP
    }

    if model_name not in prior_bounds_map:
        raise ValueError(f"Unknown model: {model_name}")

    lower, upper = prior_bounds_map[model_name]
    return np.array([lower]), np.array([upper])


def load_mcmc_chains(
    model_name: str, data_dir: str, config: DataConfig
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load MCMC chains from CSV files.

    Returns:
        chains: array of shape (1, nsteps, 1) containing the param column as one continuous sequence
        prior_bounds: dict with 'lower' and 'upper' bounds
    """
    csv_file = os.path.join(data_dir, f"{model_name}.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # Load the CSV data manually - only the param column
    param_data = []
    with open(csv_file, 'r') as f:
        # Skip header line
        next(f)
        for line in f:
            try:
                # Split by whitespace and get the 9th column (param)
                parts = line.strip().split()
                if len(parts) >= 9:
                    param_val = float(parts[8])  # 9th column (0-indexed)
                    if not np.isnan(param_val):
                        param_data.append(param_val)
            except (ValueError, IndexError):
                continue
    
    if not param_data:
        raise ValueError(f"No valid parameter data found in {csv_file}")
    
    param_data = np.array(param_data)
    
    # Remove burn-in (first 10% of samples)
    burn_in = int(len(param_data) * 0.1)
    param_data = param_data[burn_in:]
    
    # Create a single "chain" with shape (1, nsteps, 1)
    # This represents the param column as one continuous sequence
    chains = param_data.reshape(1, -1, 1)  # Shape: (1, nsteps, 1)

    prior_bounds = get_prior_bounds(model_name)
    prior_bounds_dict = {"lower": prior_bounds[0], "upper": prior_bounds[1]}

    return chains, prior_bounds_dict


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_representative_trace(
    chains: np.ndarray,
    outdir: str,
    model_name: str,
    param_names: List[str],
    max_chains_to_plot: int = 8,
) -> Optional[str]:
    """
    Plot representative trace plots for all parameters.
    Shows the param column as one continuous sequence.
    Returns the file path of the generated figure, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping trace plot (matplotlib not available)")
        return None

    nchains, nsteps, ndim = chains.shape

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Extract the single chain data
    param_trace = chains[0, :, 0]  # Shape: (nsteps,)
    
    # Create step indices for the trace
    step_indices = np.arange(len(param_trace))
    
    # Define model-specific titles and parameter names
    model_titles = {
        "dispersion_1": "Model I: $\\omega(k) = kc(1 + \\lambda \\hbar \\omega)$",
        "dispersion_2": "Model II: $\\omega(k) = kc \\sqrt{1 - 2\\beta_0 \\hbar^2 \\omega^2}$", 
        "dispersion_3": "Model III: $\\omega(k) = kc \\sqrt{1 - 2\\alpha_0 \\hbar \\omega}$"
    }
    
    param_labels = {
        "dispersion_1": "$\\lambda \\, [\\mathrm{MeV}^{-1}]$",
        "dispersion_2": "$\\beta_0 \\, [\\mathrm{MeV}^{-2}]$",
        "dispersion_3": "$\\alpha_0 \\, [\\mathrm{MeV}^{-1}]$"
    }
    
    title = model_titles.get(model_name, f"{model_name.upper()} — Dispersion Parameter")
    param_label = param_labels.get(model_name, "Dispersion Parameter")
    
    # Plot the parameter trace
    ax.plot(
        step_indices,
        param_trace,
        alpha=0.8,
        lw=1.0,
        color='#1f77b4',
        label=f"Parameter trace ({len(param_trace)} samples)"
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Step", fontsize=18)
    ax.set_ylabel(param_label, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=18)

    fig.tight_layout()

    outpath = os.path.join(outdir, f"{model_name}_trace.eps")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def create_rhat_table(
    all_models: List[str], data_dir: str, config: DataConfig, outdir: str, 
    create_artificial_chains: bool = False
) -> Tuple[str, str]:
    """
    Create a comprehensive table showing both parameter values and R-hat convergence diagnostics.

    Args:
        all_models: List of model names to analyze
        data_dir: Directory containing CSV files
        outdir: Output directory for the table
        create_artificial_chains: If True, split single-chain data into multiple chains for R-hat

    Returns:
        Tuple of (png_table_path, csv_table_path)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping comprehensive table (matplotlib not available)")
        return "", ""

    table_data = []

    for model_name in all_models:
        try:
            # First, get parameter statistics from CSV
            csv_file = os.path.join(data_dir, f"{model_name}.csv")
            if not os.path.exists(csv_file):
                row = [model_name.upper(), "FILE_NOT_FOUND", "N/A", "N/A"]
                table_data.append(row)
                continue
            
            # Extract parameter values from the CSV
            param_values = []
            with open(csv_file, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    try:
                        parts = line.strip().split()
                        if len(parts) >= 9:
                            param_val = float(parts[8])  # 9th column (param)
                            if not np.isnan(param_val):
                                param_values.append(param_val)
                    except (ValueError, IndexError):
                        continue
            
            if not param_values:
                row = [model_name.upper(), "NO_DATA", "N/A", "N/A"]
                table_data.append(row)
                continue
            
            # Remove burn-in (first 10%)
            burn_in = int(len(param_values) * 0.1)
            param_values = param_values[burn_in:]
            
            # Calculate parameter statistics
            mean_param = np.mean(param_values)
            std_param = np.std(param_values)
            param_display = f"{mean_param:.2e} ± {std_param:.2e}"
            
            # Now try to compute R-hat
            try:
                if create_artificial_chains:
                    # Create artificial chains by splitting the data
                    n_samples = len(param_values)
                    n_chains = min(4, n_samples // 100)  # Create up to 4 chains, minimum 100 samples each
                    
                    if n_chains >= 2:
                        # Split data into chains
                        samples_per_chain = n_samples // n_chains
                        chains = []
                        for i in range(n_chains):
                            start_idx = i * samples_per_chain
                            end_idx = start_idx + samples_per_chain
                            if end_idx <= len(param_values):
                                chain = param_values[start_idx:end_idx]
                                chains.append(chain)
                        
                        if len(chains) >= 2:
                            # Ensure all chains have the same length
                            min_length = min(len(chain) for chain in chains)
                            chains = [chain[:min_length] for chain in chains]
                            
                            # Convert to numpy array with proper shape (n_chains, n_samples, n_params)
                            chains_array = np.array(chains).reshape(len(chains), min_length, 1)
                            
                            # Compute R-hat using the split chains
                            try:
                                # Apply burn-in to each chain (first 25%)
                                burn_in = int(min_length * 0.25)
                                chains_post = chains_array[:, burn_in:, :]
                                
                                # Compute R-hat manually since we have the chains
                                n_chains, n_samples, n_params = chains_post.shape
                                
                                # Calculate within-chain variance
                                within_var = np.mean(np.var(chains_post, axis=1, ddof=1), axis=0)
                                
                                # Calculate between-chain variance
                                chain_means = np.mean(chains_post, axis=1)
                                overall_mean = np.mean(chain_means, axis=0)
                                between_var = np.var(chain_means, axis=0, ddof=1)
                                
                                # Calculate R-hat
                                rhat = np.sqrt((between_var + within_var) / within_var)
                                
                                rhat_display = f"{rhat[0]:.3f}*"
                                convergence_status = "EXCELLENT*" if rhat[0] < 1.05 else "WARNING*" if rhat[0] < 1.1 else "POOR*"
                            except Exception as e:
                                rhat_display = f"R-hat Error: {str(e)[:20]}"
                                convergence_status = "ERROR"
                        else:
                            rhat_display = "Single Chain"
                            convergence_status = "N/A"
                    else:
                        rhat_display = "Single Chain"
                        convergence_status = "N/A"
                else:
                    # Use original approach
                    chains, _ = load_mcmc_chains(model_name, data_dir, config)
                    
                    if chains.shape[0] >= 2:
                        # Multiple chains available - compute R-hat
                        chains_post, _ = process_loaded_chains(chains, 0.25)
                        rhat = compute_split_rhat(chains_post)
                        rhat_display = f"{rhat[0]:.3f}"
                        convergence_status = "EXCELLENT" if rhat[0] < 1.05 else "WARNING" if rhat[0] < 1.1 else "POOR"
                    else:
                        # Single chain - can't compute R-hat
                        rhat_display = "Single Chain"
                        convergence_status = "N/A"
                        
            except Exception as e:
                # If R-hat computation fails, still show parameter values
                rhat_display = f"R-hat Error: {str(e)[:20]}"
                convergence_status = "ERROR"
            
            row = [model_name.upper(), param_display, rhat_display, convergence_status]
            table_data.append(row)

        except Exception as e:
            print(f"Warning: Could not process model {model_name}: {e}")
            row = [model_name.upper(), "ERROR", "ERROR", "ERROR"]
            table_data.append(row)

    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.5 + 2))
    ax.axis("tight")
    ax.axis("off")

    headers = ["Model", "Dispersion Parameter [MeV⁻ⁿ] (mean ± std)", "R-hat Value", "Convergence Status"]

    table = ax.table(
        cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.5)

    # Color code based on R-hat values or status
    for i, row in enumerate(table_data):
        if len(row) > 2:
            rhat_cell = row[2]
            if rhat_cell == "Single Chain":
                table[(i + 1, 2)].set_facecolor("#e9ecef")  # Gray for single chain
                table[(i + 1, 3)].set_facecolor("#e9ecef")  # Gray for status
            elif rhat_cell == "R-hat Error" or rhat_cell == "ERROR":
                table[(i + 1, 2)].set_facecolor("#f8d7da")  # Red for error
                table[(i + 1, 3)].set_facecolor("#f8d7da")  # Red for error
            elif rhat_cell != "N/A":
                try:
                    rhat_val = float(rhat_cell.replace("*", ""))  # Remove asterisk for comparison
                    if rhat_val < 1.05:
                        table[(i + 1, 2)].set_facecolor("#d4edda")  # Green for excellent
                        table[(i + 1, 3)].set_facecolor("#d4edda")  # Green for excellent
                    elif rhat_val < 1.1:
                        table[(i + 1, 2)].set_facecolor("#fff3cd")  # Yellow for warning
                        table[(i + 1, 3)].set_facecolor("#fff3cd")  # Yellow for warning
                    else:
                        table[(i + 1, 2)].set_facecolor("#f8d7da")  # Red for poor
                        table[(i + 1, 3)].set_facecolor("#f8d7da")  # Red for poor
                except ValueError:
                    pass

    # Style headers
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#e9ecef")
        table[(0, j)].set_text_props(weight="bold")

    title = "Dispersion Models: Parameters & Convergence Diagnostics"
    if create_artificial_chains:
        title += " (* = Artificial Chains)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#d4edda", label="Excellent (R-hat < 1.05)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#fff3cd", label="Warning (1.05 ≤ R-hat < 1.1)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#f8d7da", label="Poor (R-hat ≥ 1.1)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#e9ecef", label="Single Chain (N/A)"),
    ]
    if create_artificial_chains:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor="#d4edda", label="* = Artificial Chains"))
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 0.98))

    eps_outpath = os.path.join(outdir, "all_models_R.eps")
    fig.savefig(eps_outpath, format='eps', bbox_inches="tight")
    plt.close(fig)

    # Save CSV version
    csv_outpath = os.path.join(outdir, "all_models_R.csv")
    with open(csv_outpath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Model", "Dispersion_Parameter_Mean", "Dispersion_Parameter_Std", 
            "Min_Value", "Max_Value", "Sample_Count", "Rhat_Value", "Convergence_Status"
        ])

        for row in table_data:
            model_name = row[0]
            param_display = row[1] if len(row) > 1 else "N/A"
            rhat_display = row[2] if len(row) > 2 else "N/A"
            convergence_status = row[3] if len(row) > 3 else "N/A"
            
            # Extract the actual parameter values for detailed CSV
            csv_file = os.path.join(data_dir, f"{model_name.lower()}.csv")
            if os.path.exists(csv_file):
                try:
                    param_values = []
                    with open(csv_file, 'r') as f:
                        next(f)  # Skip header
                        for line in f:
                            try:
                                parts = line.strip().split()
                                if len(parts) >= 9:
                                    param_val = float(parts[8])
                                    if not np.isnan(param_val):
                                        param_values.append(param_val)
                            except (ValueError, IndexError):
                                continue
                    
                    if param_values:
                        burn_in = int(len(param_values) * 0.1)
                        param_values = param_values[burn_in:]
                        
                        mean_val = np.mean(param_values)
                        std_val = np.std(param_values)
                        min_val = np.min(param_values)
                        max_val = np.max(param_values)
                        count = len(param_values)
                        
                        writer.writerow([
                            model_name, 
                            f"{mean_val:.6e}", 
                            f"{std_val:.6e}", 
                            f"{min_val:.6e}", 
                            f"{max_val:.6e}", 
                            count,
                            rhat_display,
                            convergence_status
                        ])
                    else:
                        writer.writerow([model_name, "N/A", "N/A", "N/A", "N/A", 0, rhat_display, convergence_status])
                except Exception as e:
                    writer.writerow([model_name, "ERROR", "ERROR", "ERROR", "ERROR", 0, rhat_display, convergence_status])
            else:
                writer.writerow([model_name, "FILE_NOT_FOUND", "FILE_NOT_FOUND", "FILE_NOT_FOUND", "FILE_NOT_FOUND", 0, rhat_display, convergence_status])

    return eps_outpath, csv_outpath


def plot_prior_posterior_overlays(
    posterior_samples: np.ndarray,
    prior_bounds: Dict[str, np.ndarray],
    outdir: str,
    model_name: str,
    param_names: List[str],
) -> List[str]:
    """Plot prior vs posterior overlays."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping prior/posterior overlay plots (matplotlib not available)")
        return []

    if not SCIPY_AVAILABLE:
        print("Skipping prior/posterior overlay plots (scipy not available)")
        return []

    paths: List[str] = []
    ndim = posterior_samples.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))

    lower_bound = prior_bounds["lower"][0]
    upper_bound = prior_bounds["upper"][0]

    x_range = upper_bound - lower_bound
    x_min = lower_bound - 0.1 * x_range
    x_max = upper_bound + 0.1 * x_range
    x = np.linspace(x_min, x_max, 1000)

    # Define model-specific parameter names
    param_labels = {
        "dispersion_1": "$\\lambda \\, [\\mathrm{MeV}^{-1}]$",
        "dispersion_2": "$\\beta_0 \\, [\\mathrm{MeV}^{-2}]$", 
        "dispersion_3": "$\\alpha_0 \\, [\\mathrm{MeV}^{-1}]$"
    }
    
    param_label = param_labels.get(model_name, "Dispersion Parameter")

    # Plot prior (uniform)
    prior_height = 1.0 / (upper_bound - lower_bound)
    ax.fill_between(
        [lower_bound, upper_bound],
        [0, 0],
        [prior_height, prior_height],
        alpha=0.3,
        color="#1f77b4",
        label="Prior (Uniform)",
        step="mid",
    )
    ax.plot(
        [lower_bound, lower_bound, upper_bound, upper_bound],
        [0, prior_height, prior_height, 0],
        color="#1f77b4",
        linewidth=2,
    )

    # Plot posterior (fitted normal)
    posterior_param = posterior_samples[:, 0]
    mu, sigma = stats.norm.fit(posterior_param)

    posterior_pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(
        x,
        posterior_pdf,
        color="#ff7f0e",
        linewidth=2,
        label=f"Posterior (Normal)\nμ={mu:.4e}, σ={sigma:.4e}",
    )
    ax.fill_between(x, 0, posterior_pdf, alpha=0.3, color="#ff7f0e")

    ax.set_xlabel(param_label, fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(True, alpha=0.3)

    ax.axvline(lower_bound, color="#1f77b4", linestyle="--", alpha=0.7, linewidth=1)
    ax.axvline(upper_bound, color="#1f77b4", linestyle="--", alpha=0.7, linewidth=1)

    fig.tight_layout()

    outpath = os.path.join(outdir, f"{model_name}_prior_posterior_overlay.eps")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(outpath)

    return paths


def process_loaded_chains(
    chains: np.ndarray, burn_in_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process loaded chains by removing burn-in and creating flat samples.

    Args:
        chains: array of shape (nchains, nsteps, ndim)
        burn_in_frac: fraction of chain to discard as burn-in

    Returns:
        chains_post: chains after burn-in removal
        flat_samples: flattened posterior samples
    """
    nchains, nsteps, ndim = chains.shape
    burn_in_steps = int(burn_in_frac * nsteps)

    if burn_in_steps >= nsteps:
        raise ValueError("burn-in fraction too large")

    chains_post = chains[:, burn_in_steps:, :]
    flat_samples = chains_post.reshape(-1, ndim)

    return chains_post, flat_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convergence diagnostics and prior-vs-posterior analysis for dispersion models: "
            "computes Gelman–Rubin R-hat, generates trace plots, and "
            "overlays prior and posterior distributions."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dispersion_1",
        choices=["dispersion_1", "dispersion_2", "dispersion_3"],
        help="Model to analyze",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--min_samples", type=int, default=100, help="Minimum samples per chain"
    )
    parser.add_argument(
        "--max_chains", type=int, default=5, help="Maximum number of chains to create"
    )
    parser.add_argument(
        "--burnin_frac",
        type=float,
        default=0.25,
        help="Fraction of chain to discard as burn-in",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--outdir",
        type=str,
        default="conv_output",
        help="Directory for saving plots",
    )
    parser.add_argument(
        "--create_table",
        action="store_true",
        help="Create comprehensive R-hat table for all models",
    )
    parser.add_argument(
        "--create_artificial_chains",
        action="store_true",
        help="Create artificial chains by splitting single-chain data for R-hat computation",
    )

    args = parser.parse_args()

    cfg = DataConfig(
        data_dir=args.data_dir,
        min_samples_per_chain=args.min_samples,
        max_chains=args.max_chains,
        random_seed=args.seed,
    )

    ensure_outdir(args.outdir)

    print(f"Loading chains for model {args.model} from {args.data_dir}...")
    chains, prior_bounds = load_mcmc_chains(args.model, args.data_dir, cfg)
    print(
        f"Loaded parameter data: {chains.shape[1]} samples, {chains.shape[2]} parameters"
    )

    chains_post, flat_samples = process_loaded_chains(chains, args.burnin_frac)
    print(f"After burn-in removal: {chains_post.shape[1]} samples")

    # For single chain, we can't compute R-hat (need multiple chains)
    # So we'll skip the convergence diagnostics
    print("\n=== Note ===")
    print("Single chain detected - R-hat convergence diagnostics require multiple chains.")
    print("Trace plot and prior-posterior analysis will still be generated.")
    print("To assess convergence, consider running multiple independent MCMC chains.")

    display_names = get_display_names(args.model)
    if not display_names:
        raise ValueError(f"Model {args.model} has no free parameters")

    trace_path = plot_representative_trace(
        chains_post, args.outdir, args.model, display_names
    )

    overlay_paths = plot_prior_posterior_overlays(
        posterior_samples=flat_samples,
        prior_bounds=prior_bounds,
        outdir=args.outdir,
        model_name=args.model,
        param_names=display_names,
    )

    # Skip R-hat computation for single chain
    print("\n=== Trace plot ===")
    if trace_path:
        print(f"Saved: {trace_path}")
    else:
        print("Skipped (matplotlib not available)")

    print("\n=== Priors specification ===")
    print("Actual priors used in MCMC (from model.py):")
    for i, (name, lower, upper) in enumerate(
        zip(display_names, prior_bounds["lower"], prior_bounds["upper"])
    ):
        print(f"  {name}: uniform U({lower:.2e}, {upper:.2e})")

    print("\n=== Prior vs Posterior overlays ===")
    if overlay_paths:
        for pth in overlay_paths:
            print(f"Saved: {pth}")
    else:
        print("Skipped (matplotlib not available)")

    print(
        "\nInterpretation notes: If the posterior density is concentrated away from the prior bounds and "
        "differs substantially from the (uniform) prior histogram, this indicates that the data "
        "dominate the posterior (i.e., results are not prior-dominated)."
    )

    if args.create_table:
        print("\n=== Creating comprehensive parameter and convergence table ===")
        all_models = ["dispersion_1", "dispersion_2", "dispersion_3"]
        table_paths = create_rhat_table(all_models, args.data_dir, cfg, args.outdir, args.create_artificial_chains)
        if table_paths:
            eps_path, csv_path = table_paths
            print(f"Comprehensive table (EPS) saved to: {eps_path}")
            print(f"Comprehensive table (CSV) saved to: {csv_path}")
        else:
            print("Comprehensive table creation failed")


if __name__ == "__main__":
    main()
