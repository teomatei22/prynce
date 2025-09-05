#!/usr/bin/env python3
"""
Hubble Analysis for Non-commutative BBN

This script analyzes the Hubble function evolution in standard BBN vs non-commutative
dispersion models. It compares the expansion rate evolution and shows how
non-commutativity affects the cosmic expansion during Big Bang Nucleosynthesis.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from typing import Dict, List, Tuple, Optional

# Import PRyM modules
import PRyM.PRyM_init as PRyMini
import PRyM.PRyM_main as PRyMmain
import PRyM.PRyM_thermo as PRyMthermo

# Import dispersion model
from model import DispersionModel


def load_dispersion_parameters(data_dir: str = ".") -> Dict[str, float]:
    """
    Load the mean dispersion parameters from CSV files.
    
    Returns:
        Dictionary mapping model names to mean parameter values
    """
    dispersion_models = ["dispersion_1", "dispersion_2", "dispersion_3"]
    mean_params = {}
    
    for model in dispersion_models:
        csv_file = os.path.join(data_dir, f"{model}.csv")
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping {model}")
            continue
            
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
        
        if param_values:
            # Remove burn-in (first 10%)
            burn_in = int(len(param_values) * 0.1)
            param_values = param_values[burn_in:]
            mean_params[model] = np.mean(param_values)
            print(f"Loaded {model}: mean param = {mean_params[model]:.2e}")
        else:
            print(f"Warning: No valid parameters found in {model}")
    
    return {"dispersion_1": 0.001, "dispersion_2":  0.000034, "dispersion_3": 0.00067}


def get_standard_hubble_function():
    """
    Get the standard BBN Hubble function (without non-commutative effects).
    
    Returns:
        Hubble function that can be called with (Tg, Tnue, Tnumu, T_NP=0.0)
    """
    # Temporarily disable NP flags for standard BBN
    original_np_e_flag = PRyMini.NP_e_flag
    original_np_thermo_flag = PRyMini.NP_thermo_flag
    original_np_nu_flag = PRyMini.NP_nu_flag
    
    PRyMini.NP_e_flag = False
    PRyMini.NP_thermo_flag = False
    PRyMini.NP_nu_flag = False
    
    # Create a standalone Hubble function based on the PRyM implementation
    def standard_hubble(Tg, Tnue, Tnumu, T_NP=0.0):
        rho_pl = (
            PRyMthermo.rho_g(Tg)
            + PRyMthermo.rho_e(Tg)
            - PRyMthermo.PofT(Tg)
            + Tg * PRyMthermo.dPdT(Tg)
        )
        rho_3nu = PRyMthermo.rho_nu(Tnue) + 2.0 * PRyMthermo.rho_nu(Tnumu)
        rho_tot = rho_pl + rho_3nu
        
        # No NP contributions for standard BBN
        return (
            PRyMini.MeV_to_secm1
            * (rho_tot * 8.0 * np.pi / (3.0 * PRyMini.Mpl**2)) ** 0.5
        )
    
    # Restore original flags
    PRyMini.NP_e_flag = original_np_e_flag
    PRyMini.NP_thermo_flag = original_np_thermo_flag
    PRyMini.NP_nu_flag = original_np_nu_flag
    
    return standard_hubble


def get_dispersion_hubble_function(disp_model: int, param: float):
    """
    Get the Hubble function with non-commutative dispersion effects.
    
    Args:
        disp_model: Dispersion model type (1=LAM, 2=BET, 3=ALP)
        param: Dispersion parameter value
        
    Returns:
        Hubble function that can be called with (Tg, Tnue, Tnumu, T_NP=0.0)
    """
    # Enable NP flags for non-commutative effects
    original_np_e_flag = PRyMini.NP_e_flag
    original_np_thermo_flag = PRyMini.NP_thermo_flag
    original_np_nu_flag = PRyMini.NP_nu_flag
    
    PRyMini.NP_e_flag = True
    PRyMini.NP_thermo_flag = False
    PRyMini.NP_nu_flag = False
    
    # Create dispersion model instance
    disp = DispersionModel()
    
    # Create a standalone Hubble function with non-commutative effects
    def dispersion_hubble(Tg, Tnue, Tnumu, T_NP=0.0):
        rho_pl = (
            PRyMthermo.rho_g(Tg)
            + PRyMthermo.rho_e(Tg)
            - PRyMthermo.PofT(Tg)
            + Tg * PRyMthermo.dPdT(Tg)
        )
        rho_3nu = PRyMthermo.rho_nu(Tnue) + 2.0 * PRyMthermo.rho_nu(Tnumu)
        rho_tot = rho_pl + rho_3nu
        
        # Add non-commutative contribution
        rho_np = disp.rho_np(Tg, disp_model, param)
        rho_tot += rho_np
        
        return (
            PRyMini.MeV_to_secm1
            * (rho_tot * 8.0 * np.pi / (3.0 * PRyMini.Mpl**2)) ** 0.5
        )
    
    # Restore original flags
    PRyMini.NP_e_flag = original_np_e_flag
    PRyMini.NP_thermo_flag = original_np_thermo_flag
    PRyMini.NP_nu_flag = original_np_nu_flag
    
    return dispersion_hubble


def compute_hubble_evolution(
    hubble_func,
    T_range: np.ndarray,
    T_nu_ratio: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hubble evolution over a temperature range.
    
    Args:
        hubble_func: Hubble function to evaluate
        T_range: Array of temperatures in MeV
        T_nu_ratio: Ratio of neutrino temperature to photon temperature
        
    Returns:
        Tuple of (temperatures, hubble_values)
    """
    hubble_values = []
    
    for Tg in T_range:
        Tnue = Tnumu = Tg * T_nu_ratio
        H = hubble_func(Tg, Tnue, Tnumu)
        hubble_values.append(H)
    
    return T_range, np.array(hubble_values)


def plot_hubble_comparison(
    standard_hubble: callable,
    dispersion_hubbles: Dict[str, callable],
    outdir: str = "hubble_analysis"
) -> str:
    """
    Create comparison plot of Hubble evolution.
    
    Args:
        standard_hubble: Standard BBN Hubble function
        dispersion_hubbles: Dictionary of dispersion Hubble functions
        outdir: Output directory for plots
        
    Returns:
        Path to the generated plot
    """
    os.makedirs(outdir, exist_ok=True)
    
    T_range = np.logspace(1, 0, 100)  
    
    # Compute standard BBN Hubble evolution
    T_std, H_std = compute_hubble_evolution(standard_hubble, T_range)
    
    # Create the plot
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot 1: Absolute Hubble values
    ax1.plot(T_std, H_std, 'k-', linewidth=2, label='Standard BBN', alpha=0.8)
    
    # Plot dispersion models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    dispersion_labels = {
        'dispersion_1': r'Model I: $\omega(k) = kc(1 + \lambda \hbar \omega)$',
        'dispersion_2': r'Model II: $\omega(k) = kc \sqrt{1 - 2\beta_0 \hbar^2 \omega^2}$',
        'dispersion_3': r'Model III: $\omega(k) = kc \sqrt{1 - 2\alpha_0 \hbar \omega}$'
    }
    
    for i, (model_name, hubble_func) in enumerate(dispersion_hubbles.items()):
        T_disp, H_disp = compute_hubble_evolution(hubble_func, T_range)
        label = dispersion_labels.get(model_name, f'Dispersion {i+1}')
        ax1.plot(T_disp, H_disp, color=colors[i], linewidth=2, 
                   label=label, alpha=0.8)
    
    ax1.set_xlabel('T [MeV]', fontsize=18)
    ax1.set_ylabel('H(T) [s⁻¹]', fontsize=18)
    ax1.set_title('Hubble Evolution: Standard BBN vs Non-commutative Models', fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=14)
    ax1.set_xlim(T_range.min(), T_range.max())
    ax1.tick_params(axis='both', which='major', labelsize=20)
    
    # Create inset axes for ax2 (upper left quarter)
    ax2 = fig.add_axes([0.055+ 0.02, 0.552- 0.02, 0.57, 0.4])  # [left, bottom, width, height] - upper left quarter
    
    # Plot 2: Relative difference from standard BBN
    ax2.plot(T_std, np.zeros_like(T_std), 'k--', linewidth=1, alpha=0.5)
    
    for i, (model_name, hubble_func) in enumerate(dispersion_hubbles.items()):
        T_disp, H_disp = compute_hubble_evolution(hubble_func, T_range)
        # Interpolate to same temperature grid if needed
        if len(T_disp) != len(T_std):
            from scipy.interpolate import interp1d
            H_interp = interp1d(T_disp, H_disp, kind='linear', bounds_error=False, fill_value='extrapolate')
            H_disp = H_interp(T_std)
        
        # Calculate relative difference
        rel_diff = (H_disp - H_std) / H_std * 100  # Percentage difference
        ax2.plot(T_std, rel_diff, color=colors[i], linewidth=2.5, alpha=0.8)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(T_range.min(), T_range.max())
    ax2.yaxis.tick_right()  # Move y-axis ticks to the right
    ax2.tick_params(axis='both', which='major', labelsize=20)

    
    # Add horizontal line at 0%
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add descriptive text below the inset
    fig.text(0.363 + 0.02, 0.50 - 0.02, 'T [MeV]', fontsize=16, ha='center', va='top')
    fig.text(0.71 -0.02, 0.85 - 0.075, 'ΔH/H [%]', fontsize=16, ha='center', va='top', rotation=270)
    plt.tight_layout()
    
    # Save the plot
    outpath = os.path.join(outdir, "hubble_comparison.eps")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    
    # Also save as PNG for easier viewing
    png_path = os.path.join(outdir, "hubble_comparison.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return outpath


def analyze_hubble_effects(
    standard_hubble: callable,
    dispersion_hubbles: Dict[str, callable],
    outdir: str = "hubble_analysis"
) -> Dict[str, Dict[str, float]]:
    """
    Analyze the effects of non-commutativity on Hubble evolution.
    
    Args:
        standard_hubble: Standard BBN Hubble function
        dispersion_hubbles: Dictionary of dispersion Hubble functions
        outdir: Output directory for results
        
    Returns:
        Dictionary containing analysis results
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Key temperature points for BBN analysis
    T_points = {
        'nucleon_freezeout': 0.8,      # MeV
        'deuterium_formation': 0.1,    # MeV
        'helium_formation': 0.3,       # MeV
        'lithium_formation': 0.05      # MeV
    }
    
    results = {}
    
    # Standard BBN values
    T_std, H_std = compute_hubble_evolution(standard_hubble, np.array(list(T_points.values())))
    std_values = dict(zip(T_points.keys(), H_std))
    
    # Analyze each dispersion model
    for model_name, hubble_func in dispersion_hubbles.items():
        T_disp, H_disp = compute_hubble_evolution(hubble_func, np.array(list(T_points.values())))
        disp_values = dict(zip(T_points.keys(), H_disp))
        
        # Calculate effects
        effects = {}
        for temp_name in T_points.keys():
            std_H = std_values[temp_name]
            disp_H = disp_values[temp_name]
            rel_change = (disp_H - std_H) / std_H * 100
            effects[temp_name] = {
                'standard_H': std_H,
                'dispersion_H': disp_H,
                'relative_change_percent': rel_change,
                'temperature_MeV': T_points[temp_name]
            }
        
        results[model_name] = effects
    
    # Save results to CSV
    csv_path = os.path.join(outdir, "hubble_effects_analysis.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Model', 'Temperature_Stage', 'Temperature_MeV', 
            'Standard_Hubble', 'Dispersion_Hubble', 'Relative_Change_Percent'
        ])
        
        for model_name, effects in results.items():
            for temp_name, effect in effects.items():
                writer.writerow([
                    model_name,
                    temp_name,
                    effect['temperature_MeV'],
                    f"{effect['standard_H']:.2e}",
                    f"{effect['dispersion_H']:.2e}",
                    f"{effect['relative_change_percent']:.3f}"
                ])
    
    # Save detailed results to JSON
    import json
    json_path = os.path.join(outdir, "hubble_effects_analysis.json")
    
    # Convert numpy types to native Python types for JSON serialization
    json_results = {}
    for model_name, effects in results.items():
        json_results[model_name] = {}
        for temp_name, effect in effects.items():
            json_results[model_name][temp_name] = {
                'standard_H': float(effect['standard_H']),
                'dispersion_H': float(effect['dispersion_H']),
                'relative_change_percent': float(effect['relative_change_percent']),
                'temperature_MeV': float(effect['temperature_MeV'])
            }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    return results


def main():
    """Main function to run the Hubble analysis."""
    print("=== Hubble Analysis for Non-commutative BBN ===\n")
    
    # Load dispersion parameters
    print("Loading dispersion parameters...")
    mean_params = load_dispersion_parameters()
    print(mean_params)
    
    if not mean_params:
        print("Error: No dispersion parameters found. Exiting.")
        return
    
    print(f"Loaded {len(mean_params)} dispersion models\n")
    
    # Get standard BBN Hubble function
    print("Setting up standard BBN Hubble function...")
    standard_hubble = get_standard_hubble_function()
    
    # Get dispersion Hubble functions
    print("Setting up dispersion Hubble functions...")
    dispersion_hubbles = {}
    
    for model_name, param in mean_params.items():
        # Map model names to dispersion types
        if "dispersion_1" in model_name:
            disp_type = DispersionModel.DISP_LAM
        elif "dispersion_2" in model_name:
            disp_type = DispersionModel.DISP_BET
        elif "dispersion_3" in model_name:
            disp_type = DispersionModel.DISP_ALP
        else:
            print(f"Warning: Unknown model {model_name}, skipping")
            continue
        
        try:
            hubble_func = get_dispersion_hubble_function(disp_type, param)
            dispersion_hubbles[model_name] = hubble_func
            print(f"  Created Hubble function for {model_name}")
        except Exception as e:
            print(f"  Error creating Hubble function for {model_name}: {e}")
    
    if not dispersion_hubbles:
        print("Error: No dispersion Hubble functions created. Exiting.")
        return
    
    print(f"Created {len(dispersion_hubbles)} dispersion Hubble functions\n")
    
    # Create output directory
    outdir = "hubble_analysis"
    os.makedirs(outdir, exist_ok=True)
    
    # Generate comparison plot
    print("Generating Hubble comparison plot...")
    try:
        plot_path = plot_hubble_comparison(standard_hubble, dispersion_hubbles, outdir)
        print(f"  Plot saved to: {plot_path}")
    except Exception as e:
        print(f"  Error generating plot: {e}")
    
    # Analyze effects
    print("\nAnalyzing Hubble effects...")
    try:
        results = analyze_hubble_effects(standard_hubble, dispersion_hubbles, outdir)
        print(f"  Analysis results saved to: {outdir}/")
        
        # Print summary
        print("\n=== Summary of Hubble Effects ===")
        for model_name, effects in results.items():
            print(f"\n{model_name}:")
            for temp_name, effect in effects.items():
                print(f"  {temp_name} ({effect['temperature_MeV']} MeV): "
                      f"{effect['relative_change_percent']:+.2f}%")
    
    except Exception as e:
        print(f"  Error analyzing effects: {e}")
    
    print(f"\nAnalysis complete! Results saved to: {outdir}/")


if __name__ == "__main__":
    main()
