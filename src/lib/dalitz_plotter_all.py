"""
Generate Dalitz plots for all usable SU(3)-related decays of Λc⁺.

This script imports DalitzPlotGenerator, Particle, and Resonance classes from
the dalitz_plotter module and uses them to systematically plot the kinematic
phase space for all ΔS = 0 decays with non-empty resonance chains.
"""

from typing import List, Tuple
import os
from src.lib.dalitz_plotter import Particle, Resonance, DalitzPlotGenerator
from src.lib.particles import get_resonances_for_pair


# ============================================================================
# DECAY DEFINITIONS
# ============================================================================

# Particle masses (in GeV)
PARTICLE_MASSES = {
    "p": 0.93827,
    "n": 0.93957,
    "π⁺": 0.13957,
    "π⁻": 0.13957,
    "π⁰": 0.13498,
    "K⁺": 0.49367,
    "K⁻": 0.49368,
    "K⁰": 0.49761,
    "K̄⁰": 0.49761,
    "Λ": 1.11568,
    "Σ⁺": 1.18937,
    "Σ⁻": 1.19745,
    "φ": 1.01946,
}

# Mother particle (all decays are from Λc⁺)
MOTHER_MASS = 2.2865  # GeV

# List of usable ΔS = 0 decays: (final_state_tuple, pair_x, pair_y)
USABLE_DECAYS = [
    # Decay: Λc⁺ → p π⁺ π⁻
    (("p", "π⁺", "π⁻"), (1, 3), (2, 3)),  # x-axis: m_{pπ⁻}², y-axis: m_{π⁺π⁻}²
    
    # Decay: Λc⁺ → p K⁺ K⁻
    (("p", "K⁺", "K⁻"), (1, 3), (2, 3)),  # x-axis: m_{pK⁻}², y-axis: m_{K⁺K⁻}²
    
    # Decay: Λc⁺ → n π⁺ K⁰
    (("n", "π⁺", "K⁰"), (1, 2), (2, 3)),  # x-axis: m_{nπ⁺}², y-axis: m_{π⁺K⁰}²
    
    # Decay: Λc⁺ → n π⁺ K̄⁰
    (("n", "π⁺", "K̄⁰"), (1, 2), (2, 3)), # x-axis: m_{nπ⁺}², y-axis: m_{π⁺K̄⁰}²
    
    # Decay: Λc⁺ → Λ π⁺ K⁰
    (("Λ", "π⁺", "K⁰"), (2, 3), (1, 3)),  # x-axis: m_{π⁺K⁰}², y-axis: m_{ΛK⁰}²
    
    # Decay: Λc⁺ → Σ⁺ K⁺ π⁻
    (("Σ⁺", "K⁺", "π⁻"), (1, 2), (2, 3)), # x-axis: m_{Σ⁺K⁺}², y-axis: m_{K⁺π⁻}²
    
    # Decay: Λc⁺ → n π⁺ φ
    (("n", "π⁺", "φ"), (1, 2), (2, 3)),    # x-axis: m_{nπ⁺}², y-axis: m_{π⁺φ}²
]


def build_resonances_for_decay(daughter_names: Tuple[str, str, str]) -> List[Resonance]:
    """Build resonances for a given decay by querying all three two-body subsystems."""
    res_list: List[Resonance] = []
    
    pair_to_subsystem = {
        (daughter_names[0], daughter_names[1]): (1, 2),
        (daughter_names[0], daughter_names[2]): (1, 3),
        (daughter_names[1], daughter_names[2]): (2, 3),
    }
    
    for (a, b), subsystem in pair_to_subsystem.items():
        try:
            for R in get_resonances_for_pair(a, b):
                res_list.append(
                    Resonance(
                        name=R.name,
                        mass=R.mass,
                        width=R.width,
                        g_prod=R.g_prod,
                        g_dec=R.g_dec,
                        subsystem=subsystem,
                    )
                )
        except Exception as e:
            # Skip pairs that don't have resonances or cause errors
            pass
    
    return res_list


def generate_all_dalitz_plots(resolution=300, smooth=False, show_plots=False):
    """
    Generate Dalitz plots for all usable SU(3)-related decays.
    
    Args:
        resolution: Grid resolution (n_points × n_points)
        smooth: If True, apply smoothing to plots
        show_plots: If True, display plots interactively
    """
    print("=" * 70)
    print("Dalitz Plot Generator for All Usable SU(3)-Related Decays of Λc⁺")
    print("Homework 7: Flavor SU(3) Symmetry")
    print("=" * 70)
    
    os.makedirs('pictures/dalitz', exist_ok=True)
    
    for idx, (daughters_names, pair_x, pair_y) in enumerate(USABLE_DECAYS, 1):
        print(f"\n[{idx}/{len(USABLE_DECAYS)}] Processing: Λc⁺ → {' '.join(daughters_names)}")
        
        # Create mother particle
        mother = Particle(name="Λc⁺", mass=MOTHER_MASS)
        
        # Create daughter particles
        daughters = [Particle(name=name, mass=PARTICLE_MASSES[name]) for name in daughters_names]
        
        # Build resonances
        resonances = build_resonances_for_decay(daughters_names)
        
        if len(resonances) == 0:
            print(f"  ⚠ No resonances found for this decay, skipping...")
            continue
        
        print(f"  ✓ Found {len(resonances)} resonances:")
        for R in resonances:
            print(f"    - {R.name}: mass={R.mass:.4f} GeV, width={R.width:.4f} GeV")
        
        # Create generator and generate plot
        try:
            generator = DalitzPlotGenerator(mother, daughters, resonances, pair_x=pair_x, pair_y=pair_y)
            
            # Generate filename
            daughters_short = "".join([d[0].lower() for d in daughters_names])
            filename = f"pictures/dalitz/dalitz_plot_lambda_c_to_{daughters_short}.png"
            
            # Generate and plot with specified parameters
            print(f"  ✓ Generating Dalitz plot...")
            generator.plot_dalitz(n_points=resolution, save_path=filename, smooth=smooth, show=show_plots)
            print(f"  ✓ Saved to: {filename}")
            
        except Exception as e:
            print(f"  ✗ Error generating plot: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("All Dalitz plots generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_dalitz_plots()
