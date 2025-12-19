"""
Master script for Homework 7: Flavor SU(3) Symmetry

This script runs all tasks consecutively:
  Task 1: Generate SU(3) multiplet diagrams
  Task 2: Generate SU(3)-related decay chains via ladder operators
  Task 3: Coupling mapping with PDG values and SU(3) CG factors (integrated into particles module)
  Task 4: Generate Dalitz plots for reference and related decays
"""

import os
import sys
import traceback

# Allow running as a script (python src/run_homework.py) by ensuring project root is on sys.path
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib import dalitz_plotter
from src.lib import dalitz_plotter_all
from src.lib import decay_generator
from src.lib import multiplet_plotter


# Ensure output directories exist
os.makedirs("pictures/multiplets", exist_ok=True)
os.makedirs("pictures/dalitz", exist_ok=True)


def main(show_plots: bool = False, smooth_dalitz: bool = False, resolution: int = 300):
    """Run all homework tasks consecutively."""

    print("\n" + "=" * 80)
    print(" " * 20 + "HOMEWORK 7: FLAVOR SU(3) SYMMETRY")
    print(" " * 15 + "Master Script - Running All Tasks Consecutively")
    print("=" * 80 + "\n")

    print("Configuration:")
    print(f"  - Show plots: {show_plots}")
    print(f"  - Smooth Dalitz plots: {smooth_dalitz}")
    print(f"  - Dalitz resolution: {resolution}×{resolution} grid")
    print()

    # TASK 1: Multiplet diagrams
    print("\n" + "-" * 80)
    print("TASK 1: SU(3) Multiplet Diagrams")
    print("-" * 80 + "\n")
    try:
        multiplet_plotter.main(show_plots=show_plots)
        print("\n✓ Task 1 completed successfully!")
    except Exception as e:
        print(traceback.format_exc())
        print(f"\n✗ Task 1 failed: {e}")
        sys.exit(1)

    # TASK 2: Decay chains via ladder operators
    print("\n" + "-" * 80)
    print("TASK 2: SU(3)-Related Decays and Ladder Operators")
    print("-" * 80 + "\n")
    try:
        decay_generator.main()
        print("\n✓ Task 2 completed successfully!")
    except Exception as e:
        print(traceback.format_exc())
        print(f"\n✗ Task 2 failed: {e}")
        sys.exit(1)

    # TASK 3: Coupling mapping (informational)
    print("\n" + "-" * 80)
    print("TASK 3: Coupling Mapping with PDG Values and SU(3) CG Factors")
    print("-" * 80 + "\n")
    print("✓ Task 3 is integrated into the particles module:")
    print("  - PDG values loaded from mapping_couplings.py")
    print("  - SU(3) Clebsch-Gordan factors applied to g_prod and g_dec")
    print("  - Resonance couplings mapped across charge states")
    print("\n✓ Task 3 completed successfully!")

    # TASK 4a: Reference Dalitz plot
    print("\n" + "-" * 80)
    print("TASK 4a: Reference Decay Dalitz Plot (Λc⁺ → p π⁺ K⁻)")
    print("-" * 80 + "\n")
    try:
        dalitz_plotter.main(resolution=resolution, smooth=smooth_dalitz, show_plots=show_plots)
        print("\n✓ Task 4a completed successfully!")
    except Exception as e:
        print(traceback.format_exc())
        print(f"\n✗ Task 4a failed: {e}")
        sys.exit(1)

    # TASK 4b: All usable Dalitz plots
    print("\n" + "-" * 80)
    print("TASK 4b: Dalitz Plots for All Usable SU(3)-Related Decays")
    print("-" * 80 + "\n")
    try:
        dalitz_plotter_all.generate_all_dalitz_plots(resolution=resolution, smooth=smooth_dalitz, show_plots=show_plots)
        print("\n✓ Task 4b completed successfully!")
    except Exception as e:
        print(traceback.format_exc())
        print(f"\n✗ Task 4b failed: {e}")
        sys.exit(1)

    # COMPLETION SUMMARY
    print("\n" + "=" * 80)
    print(" " * 25 + "ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")

    print("Summary of generated files:")
    print("\nTask 1 - Multiplet Diagrams:")
    print("  ✓ pictures/multiplets/baryon_octet.png")
    print("  ✓ pictures/multiplets/meson_octet.png")
    print("  ✓ pictures/multiplets/baryon_decuplet.png")
    print("  ✓ pictures/multiplets/vector_nonet.png")
    print("  ✓ pictures/multiplets/lambda_star_resonance.png")
    print("  ✓ pictures/multiplets/charmed_baryons.png")
    print("  ✓ pictures/multiplets/decay_summary.png")

    print("\nTask 2 - Decay Chains:")
    print("  ✓ outputs/decay_chains.json (33 SU(3)-related decays)")

    print("\nTask 3 - Couplings:")
    print("  ✓ PDG values integrated from mapping_couplings.py")
    print("  ✓ SU(3) CG factors applied in particles.py")

    print("\nTask 4 - Dalitz Plots:")
    print("  ✓ pictures/dalitz/dalitz_plot_lambda_c_to_p_pi_k.png (reference)")
    print("  ✓ pictures/dalitz/dalitz_plot_lambda_c_to_*.png (all usable decays)")


if __name__ == "__main__":
    main(show_plots=False, smooth_dalitz=False, resolution=100)