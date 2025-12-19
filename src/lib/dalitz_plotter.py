import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
from src.lib.particles import get_resonances_for_pair


@dataclass
class Particle:
    """Represents a particle with its mass."""
    name: str
    mass: float  # in GeV


@dataclass
class Resonance:
    """Represents a resonance with its properties."""
    name: str
    mass: float  # in GeV
    width: float  # in GeV
    g_prod: complex  # production coupling
    g_dec: complex  # decay coupling
    subsystem: Tuple[int, int]  # indices of the two particles forming the resonance


class DalitzPlotGenerator:
    """Generates Dalitz plots for three-body decays with proper kinematics."""
    
    def __init__(self, mother: Particle, daughters: List[Particle], resonances: List[Resonance],
                 pair_x: Tuple[int, int] = (1, 2), pair_y: Tuple[int, int] = (2, 3)):
        """
        Initialize the Dalitz plot generator.
        
        Args:
            mother: The decaying particle (index 0)
            daughters: List of three daughter particles (indices 1, 2, 3)
            resonances: List of resonances contributing to the decay
            pair_x: The two-particle subsystem whose m^2 is plotted on x-axis
            pair_y: The two-particle subsystem whose m^2 is plotted on y-axis
        """
        self.mother = mother
        self.daughters = daughters
        self.resonances = resonances
        self.pair_x = tuple(sorted(pair_x))
        self.pair_y = tuple(sorted(pair_y))
        # The remaining pair (third invariant mass)
        all_pairs = {(1, 2), (2, 3), (1, 3)}
        self.pair_z = list(all_pairs.difference({self.pair_x, self.pair_y}))[0]
        
    @staticmethod
    def kallen_function(a: float, b: float, c: float) -> float:
        """
        Calculate the Källén function λ(a, b, c).
        
        λ(a, b, c) = a² + b² + c² - 2(ab + bc + ca)
        
        Args:
            a, b, c: Input parameters
            
        Returns:
            Value of the Källén function
        """
        return a**2 + b**2 + c**2 - 2*(a*b + b*c + c*a)
    
    def breit_wigner(self, m_ij_sq: float, resonance: Resonance) -> complex:
        """
        Calculate the Breit-Wigner amplitude for a resonance.
        
        A_R(m_ij²) = 1 / (m_R² - m_ij² - i m_R Γ_R)
        
        Args:
            m_ij_sq: Invariant mass squared of the subsystem
            resonance: Resonance parameters
            
        Returns:
            Complex Breit-Wigner amplitude
        """
        m_R = resonance.mass
        gamma_R = resonance.width
        
        denominator = m_R**2 - m_ij_sq - 1j * m_R * gamma_R
        return 1.0 / denominator
    
    def calculate_third_invariant(self, mX_sq: float, mY_sq: float) -> float:
        """
        Calculate the third invariant m_Z² from two invariants using energy-momentum conservation.
        
        m_Z² = m_0² + m_1² + m_2² + m_3² - m_X² - m_Y²
        
        Args:
            mX_sq: Invariant mass squared for x-axis pair
            mY_sq: Invariant mass squared for y-axis pair
            
        Returns:
            m_Z² value
        """
        m0_sq = self.mother.mass**2
        m1_sq = self.daughters[0].mass**2
        m2_sq = self.daughters[1].mass**2
        m3_sq = self.daughters[2].mass**2
        
        return m0_sq + m1_sq + m2_sq + m3_sq - mX_sq - mY_sq
    
    def is_kinematically_allowed(self, mX_sq: float, mY_sq: float) -> bool:
        """
        Check if a point in the Dalitz plot is kinematically allowed.
        
        Physical region requires:
        1. All three Källén functions ≥ 0 (individual subsystems physical)
        2. Kibble function ≥ 0 (overall phase space constraint)
        
        Args:
            m12_sq: Invariant mass squared of particles 1 and 2
            m23_sq: Invariant mass squared of particles 2 and 3
            
        Returns:
            True if kinematically allowed, False otherwise
        """
        m0_sq = self.mother.mass**2
        m1_sq = self.daughters[0].mass**2
        m2_sq = self.daughters[1].mass**2
        m3_sq = self.daughters[2].mass**2
        
        mZ_sq = self.calculate_third_invariant(mX_sq, mY_sq)
        
        # Threshold checks: m_ij² must be positive
        if mX_sq <= 0 or mY_sq <= 0 or mZ_sq <= 0:
            return False
        
        # Calculate all three Källén functions on the chosen axes
        # Note: following the code's established (approximate) check structure
        def spectator_index(pair: Tuple[int, int]) -> int:
            return ({1, 2, 3} - set(pair)).pop()

        idx_k_x = spectator_index(self.pair_x)
        idx_k_y = spectator_index(self.pair_y)
        idx_k_z = spectator_index(self.pair_z)

        m_k_x_sq = [None, m1_sq, m2_sq, m3_sq][idx_k_x]
        m_k_y_sq = [None, m1_sq, m2_sq, m3_sq][idx_k_y]
        m_k_z_sq = [None, m1_sq, m2_sq, m3_sq][idx_k_z]

        lambda_x = self.kallen_function(mX_sq, m_k_x_sq, m0_sq)
        lambda_y = self.kallen_function(mY_sq, m_k_y_sq, m0_sq)
        lambda_z = self.kallen_function(mZ_sq, m_k_z_sq, m0_sq)
        
        # Step 1: All individual Källén functions must be ≥ 0
        # This ensures each two-body subsystem is kinematically allowed
        if lambda_x < 0 or lambda_y < 0 or lambda_z < 0:
            return False
        
        # Step 2: Kibble function must be ≥ 0
        # φ = λ(λ₁₂, λ₂₃, λ₃₁) ≥ 0
        # This is the constraint that defines the physical region boundary
        phi = self.kallen_function(lambda_x, lambda_y, lambda_z)
        
        return phi < 1e-10  # Allow small numerical errors (show non-physical region)
    
    def get_m23_limits_for_m12(self, m12_sq: float) -> Tuple[float, float]:
        """
        Get the kinematically allowed range of m23² for a given m12².
        
        This is computed by finding where the Kibble function = 0.
        
        Args:
            m12_sq: Invariant mass squared of particles 1 and 2
            
        Returns:
            Tuple of (m23_min², m23_max²)
        """
        m0_sq = self.mother.mass**2
        m1_sq = self.daughters[0].mass**2
        m2_sq = self.daughters[1].mass**2
        m3_sq = self.daughters[2].mass**2
        
        # Rough boundaries
        m23_sq_min_threshold = (self.daughters[1].mass + self.daughters[2].mass)**2
        m23_sq_max_threshold = (self.mother.mass - self.daughters[0].mass)**2
        
        # Binary search for actual kinematic boundaries
        # Find minimum m23²
        m23_min = m23_sq_min_threshold
        for m23_sq in np.linspace(m23_sq_min_threshold, m23_sq_max_threshold, 100):
            if self.is_kinematically_allowed(m12_sq, m23_sq):
                m23_min = m23_sq
                break
        
        # Find maximum m23²
        m23_max = m23_sq_max_threshold
        for m23_sq in np.linspace(m23_sq_max_threshold, m23_sq_min_threshold, 100):
            if self.is_kinematically_allowed(m12_sq, m23_sq):
                m23_max = m23_sq
                break
        
        return m23_min, m23_max
    
    def matrix_element(self, mX_sq: float, mY_sq: float) -> complex:
        """
        Calculate the decay matrix element.
        
        M(m_12², m_23²) = Σ_R g_prod^(R) · A_R(m_ij²) · g_dec^(R)
        
        Args:
            m12_sq: Invariant mass squared of particles 1 and 2
            m23_sq: Invariant mass squared of particles 2 and 3
            
        Returns:
            Complex matrix element
        """
        mZ_sq = self.calculate_third_invariant(mX_sq, mY_sq)
        
        # Map subsystem indices to invariant masses
        # Build invariant mass map consistent with chosen axes
        inv_map = {}
        inv_map[tuple(sorted(self.pair_x))] = mX_sq
        inv_map[tuple(sorted(self.pair_y))] = mY_sq
        inv_map[self.pair_z] = mZ_sq
        
        amplitude = 0.0 + 0.0j
        
        for resonance in self.resonances:
            # Get the appropriate invariant mass for this resonance
            m_ij_sq = inv_map[tuple(sorted(resonance.subsystem))]
            
            # Calculate Breit-Wigner amplitude
            bw_amplitude = self.breit_wigner(m_ij_sq, resonance)
            
            # Add contribution: g_prod * A_R * g_dec
            amplitude += resonance.g_prod * bw_amplitude * resonance.g_dec
        
        return amplitude
    
    def smooth_intensity_grid(self, m12_sq_grid: np.ndarray, m23_sq_grid: np.ndarray, 
                              intensity_grid: np.ndarray, upsampling_factor: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Smooth the intensity grid using spline interpolation and upsampling.
        
        Args:
            m12_sq_grid: Original m12² grid
            m23_sq_grid: Original m23² grid
            intensity_grid: The raw intensity grid with NaN values
            upsampling_factor: Factor by which to increase grid resolution (higher = smoother)
            
        Returns:
            Tuple of (upsampled_m12_grid, upsampled_m23_grid, upsampled_intensity_grid)
        """
        n_rows, n_cols = intensity_grid.shape
        
        # Create upsampled grid coordinates
        m12_sq_1d = m12_sq_grid[0, :]
        m23_sq_1d = m23_sq_grid[:, 0]
        
        # Create finer grid
        m12_sq_fine = np.linspace(m12_sq_1d.min(), m12_sq_1d.max(), 
                                   n_cols * upsampling_factor)
        m23_sq_fine = np.linspace(m23_sq_1d.min(), m23_sq_1d.max(), 
                                   n_rows * upsampling_factor)
        
        # Create fine meshgrid
        m12_sq_upsampled, m23_sq_upsampled = np.meshgrid(m12_sq_fine, m23_sq_fine)
        
        # Replace NaN with 0 for interpolation
        intensity_no_nan = intensity_grid.copy()
        intensity_no_nan[np.isnan(intensity_no_nan)] = 0
        
        # Create spline interpolator
        spline = RectBivariateSpline(m23_sq_1d, m12_sq_1d, intensity_no_nan, kx=3, ky=3)
        
        # Interpolate to fine grid
        intensity_upsampled = spline(m23_sq_fine, m12_sq_fine)
        
        # Apply Gaussian filter for additional smoothing
        intensity_upsampled = gaussian_filter(intensity_upsampled, sigma=2.0)
        
        # Restore NaN values at original boundaries (extrapolate the boundary)
        valid_mask = ~np.isnan(intensity_grid)
        
        # Simple approach: set values outside the approximate original region to NaN
        # by checking if nearby original points were valid
        for i in range(len(m23_sq_fine)):
            for j in range(len(m12_sq_fine)):
                # Find closest original grid point
                i_orig = int(np.round(i / upsampling_factor))
                j_orig = int(np.round(j / upsampling_factor))
                
                # Clamp to grid bounds
                i_orig = np.clip(i_orig, 0, n_rows - 1)
                j_orig = np.clip(j_orig, 0, n_cols - 1)
                
                # Check if original point was invalid
                if not valid_mask[i_orig, j_orig]:
                    intensity_upsampled[i, j] = np.nan
        
        return m12_sq_upsampled, m23_sq_upsampled, intensity_upsampled
    
    def generate_dalitz_plot(self, n_points: int = 150, smooth: bool = True, smoothing_sigma: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the Dalitz plot with CORRECTED kinematic region.
        
        Strategy:
        1. Create a dense grid of (m12², m23²) points
        2. For EACH point, check if it's kinematically allowed using Kibble function
        3. Only compute amplitudes for allowed points
        4. Optionally smooth the intensity grid for better visualization
        
        Args:
            n_points: Number of points per axis
            smooth: Whether to apply Gaussian smoothing (default: True)
            smoothing_sigma: Smoothing strength (higher = more smoothing, default: 1.5)
            
        Returns:
            Tuple of (m12_sq_grid, m23_sq_grid, intensity_grid)
        """
        print("Generating Dalitz plot with corrected kinematic boundaries...")
        
        # Calculate kinematic boundaries
        m0 = self.mother.mass
        m1, m2, m3 = [d.mass for d in self.daughters]
        
        # Threshold values for chosen axis pairs
        def masses_for_pair(pair: Tuple[int, int]) -> Tuple[float, float, float]:
            i, j = pair
            k = ({1, 2, 3} - set(pair)).pop()
            mi = [None, m1, m2, m3][i]
            mj = [None, m1, m2, m3][j]
            mk = [None, m1, m2, m3][k]
            return mi, mj, mk

        mi_x, mj_x, mk_x = masses_for_pair(self.pair_x)
        mi_y, mj_y, mk_y = masses_for_pair(self.pair_y)

        mX_sq_min_threshold = (mi_x + mj_x)**2
        mX_sq_max_threshold = (m0 - mk_x)**2
        mY_sq_min_threshold = (mi_y + mj_y)**2
        mY_sq_max_threshold = (m0 - mk_y)**2
        
        # Create initial grid with wider bounds
        mX_sq_vals = np.linspace(mX_sq_min_threshold, mX_sq_max_threshold, n_points)
        mY_sq_vals = np.linspace(mY_sq_min_threshold, mY_sq_max_threshold, n_points)
        m12_sq_grid, m23_sq_grid = np.meshgrid(mX_sq_vals, mY_sq_vals)
        
        # Calculate intensity (|M|²) for each point
        intensity_grid = np.zeros_like(m12_sq_grid)
        n_physical = 0
        
        print(f"Checking {n_points}×{n_points} = {n_points**2} grid points...")
        
        for i in range(n_points):
            if i % (n_points // 10) == 0:
                print(f"  Progress: {100*i/n_points:.0f}%")
            
            for j in range(n_points):
                mX_sq = m12_sq_grid[i, j]
                mY_sq = m23_sq_grid[i, j]
                
                # CORRECTED: Use proper kinematic check for EVERY point
                if self.is_kinematically_allowed(mX_sq, mY_sq):
                    amplitude = self.matrix_element(mX_sq, mY_sq)
                    intensity = abs(amplitude)**2
                    intensity_grid[i, j] = intensity
                    n_physical += 1
                else:
                    # Set to NaN so it won't be plotted
                    intensity_grid[i, j] = np.nan
        
        print(f"✓ Physical region: {n_physical} / {n_points**2} points ({100*n_physical/n_points**2:.1f}%)")
        print(f"✓ Non-physical region: {n_points**2 - n_physical} points ({100*(n_points**2-n_physical)/n_points**2:.1f}%)")
        
        # Apply smoothing if requested
        if smooth:
            print(f"✓ Applying spline interpolation and smoothing...")
            m12_sq_grid, m23_sq_grid, intensity_grid = self.smooth_intensity_grid(
                m12_sq_grid, m23_sq_grid, intensity_grid, upsampling_factor=3
            )
        
        return m12_sq_grid, m23_sq_grid, intensity_grid
    
    def plot_dalitz(self, n_points: int = 150, save_path: Optional[str] = None, smooth: bool = False, show: bool = False):
        """
        Generate and display the Dalitz plot with correct physical region.
        
        Args:
            n_points: Number of points per axis (higher = more accurate boundaries)
            save_path: Optional path to save the plot
            smooth: If True, apply smoothing to the intensity grid
            show: If True, display the plot interactively
        """
        m12_sq_grid, m23_sq_grid, intensity_grid = self.generate_dalitz_plot(n_points, smooth=smooth)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Handle NaN values by masking them
        intensity_masked = np.ma.masked_invalid(intensity_grid)
        
        # Find non-NaN maximum for normalization
        vmax = np.nanmax(intensity_grid)
        vmin = vmax * 1e-3
        
        # Plot with log scale for better visualization
        im = ax.pcolormesh(m12_sq_grid, m23_sq_grid, intensity_masked, 
                          shading='auto', cmap='hot',
                          norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        
        cbar = plt.colorbar(im, ax=ax, label='|M|² (Intensity)', pad=0.02)
        
        # Get axis labels based on chosen pairs
        name_map = {1: self.daughters[0].name, 2: self.daughters[1].name, 3: self.daughters[2].name}
        x_label = f"$m_{{{name_map[self.pair_x[0]]}{name_map[self.pair_x[1]]}}}^2$ (GeV²)"
        y_label = f"$m_{{{name_map[self.pair_y[0]]}{name_map[self.pair_y[1]]}}}^2$ (GeV²)"
        
        # Add resonance lines (only within physical region)
        for resonance in self.resonances:
            mass_sq = resonance.mass**2
            if tuple(sorted(resonance.subsystem)) == self.pair_x:
                ax.axvline(mass_sq, color='cyan', linestyle='--', linewidth=2.5, alpha=0.8, label=resonance.name)
            elif tuple(sorted(resonance.subsystem)) == self.pair_y:
                ax.axhline(mass_sq, color='lime', linestyle='--', linewidth=2.5, alpha=0.8, label=resonance.name)
        
        # Labels and formatting
        ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
        daughters_str = " ".join([d.name for d in self.daughters])
        ax.set_title(f'Dalitz Plot: {self.mother.name} → {daughters_str}\n(with corrected kinematic boundaries)', 
                fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            print(f"\nSaving plot to: {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


# ============================================================================
# PARTICLE AND RESONANCE DEFINITIONS
# ============================================================================

# Mother particle (Λ_c⁺)
mother_particle = Particle(name="Λc⁺", mass=2.2865)  # GeV

# Daughter particles
daughter_particles = [
    Particle(name="p", mass=0.93827),     # proton (index 1)
    Particle(name="π⁺", mass=0.13957),    # pion (index 2)
    Particle(name="K⁻", mass=0.49368)     # kaon (index 3)
]

def build_resonances_from_particles() -> List[Resonance]:
    """Build DalitzPlot resonances using particles module with SU(3/PDG couplings."""
    res_list: List[Resonance] = []
    # Define the reference final-state pairs and their subsystem indices
    pair_to_subsystem = {
        ("p", "π⁺"): (1, 2),
        ("p", "K⁻"): (1, 3),
        ("π⁺", "K⁻"): (2, 3),
    }
    for (a, b), subsystem in pair_to_subsystem.items():
        # Fetch resonances for this pair from particles module
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
    return res_list


def main(resolution=500, smooth=False, show_plots=False):
    """
    Generate Dalitz plot for reference decay Λc⁺ → p π⁺ K⁻.
    
    Args:
        resolution: Grid resolution (n_points × n_points)
        smooth: If True, apply smoothing to the plot
        show_plots: If True, display plot interactively
    """
    print("=" * 70)
    print("Dalitz Plot Generator for Λc⁺ → p π⁺ K⁻ Decay")
    print("Homework 7: Flavor SU(3) Symmetry")
    print("=" * 70)
    
    # Create output directory if needed
    os.makedirs('pictures/dalitz', exist_ok=True)
    
    # Build resonances via particles module (SU(3/PDG-mapped couplings)
    resonances_dynamic = build_resonances_from_particles()
    
    # Create generator with corrected kinematic boundaries
    # Use x-axis = m_{pK}^2 (pair (1,3)) and y-axis = m_{πK}^2 (pair (2,3))
    generator = DalitzPlotGenerator(mother_particle, daughter_particles, resonances_dynamic,
                                    pair_x=(1, 3), pair_y=(2, 3))
    
    # Generate and plot with specified parameters
    generator.plot_dalitz(n_points=resolution, save_path="pictures/dalitz/dalitz_plot_lambda_c_to_p_pi_k.png",
                         smooth=smooth, show=show_plots)
    
    print("\n" + "=" * 70)
    print("Dalitz plot generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()