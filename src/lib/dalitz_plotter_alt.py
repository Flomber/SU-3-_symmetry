"""
Dalitz plot for alternative SU(3)-related decay: Λc⁺ → n π⁺ K̄⁰
Demonstrates how SU(3) partners show similar resonance structure but differ in:
- Kinematic phase space (different daughter masses)
- Resonance couplings (via SU(3) mapping)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
from src.lib.particles import get_resonances_for_pair, BARYONS, MESONS


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
        self.mother = mother
        self.daughters = daughters
        self.resonances = resonances
        self.pair_x = tuple(sorted(pair_x))
        self.pair_y = tuple(sorted(pair_y))
        all_pairs = {(1, 2), (2, 3), (1, 3)}
        self.pair_z = list(all_pairs - {self.pair_x, self.pair_y})[0]
    
    @staticmethod
    def kallen_function(a: float, b: float, c: float) -> float:
        """Calculate the Källén function λ(a, b, c)."""
        return a**2 + b**2 + c**2 - 2*(a*b + b*c + c*a)
    
    def breit_wigner(self, m_ij_sq: float, resonance: Resonance) -> complex:
        """Calculate the Breit-Wigner amplitude for a resonance."""
        m_R = resonance.mass
        gamma_R = resonance.width
        denominator = m_R**2 - m_ij_sq - 1j * m_R * gamma_R
        return 1.0 / denominator
    
    def calculate_third_invariant(self, mX_sq: float, mY_sq: float) -> float:
        """Calculate the third invariant mass squared."""
        m0_sq = self.mother.mass**2
        m1_sq = self.daughters[0].mass**2
        m2_sq = self.daughters[1].mass**2
        m3_sq = self.daughters[2].mass**2
        return m0_sq + m1_sq + m2_sq + m3_sq - mX_sq - mY_sq
    
    def is_kinematically_allowed(self, mX_sq: float, mY_sq: float) -> bool:
        """Check if a point in the Dalitz plot is kinematically allowed."""
        m0_sq = self.mother.mass**2
        m1_sq = self.daughters[0].mass**2
        m2_sq = self.daughters[1].mass**2
        m3_sq = self.daughters[2].mass**2
        
        mZ_sq = self.calculate_third_invariant(mX_sq, mY_sq)
        
        if mX_sq <= 0 or mY_sq <= 0 or mZ_sq <= 0:
            return False
        
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
        
        if lambda_x < 0 or lambda_y < 0 or lambda_z < 0:
            return False
        
        phi = self.kallen_function(lambda_x, lambda_y, lambda_z)
        return phi < 1e-10
    
    def matrix_element(self, mX_sq: float, mY_sq: float) -> complex:
        """Calculate the decay matrix element."""
        mZ_sq = self.calculate_third_invariant(mX_sq, mY_sq)
        
        inv_map = {}
        inv_map[tuple(sorted(self.pair_x))] = mX_sq
        inv_map[tuple(sorted(self.pair_y))] = mY_sq
        inv_map[self.pair_z] = mZ_sq
        
        amplitude = 0.0 + 0.0j
        
        for resonance in self.resonances:
            m_ij_sq = inv_map[tuple(sorted(resonance.subsystem))]
            bw_amplitude = self.breit_wigner(m_ij_sq, resonance)
            amplitude += resonance.g_prod * bw_amplitude * resonance.g_dec
        
        return amplitude
    
    def generate_dalitz_plot(self, n_points: int = 250, smooth: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the Dalitz plot."""
        print("Generating Dalitz plot with corrected kinematic boundaries...")
        
        m0 = self.mother.mass
        m1, m2, m3 = [d.mass for d in self.daughters]
        
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
        
        mX_sq_vals = np.linspace(mX_sq_min_threshold, mX_sq_max_threshold, n_points)
        mY_sq_vals = np.linspace(mY_sq_min_threshold, mY_sq_max_threshold, n_points)
        m12_sq_grid, m23_sq_grid = np.meshgrid(mX_sq_vals, mY_sq_vals)
        
        intensity_grid = np.zeros_like(m12_sq_grid)
        n_physical = 0
        
        print(f"Checking {n_points}×{n_points} = {n_points**2} grid points...")
        
        for i in range(n_points):
            if i % (n_points // 10) == 0:
                print(f"  Progress: {100*i/n_points:.0f}%")
            
            for j in range(n_points):
                mX_sq = m12_sq_grid[i, j]
                mY_sq = m23_sq_grid[i, j]
                
                if self.is_kinematically_allowed(mX_sq, mY_sq):
                    amplitude = self.matrix_element(mX_sq, mY_sq)
                    intensity = abs(amplitude)**2
                    intensity_grid[i, j] = intensity
                    n_physical += 1
                else:
                    intensity_grid[i, j] = np.nan
        
        print(f"✓ Physical region: {n_physical} / {n_points**2} points ({100*n_physical/n_points**2:.1f}%)")
        
        if smooth:
            print(f"✓ Applying spline interpolation and smoothing...")
            m12_sq_grid, m23_sq_grid, intensity_grid = self._smooth_intensity_grid(
                m12_sq_grid, m23_sq_grid, intensity_grid, upsampling_factor=3
            )
        
        return m12_sq_grid, m23_sq_grid, intensity_grid
    
    def _smooth_intensity_grid(self, m12_sq_grid: np.ndarray, m23_sq_grid: np.ndarray, 
                              intensity_grid: np.ndarray, upsampling_factor: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Smooth the intensity grid using spline interpolation."""
        n_rows, n_cols = intensity_grid.shape
        m12_sq_1d = m12_sq_grid[0, :]
        m23_sq_1d = m23_sq_grid[:, 0]
        
        m12_sq_fine = np.linspace(m12_sq_1d.min(), m12_sq_1d.max(), n_cols * upsampling_factor)
        m23_sq_fine = np.linspace(m23_sq_1d.min(), m23_sq_1d.max(), n_rows * upsampling_factor)
        m12_sq_upsampled, m23_sq_upsampled = np.meshgrid(m12_sq_fine, m23_sq_fine)
        
        intensity_no_nan = intensity_grid.copy()
        intensity_no_nan[np.isnan(intensity_no_nan)] = 0
        
        spline = RectBivariateSpline(m23_sq_1d, m12_sq_1d, intensity_no_nan, kx=3, ky=3)
        intensity_upsampled = spline(m23_sq_fine, m12_sq_fine)
        intensity_upsampled = gaussian_filter(intensity_upsampled, sigma=2.0)
        
        valid_mask = ~np.isnan(intensity_grid)
        for i in range(len(m23_sq_fine)):
            for j in range(len(m12_sq_fine)):
                i_orig = int(np.round(i / upsampling_factor))
                j_orig = int(np.round(j / upsampling_factor))
                i_orig = np.clip(i_orig, 0, n_rows - 1)
                j_orig = np.clip(j_orig, 0, n_cols - 1)
                if not valid_mask[i_orig, j_orig]:
                    intensity_upsampled[i, j] = np.nan
        
        return m12_sq_upsampled, m23_sq_upsampled, intensity_upsampled
    
    def plot_dalitz(self, n_points: int = 250, save_path: Optional[str] = None):
        """Generate and display the Dalitz plot."""
        m12_sq_grid, m23_sq_grid, intensity_grid = self.generate_dalitz_plot(n_points)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        intensity_masked = np.ma.masked_invalid(intensity_grid)
        vmax = np.nanmax(intensity_grid)
        vmin = vmax * 1e-3
        
        im = ax.pcolormesh(m12_sq_grid, m23_sq_grid, intensity_masked, 
                          shading='auto', cmap='hot',
                          norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        
        cbar = plt.colorbar(im, ax=ax, label='|M|² (Intensity)', pad=0.02)
        
        name_map = {1: self.daughters[0].name, 2: self.daughters[1].name, 3: self.daughters[2].name}
        x_label = f"$m_{{{name_map[self.pair_x[0]]}{name_map[self.pair_x[1]]}}}^2$ (GeV²)"
        y_label = f"$m_{{{name_map[self.pair_y[0]]}{name_map[self.pair_y[1]]}}}^2$ (GeV²)"
        
        for resonance in self.resonances:
            mass_sq = resonance.mass**2
            if tuple(sorted(resonance.subsystem)) == self.pair_x:
                ax.axvline(mass_sq, color='cyan', linestyle='--', linewidth=2.5, alpha=0.8, label=resonance.name)
            elif tuple(sorted(resonance.subsystem)) == self.pair_y:
                ax.axhline(mass_sq, color='lime', linestyle='--', linewidth=2.5, alpha=0.8, label=resonance.name)
        
        daughters_str = " ".join([d.name for d in self.daughters])
        ax.set_title(f'Dalitz Plot: {self.mother.name} → {daughters_str}\n(with corrected kinematic boundaries)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            print(f"\nSaving plot to: {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# ============================================================================
# Alternative decay: Λc⁺ → n π⁺ K̄⁰
# ============================================================================

# Mother particle (Λc⁺)
mother_particle = Particle(name="Λc⁺", mass=2.2865)

# Daughter particles for Λc⁺ → n π⁺ K̄⁰
daughter_particles = [
    Particle(name="n", mass=0.93957),    # neutron (index 1)
    Particle(name="π⁺", mass=0.13957),   # pion (index 2)
    Particle(name="K̄⁰", mass=0.49761)    # anti-kaon (index 3)
]


def build_resonances_from_particles() -> List[Resonance]:
    """Build resonances for n π⁺ K̄⁰ from particles module."""
    res_list: List[Resonance] = []
    pair_to_subsystem = {
        ("n", "π⁺"): (1, 2),
        ("n", "K̄⁰"): (1, 3),
        ("π⁺", "K̄⁰"): (2, 3),
    }
    for (a, b), subsystem in pair_to_subsystem.items():
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


def main():
    print("=" * 70)
    print("Dalitz Plot Generator for Λc⁺ → n π⁺ K̄⁰ Decay")
    print("Alternative SU(3)-related decay to reference Λc⁺ → p π⁺ K⁻")
    print("=" * 70)
    
    os.makedirs('pictures', exist_ok=True)
    
    resonances_dynamic = build_resonances_from_particles()
    
    # Use x-axis = m_{nK̄}^2 (pair (1,3)) and y-axis = m_{πK̄}^2 (pair (2,3))
    generator = DalitzPlotGenerator(mother_particle, daughter_particles, resonances_dynamic,
                                    pair_x=(1, 3), pair_y=(2, 3))
    
    generator.plot_dalitz(n_points=250, save_path="pictures/dalitz_plot_lambda_c_to_n_pi_kbar.png")
    
    print("\n" + "=" * 70)
    print("Dalitz plot generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
