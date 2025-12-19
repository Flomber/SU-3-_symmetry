"""
SU(3) Multiplet Plotter for Λc⁺ → p π⁺ K⁻ Decay
Homework 7: Flavor SU(3) Symmetry

Plots the following multiplets:
1. Baryon Octet (1/2⁺): Contains p in final state
2. Meson Octet (0⁻): Contains π⁺ and K⁻ in final state
3. Baryon Decuplet (3/2⁺): Contains Δ⁺(1232) resonance
4. Vector Meson Nonet (1⁻): Contains K*(892) resonance
5. Excited Baryon (3/2⁻): Contains Λ*(1520) resonance
6. Charmed Baryon Sextet: Contains Λc⁺ initial state
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from typing import Tuple, List
import os
from src.lib.particles import get_particle, BARYONS, MESONS


class MultipletPlotter:
    """Handles creation and visualization of SU(3) flavor multiplets."""
    
    def __init__(self, output_dir: str = 'pictures/multiplets'):
        """
        Initialize the multiplet plotter.
        
        Args:
            output_dir: Directory to save plots (default: 'pictures' subdirectory)
        """
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.dpi = 300
        self.figsize = (12, 10)
        
    def _draw_particle(self, ax, x: float, y: float, name: str,
                      color: str = 'lightblue', size: float = 0.3,
                      quark_content: str = '', highlight: bool = False,
                      linewidth: float = 2.0,
                      name_offset: Tuple[float, float] = (0.0, 0.05),
                      quark_offset: Tuple[float, float] = (0.0, -0.15)):
        """
        Draw a particle as a circle with label.
        
        Args:
            ax: Matplotlib axes
            x, y: Position
            name: Particle name
            color: Circle color
            size: Circle radius
            quark_content: Quark content label
            highlight: Whether to highlight the particle
            linewidth: Border line width
            name_offset: (dx, dy) offset for the name label
            quark_offset: (dx, dy) offset for the quark-content label
        """
        edge_color = 'darkred' if highlight else 'black'
        edge_width = 3.5 if highlight else linewidth
        
        circle = Circle((x, y), size, color=color, ec=edge_color, 
                       linewidth=edge_width, zorder=3)
        ax.add_patch(circle)
        
        # Add particle name
        ax.text(x + name_offset[0], y + name_offset[1], name, ha='center', va='center',
               fontsize=12, fontweight='bold', zorder=4)
        
        # Add quark content below particle name if provided
        if quark_content:
            ax.text(x + quark_offset[0], y + quark_offset[1], quark_content, ha='center', va='top',
                   fontsize=8, style='italic', color='darkblue', zorder=4)
    
    def plot_baryon_octet(self):
        """
        Plot the baryon octet (J^P = 1/2⁺).
        Contains: p (proton) - final state particle
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Baryon octet members to plot (name, highlight)
        # Exclude Λ and Σ⁰ from this list - they'll be drawn separately at center
        octet_members = [
            ('p', True),   # Final state - HIGHLIGHTED
            ('n', False),
            ('Σ⁺', False),
            ('Σ⁻', False),
            ('Ξ⁰', False),
            ('Ξ⁻', False),
        ]
        
        # Draw Λ and Σ⁰ at center (I₃=0, S=-1) with overlapping circles
        lambda_particle = get_particle('Λ')
        sigma0_particle = get_particle('Σ⁰')
        # Draw Λ (larger circle, I=0)
        self._draw_particle(ax, 0, -1, lambda_particle.name, color='lightskyblue',
                   quark_content=lambda_particle.quarks, highlight=False, size=0.4,
                   name_offset=(0, 0.08), quark_offset=(0, 0.0))
        # Draw Σ⁰ (smaller circle, I=1)
        self._draw_particle(ax, 0, -1, sigma0_particle.name, color='lightblue',
                   quark_content=sigma0_particle.quarks, highlight=False, size=0.24,
                   name_offset=(0, -0.12), quark_offset=(0, -0.26))
        
        # Draw remaining particles using data from particles module
        for name, highlight in octet_members:
            particle = get_particle(name)
            color = 'red' if highlight else 'lightblue'
            self._draw_particle(ax, particle.isospin3, particle.strangeness, 
                              particle.name, color=color, 
                              quark_content=particle.quarks, highlight=highlight)
        
        # Draw connection lines (hexagon + center)
        hex_x = [0.5, 1, 0.5, -0.5, -1, -0.5, 0.5]
        hex_y = [0, -1, -2, -2, -1, 0, 0]
        ax.plot(hex_x, hex_y, 'b--', alpha=0.3, linewidth=1.5)
        
        # Central lines
        ax.plot([0, -0.5], [0, 0], 'b--', alpha=0.3, linewidth=1.5)
        ax.plot([0, 0.5], [0, 0], 'b--', alpha=0.3, linewidth=1.5)
        ax.plot([0, 0], [0, -1], 'b--', alpha=0.3, linewidth=1.5)
        
        # Axes
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Labels
        ax.set_xlabel('Isospin I₃', fontsize=14, fontweight='bold')
        ax.set_ylabel('Strangeness S', fontsize=14, fontweight='bold')
        ax.set_title('Baryon Octet (J^P = 1/2⁺)\np - Final State Particle', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-2.5, 0.5)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        
        # Legend
        red_patch = mpatches.Patch(color='red', label='p (Proton) - Final State')
        blue_patch = mpatches.Patch(color='lightblue', label='Other Octet Members')
        ax.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=11)
        
        # Info note about center
        info_text = (
            'Center (I₃=0, S=-1):\n'
            'Λ (I=0) and Σ⁰ (I=1, I₃=0)\n'
            'shown as overlapping circles'
        )
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6))
        
        plt.tight_layout()
        return fig
    
    def plot_meson_octet(self):
        """
        Plot the pseudoscalar meson nonet (J^P = 0⁻): octet ⊕ singlet (9 states).
        Left: η' singlet. Right: octet with η/π⁰ concentric at center.
        Contains: π⁺ and K⁻ - final state particles.
        """
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = fig.add_gridspec(1, 2, wspace=0.075, width_ratios=[1, 1])
        ax_left = fig.add_subplot(gs[0, 0])   # singlet η'
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)  # octet

        # Singlet panel: η'
        eta_prime = get_particle("η'")
        self._draw_particle(ax_left, 0, 0, eta_prime.name, color='burlywood',
                           quark_content=eta_prime.quarks, highlight=False, size=0.5)
        ax_left.set_xlabel('Isospin I₃', fontsize=12, fontweight='bold')
        ax_left.set_ylabel('Strangeness S', fontsize=12, fontweight='bold')
        ax_left.set_title("η' Singlet (I=0)", fontsize=14, fontweight='bold')
        ax_left.set_xlim(-1.5, 1.5)
        ax_left.set_ylim(-1.5, 1.5)
        ax_left.grid(True, alpha=0.2)
        ax_left.set_aspect('equal')
        ax_left.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax_left.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Octet panel
        # Meson members to plot (name, highlight, size)
        meson_members = [
            ('π⁺', True, 0.3),   # Final state - HIGHLIGHTED
            ('π⁻', False, 0.3),
            ('K⁺', False, 0.3),
            ('K⁰', False, 0.3),
            ('K̄⁰', False, 0.3),
            ('K⁻', True, 0.3),   # Final state - HIGHLIGHTED
        ]

        # η (octet I=0) and π⁰ (I=1, I3=0) at center with separated labels
        eta = get_particle('η')
        pi0 = get_particle('π⁰')
        self._draw_particle(ax_right, 0, 0, eta.name, color='wheat',
                   quark_content=eta.quarks, highlight=False, size=0.4,
                   name_offset=(0, 0.08), quark_offset=(0, 0.0))
        self._draw_particle(ax_right, 0, 0, pi0.name, color='lightyellow',
                   quark_content=pi0.quarks, highlight=False, size=0.24,
                   name_offset=(0, -0.1), quark_offset=(0, -0.24))

        for name, highlight, size in meson_members:
            particle = get_particle(name)
            color = 'green' if highlight else 'lightyellow'
            self._draw_particle(ax_right, particle.isospin3, particle.strangeness, 
                               particle.name, color=color,
                               quark_content=particle.quarks, highlight=highlight, size=size)

        # Structure lines for octet
        hex_x = [0.5, 1, 0.5, -0.5, -1, -0.5, 0.5]
        hex_y = [1, 0, -1, -1, 0, 1, 1]
        ax_right.plot(hex_x, hex_y, 'orange', linestyle='--', alpha=0.3, linewidth=1.5)
        ax_right.plot([0, -1], [0, 0], 'orange', linestyle='--', alpha=0.3, linewidth=1.5)
        ax_right.plot([0, 1], [0, 0], 'orange', linestyle='--', alpha=0.3, linewidth=1.5)

        ax_right.set_xlabel('Isospin I₃', fontsize=12, fontweight='bold')
        ax_right.tick_params(labelleft=False)
        ax_right.set_title('Octet (I=1 triplet, I=0 octet member)', fontsize=14, fontweight='bold')
        ax_right.set_xlim(-1.5, 1.5)
        ax_right.grid(True, alpha=0.2)
        ax_right.set_aspect('equal')
        ax_right.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax_right.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        fig.suptitle('Pseudoscalar Meson Nonet (J^P = 0⁻)', fontsize=16, fontweight='bold', y=0.83)

        # Legends
        left_leg = [mpatches.Patch(color='burlywood', label="η' (singlet, I=0)")]
        ax_left.legend(handles=left_leg, loc='upper right', fontsize=9)

        green_patch = mpatches.Patch(color='green', label='π⁺, K⁻ - Final States')
        yellow_patch = mpatches.Patch(color='lightyellow', label='π⁰, π⁻ - Isospin Triplet')
        wheat_patch = mpatches.Patch(color='wheat', label='η (octet, I=0)')
        ax_right.legend(handles=[green_patch, yellow_patch, wheat_patch], loc='upper right', fontsize=9)

        # Info note about center (octet panel)
        info_text = (
            'Center (I₃=0, S=0):\n'
            'η octet (wheat) and π⁰ (yellow)\n'
            '(η' ' prime is the separate singlet panel)'
        )
        ax_right.text(0.02, 0.02, info_text, transform=ax_right.transAxes,
                      fontsize=8, verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6))

        return fig
    
    def plot_baryon_decuplet(self):
        """
        Plot the baryon decuplet (J^P = 3/2⁺).
        Highlights Δ⁰(1232) (neutral state) for generality
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Baryon decuplet members to plot (name, highlight)
        decuplet_members = [
            # Δ resonances (S=0)
            ('Δ⁺⁺', False),
            ('Δ⁺', False),
            ('Δ⁰', True),   # Neutral Δ highlighted
            ('Δ⁻', False),
            # Σ* resonances (S=-1)
            ('Σ*⁺', False),
            ('Σ*⁰', False),
            ('Σ*⁻', False),
            # Ξ* resonances (S=-2)
            ('Ξ*⁰', False),
            ('Ξ*⁻', False),
            # Ω resonance (S=-3)
            ('Ω⁻', False),
        ]
        
        # Draw particles using data from particles module
        for name, highlight in decuplet_members:
            particle = get_particle(name)
            color = 'red' if highlight else 'lightcyan'
            self._draw_particle(ax, particle.isospin3, particle.strangeness, 
                              particle.name, color=color,
                              quark_content=particle.quarks, highlight=highlight)
        
        # Draw connection lines (triangular structure)
        lines = [
            # Row 1 (S=0)
            ([-1.5, 1.5], [0, 0]),
            # Row 2 (S=-1)
            ([-1, 1], [-1, -1]),
            # Row 3 (S=-2)
            ([-0.5, 0.5], [-2, -2]),
            # Diagonals
            ([-1.5, -1], [0, -1]),
            ([-1, -0.5], [-1, -2]),
            ([-0.5, 0], [-2, -3]),
            ([1.5, 1], [0, -1]),
            ([1, 0.5], [-1, -2]),
            ([0.5, 0], [-2, -3]),
        ]
        
        for x, y in lines:
            ax.plot(x, y, 'purple', linestyle='--', alpha=0.3, linewidth=1.5)
        
        # Axes
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Labels
        ax.set_xlabel('Isospin I₃', fontsize=14, fontweight='bold')
        ax.set_ylabel('Strangeness S', fontsize=14, fontweight='bold')
        ax.set_title('Baryon Decuplet (J^P = 3/2⁺)\nΔ⁰(1232) - Neutral State Highlighted', 
                fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-3.5, 0.8)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        
        # Legend
        red_patch = mpatches.Patch(color='red', label='Δ⁰(1232) - Highlighted')
        cyan_patch = mpatches.Patch(color='lightcyan', label='Other Decuplet Members')
        ax.legend(handles=[red_patch, cyan_patch], loc='upper right', fontsize=11)
        
        # Info box
        info_text = (
            'Decuplet Structure:\n'
            '• 4 particles at S=0 (Δ)\n'
            '• 3 particles at S=-1 (Σ*)\n'
            '• 2 particles at S=-2 (Ξ*)\n'
            '• 1 particle at S=-3 (Ω⁻)'
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_vector_meson_nonet(self):
        """
        Plot the vector meson nonet (J^P = 1⁻) with singlet separated.
        Left: singlet φ (mostly ss̄). Right: octet with ρ, K*, and octet ω.
        """
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = fig.add_gridspec(1, 2, wspace=0.075, width_ratios=[1, 1])
        ax_left = fig.add_subplot(gs[0, 0])   # singlet φ
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)  # octet

        # Singlet panel: φ (dominantly singlet)
        phi = get_particle('φ')
        self._draw_particle(ax_left, 0, 0, phi.name, color='burlywood',
                           quark_content=f'{phi.quarks} (singlet-dominated)', highlight=False, size=0.5)
        ax_left.set_xlabel('Isospin I₃', fontsize=12, fontweight='bold')
        ax_left.set_ylabel('Strangeness S', fontsize=12, fontweight='bold')
        ax_left.set_title('φ Singlet (I=0)', fontsize=14, fontweight='bold')
        ax_left.set_xlim(-1.5, 1.5)
        ax_left.set_ylim(-1.5, 1.5)
        ax_left.grid(True, alpha=0.2)
        ax_left.set_aspect('equal')
        ax_left.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax_left.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Octet panel
        # Vector meson members to plot (name, highlight, size, name_offset, quark_offset)
        vector_members = [
            # ρ mesons (I=1, S=0)
            ('ρ⁺', False, 0.32, (0, 0.06), (0, -0.1)),
            ('ρ⁻', False, 0.32, (0, 0.06), (0, -0.1)),
            # K* mesons (I=1/2, S=±1)
            ('K*⁺', False, 0.3, (0, 0.05), (0, -0.15)),
            ('K*⁰', False, 0.3, (0, 0.05), (0, -0.15)),
            ('K̄*⁰', False, 0.3, (0, 0.05), (0, -0.15)),
            ('K*⁻', True, 0.32, (0, 0.07), (0, -0.13)),
        ]

        # Center states (draw larger first, then smaller on top)
        omega = get_particle('ω')
        rho0 = get_particle('ρ⁰')
        self._draw_particle(ax_right, 0, 0, omega.name, color='lightcoral',
                   quark_content=omega.quarks, highlight=False, size=0.38,
                   name_offset=(0, 0.1), quark_offset=(0, 0.0))
        self._draw_particle(ax_right, 0, 0, rho0.name, color='lightpink',
                           quark_content=rho0.quarks, highlight=False, size=0.26,
                           name_offset=(0, -0.12), quark_offset=(0, -0.26))

        # Draw octet particles
        for name, highlight, size, name_off, quark_off in vector_members:
            particle = get_particle(name)
            color = 'cyan' if highlight else 'lightcoral'
            self._draw_particle(ax_right, particle.isospin3, particle.strangeness, 
                               particle.name, color=color,
                               quark_content=particle.quarks, highlight=highlight, size=size,
                               name_offset=name_off, quark_offset=quark_off)

        # Structure lines for octet
        hex_x = [0.5, 1, 0.5, -0.5, -1, -0.5, 0.5]
        hex_y = [-1, 0, 1, 1, 0, -1, -1]
        ax_right.plot(hex_x, hex_y, 'red', linestyle='--', alpha=0.3, linewidth=1.5)
        ax_right.plot([0, -0.5], [0, -1], 'red', linestyle='--', alpha=0.3, linewidth=1.5)
        ax_right.plot([0, 0.5], [0, -1], 'red', linestyle='--', alpha=0.3, linewidth=1.5)
        ax_right.plot([0, -0.5], [0, 1], 'red', linestyle='--', alpha=0.3, linewidth=1.5)
        ax_right.plot([0, 0.5], [0, 1], 'red', linestyle='--', alpha=0.3, linewidth=1.5)

        ax_right.set_xlabel('Isospin I₃', fontsize=12, fontweight='bold')
        ax_right.tick_params(labelleft=False)
        ax_right.set_title('Octet (ρ, K*, ω_octet)', fontsize=14, fontweight='bold')
        ax_right.set_xlim(-1.5, 1.5)
        ax_right.grid(True, alpha=0.2)
        ax_right.set_aspect('equal')
        ax_right.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax_right.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        fig.suptitle('Vector Meson Nonet (J^P = 1⁻)', fontsize=16, fontweight='bold', y=0.8)

        # Legends
        left_leg = [mpatches.Patch(color='burlywood', label='φ (singlet, I=0)')]
        ax_left.legend(handles=left_leg, loc='upper right', fontsize=9)

        cyan_patch = mpatches.Patch(color='cyan', label='K*⁻(892) - Resonance')
        red_patch = mpatches.Patch(color='lightcoral', label='Other Octet Members')
        ax_right.legend(handles=[cyan_patch, red_patch], loc='upper right', fontsize=9)

        # Info note
        info_text = (
            'Octet center: ω (octet) and ρ⁰ at (0,0)\n'
            'Singlet: φ shown in separate panel (left).\n'
            'K*⁻ highlighted as resonance.'
        )
        ax_right.text(0.02, 0.02, info_text, transform=ax_right.transAxes,
                      fontsize=8, verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

        return fig
    
    def plot_excited_baryon_resonance(self):
        """
        Plot the negative-parity baryon octet (J^P = 3/2⁻).
        Separates Λ*(1520) singlet from the rest of the octet.
        """
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = fig.add_gridspec(1, 2, wspace=0.075, width_ratios=[1, 1])
        ax_left = fig.add_subplot(gs[0, 0])   # Λ* singlet
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)  # Rest of octet
        
        # Λ*(1520) isospin singlet (I=0, S=-1)
        lambda_singlet = ['Λ*(1520)']
        
        # Rest of the octet (I≠0 states)
        octet_rest = [
            'N*(1520)⁺', 'N*(1520)⁰',
            'Σ*(1670)⁺', 'Σ*(1670)⁰', 'Σ*(1670)⁻',
            'Ξ*(1690)⁰', 'Ξ*(1690)⁻',
        ]
        
        # Draw Λ* singlet panel
        for name in lambda_singlet:
            particle = get_particle(name)
            self._draw_particle(ax_left, particle.isospin3, particle.strangeness, 
                               particle.name, color='orange',
                               quark_content=particle.quarks, highlight=True, size=0.35)
        ax_left.set_xlabel('Isospin I₃', fontsize=12, fontweight='bold')
        ax_left.set_ylabel('Strangeness S', fontsize=12, fontweight='bold')
        ax_left.set_title('Λ*(1520) Singlet\n(I=0)', fontsize=14, fontweight='bold')
        ax_left.set_xlim(-1.5, 1.5)
        ax_left.set_ylim(-2.5, 0.5)
        ax_left.grid(True, alpha=0.2)
        ax_left.set_aspect('equal')
        ax_left.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax_left.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Draw rest of octet panel
        for name in octet_rest:
            particle = get_particle(name)
            self._draw_particle(ax_right, particle.isospin3, particle.strangeness, 
                               particle.name, color='lavender',
                               quark_content=particle.quarks, highlight=False, size=0.35)
        
        # Draw connection lines (hexagon + center) for rest of octet
        hex_x = [0.5, 1, 0.5, -0.5, -1, -0.5, 0.5]
        hex_y = [0, -1, -2, -2, -1, 0, 0]
        ax_right.plot(hex_x, hex_y, color='purple', linestyle='--', alpha=0.3, linewidth=1.5)
        
        # Central lines
        ax_right.plot([0, -0.5], [0, 0], color='purple', linestyle='--', alpha=0.3, linewidth=1.5)
        ax_right.plot([0, 0.5], [0, 0], color='purple', linestyle='--', alpha=0.3, linewidth=1.5)
        ax_right.plot([0, 0], [-1, 0], color='purple', linestyle='--', alpha=0.3, linewidth=1.5)
        
        ax_right.set_xlabel('Isospin I₃', fontsize=12, fontweight='bold')
        ax_right.tick_params(labelleft=False)
        ax_right.set_title('Rest of Octet\n(N*, Σ*, Ξ*)', fontsize=14, fontweight='bold')
        ax_right.set_xlim(-1.5, 1.5)
        ax_right.grid(True, alpha=0.2)
        ax_right.set_aspect('equal')
        ax_right.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax_right.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Super title and info
        fig.suptitle('Negative-Parity Baryon Octet (J^P = 3/2⁻)', fontsize=16, fontweight='bold', y=0.85)
        ax_left.text(0.02, 0.98,
                     'Λ*(1520) Properties:\n• Mass: 1.518 GeV\n• Width: 0.015 GeV\n• Resonance in pK⁻',
                     transform=ax_left.transAxes, fontsize=8, va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))
        
        return fig
    
    def plot_charmed_baryons(self):
        """
        Charmed baryons (one c quark): render anti-triplet (3) and sextet (6)
        in separate side-by-side panels to avoid overlap while retaining
        true SU(3) (I₃_light, S_light) coordinates.
        """
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = fig.add_gridspec(1, 2, wspace=0.075, width_ratios=[1, 1])
        ax_left = fig.add_subplot(gs[0, 0])   # anti-triplet
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)  # sextet, share y-axis

        # Anti-triplet (light diquark spin-0): Λc⁺, Ξc⁺, Ξc⁰
        anti_triplet = ['Λc⁺', 'Ξc⁺', 'Ξc⁰']

        # Sextet (light diquark spin-1): Σc, Ξc′, Ωc
        sextet = ['Σc⁺⁺', 'Σc⁺', 'Σc⁰', 'Ξc\'⁺', 'Ξc\'⁰', 'Ωc⁰']

        # Draw anti-triplet panel
        for name in anti_triplet:
            particle = get_particle(name)
            # Use light-quark quantum numbers (total I3 and S minus charm contribution)
            # For charmed baryons, these are the displayed coordinates
            color = 'purple' if name == 'Λc⁺' else 'plum'
            self._draw_particle(ax_left, particle.isospin3, particle.strangeness, 
                                particle.name, color=color,
                                quark_content=particle.quarks, 
                                highlight=(name == 'Λc⁺'), size=0.35)
        ax_left.set_xlabel('Isospin I₃ (light)', fontsize=12, fontweight='bold')
        ax_left.set_ylabel('Strangeness S (light)', fontsize=12, fontweight='bold')
        ax_left.set_title('Anti-triplet (3)', fontsize=14, fontweight='bold')
        ax_left.set_xlim(-1.5, 1.5)
        ax_left.set_ylim(-2.5, 0.8)
        ax_left.grid(True, alpha=0.2)
        ax_left.set_aspect('equal')

        # Legend for anti-triplet
        left_legend = [
            mpatches.Patch(color='purple', label='Λc⁺ (highlighted)'),
            mpatches.Patch(color='plum', label='Ξc⁺, Ξc⁰'),
        ]
        ax_left.legend(handles=left_legend, loc='upper right', fontsize=9)

        # Draw sextet panel
        for name in sextet:
            particle = get_particle(name)
            self._draw_particle(ax_right, particle.isospin3, particle.strangeness, 
                                particle.name, color='lightskyblue',
                                quark_content=particle.quarks, highlight=False, size=0.35)
        # Sextet triangular structure lines
        lines = [
            ([-1, 1], [0, 0]),
            ([-0.5, 0.5], [-1, -1]),
            ([-1, -0.5], [0, -1]),
            ([1, 0.5], [0, -1]),
            ([0.5, 0], [-1, -2]),
            ([-0.5, 0], [-1, -2]),
        ]
        for x, y in lines:
            ax_right.plot(x, y, color='steelblue', linestyle='--', alpha=0.35, linewidth=1.5)
        ax_right.set_xlabel('Isospin I₃ (light)', fontsize=12, fontweight='bold')
        # Hide duplicate y-axis labels on the right subplot
        ax_right.tick_params(labelleft=False)
        ax_right.set_title('Sextet (6)', fontsize=14, fontweight='bold')
        ax_right.set_xlim(-1.5, 1.5)
        ax_right.grid(True, alpha=0.2)
        ax_right.set_aspect('equal')

        # Legend for sextet
        right_legend = [
            mpatches.Patch(color='lightskyblue', label='Σc, Ξc′, Ωc'),
        ]
        ax_right.legend(handles=right_legend, loc='upper right', fontsize=9)

        # Super title and info box
        fig.suptitle('Charmed Baryons (one c quark) – Separate Panels', fontsize=16, fontweight='bold', y=0.85)
        ax_left.text(0.02, 0.98,
                     'Light-flavor quantum numbers shown.\nSU(3)_flavor ladders (I, U, V) act on u,d,s only.',
                     transform=ax_left.transAxes, fontsize=8, va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))

        return fig
    
    def plot_decay_process_summary(self):
        """
        Create a comprehensive summary showing the decay Λc⁺ → p π⁺ K⁻
        with all involved particles and their multiplet assignments.
        """
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('SU(3) Flavor Multiplets in Λc⁺ → p π⁺ K⁻ Decay\nTask 1: Multiplet Assignment',
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Initial state
        ax_initial = fig.add_subplot(gs[0, :])
        ax_initial.axis('off')
        
        # Draw decay diagram
        ax_initial.text(0.1, 0.7, 'Initial State:', fontsize=13, fontweight='bold')
        ax_initial.add_patch(Rectangle((0.15, 0.5), 0.1, 0.15, 
                                      facecolor='purple', edgecolor='darkred', linewidth=3))
        ax_initial.text(0.2, 0.575, 'Λc⁺', fontsize=12, fontweight='bold', 
                       ha='center', va='center', color='white')
        ax_initial.text(0.3, 0.575, '(cud)\nCharmed Baryon Sextet\nJ^P = 1/2⁺',
                       fontsize=11, va='center')
        
        ax_initial.arrow(0.5, 0.575, 0.15, 0, head_width=0.05, head_length=0.03,
                        fc='black', ec='black', linewidth=2)
        ax_initial.text(0.57, 0.65, 'decay', fontsize=11, ha='center', style='italic')
        
        # Final states
        ax_initial.text(0.7, 0.7, 'Final State:', fontsize=13, fontweight='bold')
        
        # Proton
        ax_initial.add_patch(Rectangle((0.7, 0.5), 0.08, 0.15,
                                      facecolor='red', edgecolor='darkred', linewidth=2))
        ax_initial.text(0.74, 0.575, 'p', fontsize=12, fontweight='bold',
                       ha='center', va='center', color='white')
        ax_initial.text(0.82, 0.575, '(uud)\nBaryon Octet',
                       fontsize=9, va='center')
        
        # Pion
        ax_initial.add_patch(Rectangle((0.7, 0.25), 0.08, 0.15,
                                      facecolor='green', edgecolor='darkgreen', linewidth=2))
        ax_initial.text(0.74, 0.325, 'π⁺', fontsize=12, fontweight='bold',
                       ha='center', va='center', color='white')
        ax_initial.text(0.82, 0.325, '(ud̄)\nMeson Octet',
                       fontsize=9, va='center')
        
        # Kaon
        ax_initial.add_patch(Rectangle((0.7, 0.0), 0.08, 0.15,
                                      facecolor='green', edgecolor='darkgreen', linewidth=2))
        ax_initial.text(0.74, 0.075, 'K⁻', fontsize=12, fontweight='bold',
                       ha='center', va='center', color='white')
        ax_initial.text(0.82, 0.075, '(sū)\nMeson Octet',
                       fontsize=9, va='center')
        
        ax_initial.set_xlim(0, 1)
        ax_initial.set_ylim(0, 1)
        
        # Resonances
        ax_res = fig.add_subplot(gs[1, :])
        ax_res.axis('off')
        ax_res.text(0.05, 0.8, 'Intermediate Resonances:', fontsize=13, fontweight='bold')
        
        resonances_info = [
            (0.05, 0.5, 'Δ⁰(1232)', '(udd)\nBaryon Decuplet\nJ^P = 3/2⁺\nm = 1.232 GeV\nΓ = 0.117 GeV', 'red'),
            (0.37, 0.5, 'Λ*(1520)', '(uds)\nExcited Baryon\nJ^P = 3/2⁻\nm = 1.518 GeV\nΓ = 0.015 GeV', 'orange'),
            (0.69, 0.5, 'K*(892)', '(sū)\nVector Meson Nonet\nJ^P = 1⁻\nm = 0.896 GeV\nΓ = 0.047 GeV', 'cyan'),
        ]
        
        for x, y, name, info, color in resonances_info:
            ax_res.add_patch(Rectangle((x, y-0.05), 0.08, 0.15,
                                      facecolor=color, edgecolor='black', linewidth=2))
            ax_res.text(x+0.04, y+0.025, name, fontsize=10, fontweight='bold',
                       ha='center', va='center')
            ax_res.text(x+0.12, y+0.025, info, fontsize=8, va='center')
        
        ax_res.set_xlim(0, 1)
        ax_res.set_ylim(0, 1)
        
        # Summary table
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis('off')
        
        summary_text = (
            'Summary of SU(3) Multiplets:\n'
            '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
            '• Initial State (Λc⁺): Charmed Baryon Sextet - OUTSIDE pure SU(3)_flavor due to charm quark\n'
            '• Final Baryons (p): Baryon Octet (1/2⁺) - belongs to SU(3)_flavor\n'
            '• Final Mesons (π⁺, K⁻): Pseudoscalar Meson Octet (0⁻) - belongs to SU(3)_flavor\n'
            '• Δ⁰(1232): Baryon Decuplet (3/2⁺) - neutral state highlighted\n'
            '• Λ*(1520): Excited Baryon resonance (3/2⁻) in pK⁻ subsystem\n'
            '• K*(892): Vector Meson Nonet (1⁻) resonance in π⁺K⁻ subsystem'
        )
        
        ax_summary.text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='lightyellow', alpha=0.8, pad=1))
        
        ax_summary.set_xlim(0, 1)
        ax_summary.set_ylim(0, 1)
        
        return fig
    
    def save_all_plots(self, show_plots=False):
        """
        Generate and save all multiplet diagrams.
        
        Args:
            show_plots: If True, display plots interactively
        """
        print("=" * 70)
        print("Generating SU(3) Multiplet Diagrams")
        print("=" * 70)
        
        plots = [
            ('baryon_octet.png', self.plot_baryon_octet, 'Baryon Octet (1/2⁺)'),
            ('meson_octet.png', self.plot_meson_octet, 'Pseudoscalar Meson Octet (0⁻)'),
            ('baryon_decuplet.png', self.plot_baryon_decuplet, 'Baryon Decuplet (3/2⁺)'),
            ('vector_meson_nonet.png', self.plot_vector_meson_nonet, 'Vector Meson Nonet (1⁻)'),
            ('lambda_star_resonance.png', self.plot_excited_baryon_resonance, 'Λ*(1520) Resonance (3/2⁻)'),
            ('charmed_baryons.png', self.plot_charmed_baryons, 'Charmed Baryons (Sextet)'),
            ('decay_summary.png', self.plot_decay_process_summary, 'Decay Process Summary'),
        ]
        
        for filename, plot_func, description in plots:
            print(f"\n📊 Generating: {description}")
            fig = plot_func()
            filepath = f"{self.output_dir}/{filename}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"   ✓ Saved: {filename}")
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        
        print("\n" + "=" * 70)
        print("✓ All multiplet diagrams generated successfully!")
        print("=" * 70 + "\n")


def main(show_plots=False):
    """
    Main function to generate all multiplet diagrams.
    
    Args:
        show_plots: If True, display plots interactively
    """
    # Set output directory to dedicated multiplet subfolder
    output_dir = 'pictures/multiplets'
    
    # Initialize plotter
    plotter = MultipletPlotter(output_dir=output_dir)
    
    # Generate and save all plots
    plotter.save_all_plots(show_plots=show_plots)
    
    print("Generated files:")
    for file in [
        'baryon_octet.png',
        'meson_octet.png',
        'baryon_decuplet.png',
        'vector_meson_nonet.png',
        'lambda_star_resonance.png',
        'charmed_baryons.png',
        'decay_summary.png'
    ]:
        full = os.path.join(output_dir, file)
        if os.path.exists(full):
            print(f"  ✓ {full}")


if __name__ == "__main__":
    main()
