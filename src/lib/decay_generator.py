"""
SU(3) Flavor Symmetry Decay Generator
Task 2: Generate related decays via ladder operators

Reference decay: Λc⁺ → p π⁺ K⁻
- Resonance channels: K*(892), Λ*(1520), Δ(1232)

This script:
1. Implements SU(3) ladder operators (I±, U±, V±)
2. Generates decay families by applying operators
3. Checks conservation laws (Q, B, Y)
4. Lists valid decay chains with resonances
"""

from typing import Tuple, Optional, Set, List
from itertools import product
from src.lib.particles import Particle, BARYONS, MESONS, ALL_PARTICLES, get_resonances_for_pair
import json
import os


class SU3LadderOperator:
    """
    SU(3) flavor ladder operators acting on quark content.
    
    Operators:
    - I+ : u ↔ d (raises I3 by 1)
    - U+ : d ↔ s (raises U-spin)
    - V+ : u ↔ s (raises V-spin)
    """
    
    @staticmethod
    def apply_I_plus(quarks: str) -> Optional[str]:
        """I+ operator: d → u (raises I3 by 1)"""
        if 'd' in quarks:
            return quarks.replace('d', 'u', 1)
        if 'd̄' in quarks:
            return quarks.replace('d̄', 'ū', 1)
        return None
    
    @staticmethod
    def apply_I_minus(quarks: str) -> Optional[str]:
        """I- operator: u → d (lowers I3 by 1)"""
        if 'u' in quarks:
            return quarks.replace('u', 'd', 1)
        if 'ū' in quarks:
            return quarks.replace('ū', 'd̄', 1)
        return None
    
    @staticmethod
    def apply_U_plus(quarks: str) -> Optional[str]:
        """U+ operator: s → d (raises U-spin)"""
        if 's' in quarks:
            return quarks.replace('s', 'd', 1)
        if 's̄' in quarks:
            return quarks.replace('s̄', 'd̄', 1)
        return None
    
    @staticmethod
    def apply_U_minus(quarks: str) -> Optional[str]:
        """U- operator: d → s (lowers U-spin)"""
        if 'd' in quarks:
            return quarks.replace('d', 's', 1)
        if 'd̄' in quarks:
            return quarks.replace('d̄', 's̄', 1)
        return None
    
    @staticmethod
    def apply_V_plus(quarks: str) -> Optional[str]:
        """V+ operator: s → u (raises V-spin)"""
        if 's' in quarks:
            return quarks.replace('s', 'u', 1)
        if 's̄' in quarks:
            return quarks.replace('s̄', 'ū', 1)
        return None
    
    @staticmethod
    def apply_V_minus(quarks: str) -> Optional[str]:
        """V- operator: u → s (lowers V-spin)"""
        if 'u' in quarks:
            return quarks.replace('u', 's', 1)
        if 'ū' in quarks:
            return quarks.replace('ū', 's̄', 1)
        return None
    
    @classmethod
    def find_particle_by_quarks(cls, quarks: str) -> Optional[Particle]:
        """Find particle in database matching quark content (approximately)."""
        # Normalize quark strings for comparison
        for name, particle in ALL_PARTICLES.items():
            if cls._quarks_match(particle.quarks, quarks):
                return particle
        return None
    
    @staticmethod
    def _quarks_match(q1: str, q2: str) -> bool:
        """Check if two quark strings represent the same particle."""
        # If either contains an anti-quark marker, use exact match (mesons like ud̄, sū)
        anti_markers = ('ū', 'd̄', 's̄')
        if any(m in q1 for m in anti_markers) or any(m in q2 for m in anti_markers):
            return q1 == q2
        # For baryons (no anti-quarks), compare irrespective of ordering
        return ''.join(sorted(q1)) == ''.join(sorted(q2))
    
    @classmethod
    def apply_ladder_operator(cls, particle: Particle, operator: str) -> Optional[Particle]:
        """
        Apply a ladder operator to a particle.
        
        Args:
            particle: Input particle
            operator: One of 'I+', 'I-', 'U+', 'U-', 'V+', 'V-'
        
        Returns:
            Transformed particle if it exists in the database, None otherwise
        """
        op_map = {
            'I+': cls.apply_I_plus,
            'I-': cls.apply_I_minus,
            'U+': cls.apply_U_plus,
            'U-': cls.apply_U_minus,
            'V+': cls.apply_V_plus,
            'V-': cls.apply_V_minus,
        }
        
        if operator not in op_map:
            return None
        
        new_quarks = op_map[operator](particle.quarks)
        if new_quarks is None:
            return None
        
        return cls.find_particle_by_quarks(new_quarks)


from dataclasses import dataclass


@dataclass
class Decay:
    """Represents a three-body decay."""
    initial: Particle
    final1: Particle
    final2: Particle
    final3: Particle
    
    def __hash__(self):
        return hash((self.initial, self.final1, self.final2, self.final3))
    
    def __str__(self):
        return f"{self.initial.name} → {self.final1.name} {self.final2.name} {self.final3.name}"
    
    def is_valid(self) -> Tuple[bool, str]:
        """
        Check conservation laws.
        
        Returns:
            (valid, reason) tuple
        """
        # Charge conservation
        Q_in = self.initial.charge
        Q_out = self.final1.charge + self.final2.charge + self.final3.charge
        if Q_in != Q_out:
            return False, f"Charge not conserved: {Q_in} ≠ {Q_out}"
        
        # Baryon number conservation
        B_in = self.initial.baryon_number
        B_out = self.final1.baryon_number + self.final2.baryon_number + self.final3.baryon_number
        if B_in != B_out:
            return False, f"Baryon number not conserved: {B_in} ≠ {B_out}"
        
        # Strangeness can be violated in weak decays, but in SU(3) flavor we check it
        S_in = self.initial.strangeness
        S_out = self.final1.strangeness + self.final2.strangeness + self.final3.strangeness
        
        return True, f"Valid (ΔS = {S_out - S_in})"


class DecayGenerator:
    """Generate SU(3)-related decays from a reference decay."""
    
    def __init__(self, reference_decay: Decay):
        self.reference = reference_decay
        self.operators = ['I+', 'I-', 'U+', 'U-', 'V+', 'V-']
    
    def generate_related_decays(self, max_steps: int = 2) -> Set[Decay]:
        """
        Generate SU(3)-related decays by varying ONLY the final states.
        Keeps the initial state fixed at Λc⁺ for physical relevance.
        
        Args:
            max_steps: Maximum number of operator applications
        
        Returns:
            Set of valid related decays (all with same initial state)
        """
        related = set()
        related.add(self.reference)
        
        # Single operator applications on final state particles only (indices 1, 2, 3)
        for particle_idx in range(1, 4):  # Only final states, NOT initial
            for op in self.operators:
                new_decay = self._apply_operator_to_decay(self.reference, particle_idx, op)
                if new_decay:
                    valid, reason = new_decay.is_valid()
                    if valid:
                        related.add(new_decay)
        
        # Two-step applications (apply to two different final state particles)
        if max_steps >= 2:
            base_decays = list(related)
            for decay in base_decays:
                # Only apply to final states (combinations of 1,2,3)
                for idx1, idx2 in [(1,2), (1,3), (2,3)]:
                    for op1, op2 in product(self.operators, repeat=2):
                        new_decay = self._apply_two_operators(decay, idx1, op1, idx2, op2)
                        if new_decay:
                            valid, reason = new_decay.is_valid()
                            if valid:
                                related.add(new_decay)
        
        return related
    
    @staticmethod
    def _apply_operator_to_decay(decay: Decay, particle_idx: int, operator: str) -> Optional[Decay]:
        """Apply operator to one particle in the decay."""
        particles = [decay.initial, decay.final1, decay.final2, decay.final3]
        
        new_particle = SU3LadderOperator.apply_ladder_operator(particles[particle_idx], operator)
        if new_particle is None:
            return None
        
        particles[particle_idx] = new_particle
        return Decay(*particles)
    
    @staticmethod
    def _apply_two_operators(decay: Decay, idx1: int, op1: str, idx2: int, op2: str) -> Optional[Decay]:
        """Apply operators to two particles in the decay."""
        particles = [decay.initial, decay.final1, decay.final2, decay.final3]
        
        new_p1 = SU3LadderOperator.apply_ladder_operator(particles[idx1], op1)
        if new_p1 is None:
            return None
        particles[idx1] = new_p1
        
        new_p2 = SU3LadderOperator.apply_ladder_operator(particles[idx2], op2)
        if new_p2 is None:
            return None
        particles[idx2] = new_p2
        
        return Decay(*particles)


def enumerate_subsystems(decay: Decay) -> List[Tuple[Tuple[str, str], str]]:
    """Return all two-body subsystems and the spectator name.
    Output list items are ((a,b), spectator). Order of a,b normalized alphabetically.
    """
    a = decay.final1.name
    b = decay.final2.name
    c = decay.final3.name
    subs = [((min(a, b), max(a, b)), c),
            ((min(a, c), max(a, c)), b),
            ((min(b, c), max(b, c)), a)]
    return subs


def find_decay_chains(decay: Decay) -> List[str]:
    """Build human-readable decay chain strings for all applicable resonances.
    Uses the resonance database limited to K*(892), Λ*(1520), Δ(1232).
    """
    chains: List[str] = []
    subsystems = enumerate_subsystems(decay)
    for (p1, p2), spectator in subsystems:
        resonances = get_resonances_for_pair(p1, p2)
        if not resonances:
            continue
        for R in resonances:
            # Production step: initial -> R + spectator
            prod = f"{decay.initial.name} → {R.name} {spectator}"
            # Decay step: R -> p1 p2
            dec = f"{R.name} → {p1} {p2}"
            # Full chain with indications
            chain = f"{prod} ; {dec}"
            chains.append(chain)
    return chains


def compute_delta_s(decay: Decay) -> int:
    """Compute ΔS = (S_final_total - S_initial)."""
    S_in = decay.initial.strangeness
    S_out = decay.final1.strangeness + decay.final2.strangeness + decay.final3.strangeness
    return int(S_out - S_in)


def decay_to_dict(decay: Decay) -> dict:
    """Serialize a decay and its possible resonance chains to a dict."""
    ds = compute_delta_s(decay)
    subsystems = enumerate_subsystems(decay)
    chains: List[dict] = []
    for (p1, p2), spectator in subsystems:
        resonances = get_resonances_for_pair(p1, p2)
        for R in resonances:
            chains.append({
                "subsystem": [p1, p2],
                "spectator": spectator,
                "resonance": {
                    "name": R.name,
                    "mass_GeV": R.mass,
                    "width_GeV": R.width,
                    "quarks": R.quarks,
                    "charge": R.charge,
                    "baryon_number": R.baryon_number,
                    "strangeness": R.strangeness,
                    "isospin3": R.isospin3,
                    "decay_products": R.decay_products,
                    "g_prod": [R.g_prod.real, R.g_prod.imag],
                    "g_dec": [R.g_dec.real, R.g_dec.imag],
                },
                "production": f"{decay.initial.name} → {R.name} {spectator}",
                "decay": f"{R.name} → {p1} {p2}",
            })
    return {
        "decay": str(decay),
        "deltaS": ds,
        "initial": decay.initial.name,
        "finals": [decay.final1.name, decay.final2.name, decay.final3.name],
        "chains": chains,
    }


def export_to_json(decays: Set[Decay], path: str) -> None:
    """Export decays and their chains to a JSON file, sorted by ΔS."""
    sorted_decays = sorted(decays, key=lambda d: (compute_delta_s(d), str(d)))
    payload = [decay_to_dict(d) for d in sorted_decays]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def main():
    """Main script to generate and validate related decays."""
    print("=" * 80)
    print("SU(3) Flavor Symmetry: Related Decay Generator")
    print("=" * 80)
    print()
    
    # Define reference decay: Λc⁺ → p π⁺ K⁻
    ref = Decay(
        initial=BARYONS['Λc⁺'],
        final1=BARYONS['p'],
        final2=MESONS['π⁺'],
        final3=MESONS['K⁻']
    )
    
    print(f"Reference decay: {ref}")
    valid, reason = ref.is_valid()
    print(f"  Validity: {reason}")
    print()
    
    # Generate related decays
    print("Generating SU(3)-related decays...")
    print("(Fixed initial state Λc⁺; varying final states via ladder operators)")
    print()
    
    generator = DecayGenerator(ref)
    related_decays = generator.generate_related_decays(max_steps=2)
    
    print(f"Found {len(related_decays)} related decay channels with same initial state:")
    print("-" * 80)
    
    # Sort decays by ΔS value (ascending), then by string for stable order
    sorted_decays = sorted(related_decays, key=lambda d: (compute_delta_s(d), str(d)))
    for i, decay in enumerate(sorted_decays, 1):
        ds = compute_delta_s(decay)
        # Standardized '|' line format with single spaces around the pipe
        print(f"{i:3d}. {str(decay):50s} | Valid (ΔS = {ds:+d})")
        chains = find_decay_chains(decay)
        if chains:
            for ch in chains:
                print(f"      → {ch}")
        else:
            print("      → (no listed resonances for given pairs)")

    # Export JSON
    out_path = os.path.join("outputs", "decay_chains.json")
    export_to_json(set(sorted_decays), out_path)
    print()
    print(f"Saved decay chains JSON: {out_path}")
    
    print()
    print("=" * 80)
    print("Conservation Laws Checked:")
    print("  ✓ Electric charge Q")
    print("  ✓ Baryon number B")
    print("  ⓘ Strangeness ΔS reported (can be violated in weak decays)")
    print()
    print("Note: Kinematic feasibility (phase space) requires mass checks (not done here).")
    print("=" * 80)


if __name__ == "__main__":
    main()
