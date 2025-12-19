"""
Particle Database for SU(3) Flavor Symmetry
Contains hadron definitions with quantum numbers for multiplet analysis and decay studies.
"""

from dataclasses import dataclass


@dataclass
class Particle:
    """Represents a hadron with SU(3) quantum numbers."""
    name: str
    quarks: str  # Quark content (e.g., 'uud', 'us̄')
    charge: int  # Electric charge in units of e
    baryon_number: int  # 0 for mesons, 1 for baryons
    strangeness: int  # S quantum number
    isospin3: float  # I3 quantum number
    
    @property
    def hypercharge(self) -> float:
        """Y = B + S"""
        return self.baryon_number + self.strangeness
    
    def __hash__(self):
        return hash((self.name, self.quarks))
    
    def __eq__(self, other):
        if not isinstance(other, Particle):
            return False
        return self.name == other.name and self.quarks == other.quarks


# Baryon Database
BARYONS = {
    # Octet (1/2+) - Ground state baryons
    'p': Particle('p', 'uud', 1, 1, 0, 0.5),
    'n': Particle('n', 'udd', 0, 1, 0, -0.5),
    'Λ': Particle('Λ', 'uds', 0, 1, -1, 0),
    'Σ⁺': Particle('Σ⁺', 'uus', 1, 1, -1, 1),
    'Σ⁰': Particle('Σ⁰', 'uds', 0, 1, -1, 0),
    'Σ⁻': Particle('Σ⁻', 'dds', -1, 1, -1, -1),
    'Ξ⁰': Particle('Ξ⁰', 'uss', 0, 1, -2, 0.5),
    'Ξ⁻': Particle('Ξ⁻', 'dss', -1, 1, -2, -0.5),
    
    # Decuplet (3/2+) - Excited baryons
    'Δ⁺⁺': Particle('Δ⁺⁺', 'uuu', 2, 1, 0, 1.5),
    'Δ⁺': Particle('Δ⁺', 'uud', 1, 1, 0, 0.5),
    'Δ⁰': Particle('Δ⁰', 'udd', 0, 1, 0, -0.5),
    'Δ⁻': Particle('Δ⁻', 'ddd', -1, 1, 0, -1.5),
    'Σ*⁺': Particle('Σ*⁺', 'uus', 1, 1, -1, 1),
    'Σ*⁰': Particle('Σ*⁰', 'uds', 0, 1, -1, 0),
    'Σ*⁻': Particle('Σ*⁻', 'dds', -1, 1, -1, -1),
    'Ξ*⁰': Particle('Ξ*⁰', 'uss', 0, 1, -2, 0.5),
    'Ξ*⁻': Particle('Ξ*⁻', 'dss', -1, 1, -2, -0.5),
    'Ω⁻': Particle('Ω⁻', 'sss', -1, 1, -3, 0),
    
    # Excited baryons (3/2-) - Negative parity octet
    'N*(1520)⁺': Particle('N*(1520)⁺', 'uud', 1, 1, 0, 0.5),
    'N*(1520)⁰': Particle('N*(1520)⁰', 'udd', 0, 1, 0, -0.5),
    'Λ*(1520)': Particle('Λ*(1520)', 'uds', 0, 1, -1, 0),
    'Σ*(1670)⁺': Particle('Σ*(1670)⁺', 'uus', 1, 1, -1, 1),
    'Σ*(1670)⁰': Particle('Σ*(1670)⁰', 'uds', 0, 1, -1, 0),
    'Σ*(1670)⁻': Particle('Σ*(1670)⁻', 'dds', -1, 1, -1, -1),
    'Ξ*(1690)⁰': Particle('Ξ*(1690)⁰', 'uss', 0, 1, -2, 0.5),
    'Ξ*(1690)⁻': Particle('Ξ*(1690)⁻', 'dss', -1, 1, -2, -0.5),
    
    # Charmed baryons - Anti-triplet (spin-0 light diquark)
    'Λc⁺': Particle('Λc⁺', 'udc', 1, 1, 0, 0),
    'Ξc⁺': Particle('Ξc⁺', 'usc', 1, 1, -1, 0.5),
    'Ξc⁰': Particle('Ξc⁰', 'dsc', 0, 1, -1, -0.5),
    
    # Charmed baryons - Sextet (spin-1 light diquark)
    'Σc⁺⁺': Particle('Σc⁺⁺', 'uuc', 2, 1, 0, 1),
    'Σc⁺': Particle('Σc⁺', 'udc', 1, 1, 0, 0),
    'Σc⁰': Particle('Σc⁰', 'ddc', 0, 1, 0, -1),
    'Ξc\'⁺': Particle('Ξc\'⁺', 'usc', 1, 1, -1, 0.5),
    'Ξc\'⁰': Particle('Ξc\'⁰', 'dsc', 0, 1, -1, -0.5),
    'Ωc⁰': Particle('Ωc⁰', 'ssc', 0, 1, -2, 0),
}


# Meson Database
MESONS = {
    # Pseudoscalar nonet (0-) - Ground state mesons
    'π⁺': Particle('π⁺', 'ud̄', 1, 0, 0, 1),
    'π⁰': Particle('π⁰', 'uū-dd̄', 0, 0, 0, 0),
    'π⁻': Particle('π⁻', 'dū', -1, 0, 0, -1),
    'η': Particle('η', 'uū+dd̄-2ss̄', 0, 0, 0, 0),
    'η\'': Particle('η\'', 'uū+dd̄+ss̄', 0, 0, 0, 0),
    'K⁺': Particle('K⁺', 'us̄', 1, 0, 1, 0.5),
    'K⁰': Particle('K⁰', 'ds̄', 0, 0, 1, -0.5),
    'K̄⁰': Particle('K̄⁰', 'sd̄', 0, 0, -1, 0.5),
    'K⁻': Particle('K⁻', 'sū', -1, 0, -1, -0.5),
    
    # Vector nonet (1-) - Excited mesons
    'ρ⁺': Particle('ρ⁺', 'ud̄', 1, 0, 0, 1),
    'ρ⁰': Particle('ρ⁰', 'uū-dd̄', 0, 0, 0, 0),
    'ρ⁻': Particle('ρ⁻', 'dū', -1, 0, 0, -1),
    'ω': Particle('ω', 'uū+dd̄', 0, 0, 0, 0),
    'φ': Particle('φ', 'ss̄', 0, 0, 0, 0),
    'K*⁺': Particle('K*⁺', 'us̄', 1, 0, 1, 0.5),
    'K*⁰': Particle('K*⁰', 'ds̄', 0, 0, 1, -0.5),
    'K̄*⁰': Particle('K̄*⁰', 'sd̄', 0, 0, -1, 0.5),
    'K*⁻': Particle('K*⁻', 'sū', -1, 0, -1, -0.5),
}


# Combined dictionary for easy lookup
ALL_PARTICLES = {**BARYONS, **MESONS}


# Convenience functions
def get_particle(name: str) -> Particle:
    """Get particle by name."""
    if name in ALL_PARTICLES:
        return ALL_PARTICLES[name]
    raise ValueError(f"Particle '{name}' not found in database")


def find_particle_by_quarks(quarks: str) -> Particle:
    """Find particle by quark content."""
    for particle in ALL_PARTICLES.values():
        if particle.quarks == quarks:
            return particle
    raise ValueError(f"No particle found with quark content '{quarks}'")


def find_particle_by_quantum_numbers(charge: int, baryon_number: int, 
                                     strangeness: int, isospin3: float) -> Particle:
    """Find particle by quantum numbers (Q, B, S, I3)."""
    for particle in ALL_PARTICLES.values():
        if (particle.charge == charge and 
            particle.baryon_number == baryon_number and
            particle.strangeness == strangeness and
            abs(particle.isospin3 - isospin3) < 0.01):
            return particle
    raise ValueError(f"No particle found with Q={charge}, B={baryon_number}, S={strangeness}, I3={isospin3}")


def get_baryons() -> dict:
    """Get all baryons."""
    return BARYONS.copy()


def get_mesons() -> dict:
    """Get all mesons."""
    return MESONS.copy()


# ============================================================================
# RESONANCE DATABASE
# ============================================================================

from typing import List
from src.lib.mapping_couplings import (
    PDG_PLACEHOLDERS,
    infer_g_dec_placeholder,
    map_production_coupling,
    map_decay_coupling,
)

@dataclass
class Resonance:
    """Represents a resonance with production/decay properties."""
    name: str
    mass: float  # in GeV
    width: float  # in GeV
    quarks: str  # Quark content
    charge: int  # Electric charge
    baryon_number: int  # 0 for mesons, 1 for baryons
    strangeness: int  # S quantum number
    isospin3: float  # I3 quantum number
    decay_products: List[str]  # List of two-body decay product names
    g_prod: complex  # Production coupling (dimensionless)
    g_dec: complex = 1.0 + 0j  # Decay coupling (default 1.0)
    
    @property
    def hypercharge(self) -> float:
        """Y = B + S"""
        return self.baryon_number + self.strangeness


# Resonance database for Λc⁺ → p π⁺ K⁻ decay analysis
RESONANCES = {
    # Vector meson resonance
    'K*(892)': Resonance(
        name='K*(892)',
        mass=0.896,
        width=0.047,
        quarks='us̄',
        charge=-1,
        baryon_number=0,
        strangeness=-1,
        isospin3=-0.5,
        decay_products=['π⁺', 'K⁻'],  # Decays to final meson pair
        g_prod=1.0 + 0j,
        g_dec=1.0 + 0j,
    ),
    
    # Baryon resonance (negative parity)
    'Λ*(1520)': Resonance(
        name='Λ*(1520)',
        mass=1.518,
        width=0.015,
        quarks='uds',
        charge=0,
        baryon_number=1,
        strangeness=-1,
        isospin3=0,
        decay_products=['p', 'K⁻'],  # Decays to baryon + meson
        g_prod=complex(3.2582, 1.7589),
        g_dec=1.0 + 0j,
    ),
    
    # Baryon resonance (3/2+ decuplet)
    'Δ(1232)': Resonance(
        name='Δ(1232)',
        mass=1.232,
        width=0.117,
        quarks='uud',
        charge=1,
        baryon_number=1,
        strangeness=0,
        isospin3=0.5,
        decay_products=['p', 'π⁺'],  # Decays to baryon + meson
        g_prod=complex(0.665593, 1.08922),
        g_dec=1.0 + 0j,
    ),
}


def get_resonance(name: str) -> Resonance:
    """Get resonance by name."""
    if name in RESONANCES:
        return RESONANCES[name]
    raise ValueError(f"Resonance '{name}' not found in database")


def get_all_resonances() -> dict:
    """Get all resonances."""
    return RESONANCES.copy()


# ----------------------------------------------------------------------------
# Helpers to resolve charge states based on final pairs
# ----------------------------------------------------------------------------

def _build_delta_resonance(charge_state: str, decay_products: List[str]) -> Resonance:
    """Create a specific Δ(1232) charge-state resonance for given decay products."""
    quarks_map = {
        'Δ⁺⁺': ('uuu', 1.5, 2),
        'Δ⁺': ('uud', 0.5, 1),
        'Δ⁰': ('udd', -0.5, 0),
        'Δ⁻': ('ddd', -1.5, -1),
    }
    quarks, i3, q = quarks_map[charge_state]
    # Canonical mapping names for CG/PDG tables
    canon_name_map = {
        'Δ⁺⁺': 'Delta(1232)++',
        'Δ⁺': 'Delta(1232)+',
        'Δ⁰': 'Delta(1232)0',
        'Δ⁻': 'Delta(1232)-',
    }
    canon_res = canon_name_map[charge_state]
    # Build final-state label in N π order with canonical charges
    charge_map = {'π⁺': 'π+', 'π⁰': 'π0', 'π⁻': 'π-', 'p': 'p', 'n': 'n'}
    # Ensure order: N then π
    if 'p' in decay_products or 'n' in decay_products:
        if decay_products[0] in ('p', 'n'):
            N, P = decay_products[0], decay_products[1]
        else:
            N, P = decay_products[1], decay_products[0]
    else:
        N, P = decay_products[0], decay_products[1]
    final_label = f"{charge_map[N]} {charge_map[P]}"

    # Production coupling: scale reference by CG
    g_prod_ref = RESONANCES['Δ(1232)'].g_prod
    g_prod = map_production_coupling(canon_res, final_label, g_prod_ref)

    # Decay coupling: magnitude from PDG partial width × CG
    pdg_in = PDG_PLACEHOLDERS.get(canon_res)
    g_dec_mag_ref = infer_g_dec_placeholder(pdg_in) if pdg_in else 1.0
    g_dec = map_decay_coupling(canon_res, final_label, g_dec_mag_ref + 0j)

    return Resonance(
        name=f'{charge_state}(1232)',
        mass=RESONANCES['Δ(1232)'].mass,
        width=RESONANCES['Δ(1232)'].width,
        quarks=quarks,
        charge=q,
        baryon_number=1,
        strangeness=0,
        isospin3=i3,
        decay_products=decay_products,
        g_prod=g_prod,
        g_dec=g_dec,
    )


def _build_kstar_resonance(state_name: str, decay_products: List[str]) -> Resonance:
    """Create a specific K*(892) charge-state resonance using meson quantum numbers."""
    # Fetch particle to read quantum numbers
    p = get_particle(state_name)
    # Canonical resonance name mapping
    canon_name_map = {
        'K*⁺': 'K*(892)+',
        'K*⁰': 'K*(892)0',
        'K̄*⁰': 'K*(892)bar0',
        'K*⁻': 'K*(892)-',
    }
    base_name = state_name.replace('(892)', '').strip()
    canon_res = canon_name_map.get(base_name, canon_name_map.get(state_name.replace('(892)', ''), None))
    # Final-state label in K π order with canonical charges
    charge_map = {
        'K⁺': 'K+', 'K⁰': 'K0', 'K̄⁰': 'Kbar0', 'K⁻': 'K-',
        'π⁺': 'π+', 'π⁰': 'π0', 'π⁻': 'π-',
    }
    # Ensure order: K then π
    if decay_products[0].startswith('K') or decay_products[0] in ('K⁺', 'K⁰', 'K̄⁰', 'K⁻'):
        K, P = decay_products[0], decay_products[1]
    else:
        K, P = decay_products[1], decay_products[0]
    final_label = f"{charge_map[K]} {charge_map[P]}"

    # Production coupling: scale reference by CG
    g_prod_ref = RESONANCES['K*(892)'].g_prod
    g_prod = map_production_coupling(canon_res, final_label, g_prod_ref) if canon_res else g_prod_ref

    # Decay coupling: magnitude from PDG partial width × CG
    pdg_in = PDG_PLACEHOLDERS.get(canon_res) if canon_res else None
    g_dec_mag_ref = infer_g_dec_placeholder(pdg_in) if pdg_in else 1.0
    g_dec = map_decay_coupling(canon_res, final_label, g_dec_mag_ref + 0j) if canon_res else (g_dec_mag_ref + 0j)

    return Resonance(
        name=f'{state_name}(892)',
        mass=RESONANCES['K*(892)'].mass,
        width=RESONANCES['K*(892)'].width,
        quarks=p.quarks,
        charge=p.charge,
        baryon_number=0,
        strangeness=p.strangeness,
        isospin3=p.isospin3,
        decay_products=decay_products,
        g_prod=g_prod,
        g_dec=g_dec,
    )


def get_resonances_for_pair(p1_name: str, p2_name: str) -> List[Resonance]:
    """
    Given two final-state particle names, return applicable resonances among
    K*(892), Λ*(1520), and Δ(1232) with the correct charge states.
    Order of p1/p2 is irrelevant.
    """
    a = get_particle(p1_name)
    b = get_particle(p2_name)
    # Normalize order
    names = sorted([p1_name, p2_name])
    res: List[Resonance] = []

    # Baryon–pion: Δ(1232) family
    if any(x in ['π⁺', 'π⁰', 'π⁻'] for x in names) and any(x in ['p', 'n'] for x in names):
        # Map to Δ charge state by specific combination
        if set(names) == {'p', 'π⁺'}:
            res.append(_build_delta_resonance('Δ⁺⁺', ['p', 'π⁺']))
        elif set(names) == {'p', 'π⁰'}:
            res.append(_build_delta_resonance('Δ⁺', ['p', 'π⁰']))
        elif set(names) == {'p', 'π⁻'}:
            res.append(_build_delta_resonance('Δ⁰', ['p', 'π⁻']))
        elif set(names) == {'n', 'π⁺'}:
            res.append(_build_delta_resonance('Δ⁺', ['n', 'π⁺']))
        elif set(names) == {'n', 'π⁰'}:
            res.append(_build_delta_resonance('Δ⁰', ['n', 'π⁰']))
        elif set(names) == {'n', 'π⁻'}:
            res.append(_build_delta_resonance('Δ⁻', ['n', 'π⁻']))

    # Baryon–kaon: Λ*(1520) (isospin singlet). Adjust couplings per PDG.
    if set(names) == {'p', 'K⁻'} or set(names) == {'n', 'K̄⁰'}:
        base = RESONANCES['Λ*(1520)']
        # Build final-state canonical label
        charge_map = {'K⁻': 'K-', 'K̄⁰': 'Kbar0', 'p': 'p', 'n': 'n'}
        if set(names) == {'p', 'K⁻'}:
            fs_label = f"{charge_map['p']} {charge_map['K⁻']}"
        else:
            fs_label = f"{charge_map['n']} {charge_map['K̄⁰']}"
        pdg_in = PDG_PLACEHOLDERS.get('Lambda(1520)')
        g_dec_mag_ref = infer_g_dec_placeholder(pdg_in) if pdg_in else 1.0
        g_dec = map_decay_coupling('Lambda(1520)', fs_label, g_dec_mag_ref + 0j)
        # Production left as base (singlet), but could be adjusted if needed
        res.append(Resonance(
            name=base.name,
            mass=base.mass,
            width=base.width,
            quarks=base.quarks,
            charge=base.charge,
            baryon_number=base.baryon_number,
            strangeness=base.strangeness,
            isospin3=base.isospin3,
            decay_products=list(names),
            g_prod=base.g_prod,
            g_dec=g_dec,
        ))

    # Pion–kaon: K*(892) family (physical charge combinations)
    if any(x in ['π⁺', 'π⁰', 'π⁻'] for x in names) and any(x in ['K⁺', 'K⁰', 'K̄⁰', 'K⁻'] for x in names):
        if set(names) == {'π⁺', 'K⁻'}:
            # K̄*⁰ → K⁻ π⁺
            res.append(_build_kstar_resonance('K̄*⁰', ['K⁻', 'π⁺']))
        elif set(names) == {'π⁻', 'K⁺'}:
            # K*⁰ → K⁺ π⁻
            res.append(_build_kstar_resonance('K*⁰', ['K⁺', 'π⁻']))
        elif set(names) == {'π⁰', 'K⁺'}:
            # K*⁺ → K⁺ π⁰
            res.append(_build_kstar_resonance('K*⁺', ['K⁺', 'π⁰']))
        elif set(names) == {'π⁰', 'K⁻'}:
            # K*⁻ → K⁻ π⁰
            res.append(_build_kstar_resonance('K*⁻', ['K⁻', 'π⁰']))
        elif set(names) == {'π⁺', 'K⁰'}:
            # K*⁺ → K⁰ π⁺
            res.append(_build_kstar_resonance('K*⁺', ['K⁰', 'π⁺']))
        elif set(names) == {'π⁻', 'K̄⁰'}:
            # K*⁻ → K̄⁰ π⁻
            res.append(_build_kstar_resonance('K*⁻', ['K̄⁰', 'π⁻']))
        elif set(names) == {'π⁰', 'K⁰'}:
            # K*⁰ → K⁰ π⁰
            res.append(_build_kstar_resonance('K*⁰', ['K⁰', 'π⁰']))
        elif set(names) == {'π⁰', 'K̄⁰'}:
            # K̄*⁰ → K̄⁰ π⁰
            res.append(_build_kstar_resonance('K̄*⁰', ['K̄⁰', 'π⁰']))

    return res
