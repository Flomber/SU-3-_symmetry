"""
SU(3) coupling mapper for Task 3.

Purpose
- Centralize PDG inputs (mass, width, branching) for the resonances used in Λc+ → p π+ K−
  and their SU(3)/isospin partners.
- Provide Clebsch–Gordan-based relative factors to map production/decay couplings across
  charge states (Δ(1232) family, K*(892) family, Λ*(1520) singlet).
- Leave placeholders for PDG values; fill them from pdgLive before numerical evaluation.

Notes
- Charm is treated as spectator; SU(3)_flavor acts on u, d, s only.
- Spin dynamics and Blatt–Weisskopf factors are neglected here; g_prod/g_dec are reduced
  couplings up to a common overall normalization.
- Signs follow the standard Condon–Shortley phase convention; adjust consistently if you
  change the convention.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

# -----------------------------------------------------------------------------
# PDG input placeholders (fill from pdgLive)
# -----------------------------------------------------------------------------

@dataclass
class PDGInputs:
    mass: Optional[float]  # GeV
    width: Optional[float]  # GeV (total)
    br_to_mode: Optional[float]  # branching fraction to the specific two-body mode in decimal form (e.g. 0.67 for 67%)


PDG_PLACEHOLDERS: Dict[str, PDGInputs] = {
    # Vector meson
    "K*(892)+": PDGInputs(mass=0.896, width=0.0514, br_to_mode=1.0),
    "K*(892)0": PDGInputs(mass=0.896, width=0.0473, br_to_mode=1.0),
    "K*(892)-": PDGInputs(mass=0.896, width=0.0514, br_to_mode=1.0),
    "K*(892)bar0": PDGInputs(mass=0.896, width=0.0473, br_to_mode=1.0),
    # Baryon decuplet
    "Delta(1232)++": PDGInputs(mass=1.23055, width=0.1122, br_to_mode=0.9939),
    "Delta(1232)+": PDGInputs(mass=1.2349, width=0.1311, br_to_mode=0.9939),
    "Delta(1232)0": PDGInputs(mass=1.2334, width=0.1169, br_to_mode=0.9939),
    # Baryon negative-parity singlet (dominant NK mode used here)
    "Lambda(1520)": PDGInputs(mass=1.51942, width=0.01573, br_to_mode=0.45),
}


# -----------------------------------------------------------------------------
# Simple helpers for partial widths and placeholder decay couplings
# -----------------------------------------------------------------------------

def partial_width(inputs: PDGInputs) -> float:
    if inputs.width is None or inputs.br_to_mode is None:
        raise ValueError("PDG inputs missing: width or branching fraction is None")
    return inputs.width * inputs.br_to_mode


def infer_g_dec_placeholder(inputs: PDGInputs, phase_space_norm: float = 1.0) -> float:
    """Toy estimator: g_dec ∝ sqrt(Γ_partial / phase_space_norm).
    Replace with your preferred dynamical model (e.g. P-wave factors) if needed.
    """
    pw = partial_width(inputs)
    return math.sqrt(pw / phase_space_norm)


# -----------------------------------------------------------------------------
# Clebsch–Gordan factors (isospin) for decay vertices R → AB
# Values are relative amplitudes; overall normalization set by reference channel.
# -----------------------------------------------------------------------------

# Δ(1232) → N π (I=3/2 → 1/2 ⊗ 1)
DELTA_DECAY_CG: Dict[Tuple[str, str], float] = {
    ("Delta(1232)++", "p π+" ): 1.0,
    ("Delta(1232)+",  "p π0" ): math.sqrt(2/3),
    ("Delta(1232)+",  "n π+" ): math.sqrt(1/3),
    ("Delta(1232)0",  "p π-" ): math.sqrt(1/3),
    ("Delta(1232)0",  "n π0" ): math.sqrt(2/3),
    ("Delta(1232)-",  "n π-" ): 1.0,
}

# K*(892) → K π (I=1/2 → 1/2 ⊗ 1), signs per common convention
KSTAR_DECAY_CG: Dict[Tuple[str, str], float] = {
    ("K*(892)+",  "K+ π0" ):  1 / math.sqrt(2),
    ("K*(892)+",  "K0 π+" ):  1.0,
    ("K*(892)0",  "K+ π-" ): -1.0,
    ("K*(892)0",  "K0 π0" ): -1 / math.sqrt(2),
    ("K*(892)-",  "K- π0" ):  1 / math.sqrt(2),
    ("K*(892)-",  "Kbar0 π-" ): 1.0,
    ("K*(892)bar0", "K- π+" ): -1.0,
    ("K*(892)bar0", "Kbar0 π0" ): -1 / math.sqrt(2),
}

# Λ*(1520) → N K (I=0 → 1/2 ⊗ 1/2) is unique up to a sign; set to 1.0
LAMBDA1520_DECAY_CG: Dict[Tuple[str, str], float] = {
    ("Lambda(1520)", "p K-"  ): 1.0,
    ("Lambda(1520)", "n Kbar0" ): 1.0,
}


def decay_cg(resonance: str, final_state: str) -> float:
    """Return the relative CG factor for R → final_state.
    final_state must be a space-separated string with explicit charges.
    """
    if resonance.startswith("Delta(1232)"):
        return DELTA_DECAY_CG[(resonance, final_state)]
    if resonance.startswith("K*(892)"):
        return KSTAR_DECAY_CG[(resonance, final_state)]
    if resonance.startswith("Lambda(1520)"):
        return LAMBDA1520_DECAY_CG[(resonance, final_state)]
    raise KeyError(f"No CG entry for {resonance} → {final_state}")


# -----------------------------------------------------------------------------
# Production CG placeholders
# For simplicity, reuse the decay CG of the produced resonance with the pair it forms.
# Replace with a detailed SU(3) Wigner–Eckart treatment if needed.
# -----------------------------------------------------------------------------

def production_cg(resonance: str, pair_label: str) -> float:
    """Placeholder: map production strength using the same CG factor structure as decay.
    pair_label is the two-body decay mode of the resonance (same string as final_state).
    """
    return decay_cg(resonance, pair_label)


# -----------------------------------------------------------------------------
# Mapping utility: given a reference coupling, rescale to partner states
# -----------------------------------------------------------------------------

def map_decay_coupling(resonance: str, final_state: str, g_dec_ref: complex) -> complex:
    """Scale g_dec_ref by CG ratio so all charge states follow SU(2)/SU(3) symmetry."""
    factor = decay_cg(resonance, final_state)
    return g_dec_ref * factor


def map_production_coupling(resonance: str, pair_label: str, g_prod_ref: complex) -> complex:
    factor = production_cg(resonance, pair_label)
    return g_prod_ref * factor


# -----------------------------------------------------------------------------
# Example recipe (non-numeric until PDG values are filled):
# 1) Fill PDG_PLACEHOLDERS with mass, width, branching for each charge state.
# 2) Choose a reference channel, e.g. g_prod_ref = (0.665593+1.08922j) for Δ++ K−
#    and g_dec_ref = infer_g_dec_placeholder(PDG_PLACEHOLDERS["Delta(1232)++"], phase_space_norm=1.0).
# 3) Map to other Δ charge states via map_production_coupling/map_decay_coupling.
# 4) Repeat for K* family using the reference K̄*0 → K− π+ channel from Table 1.
# 5) Λ*(1520) is isospin singlet: use the same coupling for pK− and nK̄0.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Lightweight sanity printout of CG tables
    print("Δ(1232) decay CG factors:")
    for k, v in DELTA_DECAY_CG.items():
        print(f"  {k[0]} → {k[1]} : {v:.4f}")

    print("\nK*(892) decay CG factors:")
    for k, v in KSTAR_DECAY_CG.items():
        print(f"  {k[0]} → {k[1]} : {v:.4f}")

    print("\nΛ*(1520) decay CG factors:")
    for k, v in LAMBDA1520_DECAY_CG.items():
        print(f"  {k[0]} → {k[1]} : {v:.4f}")
