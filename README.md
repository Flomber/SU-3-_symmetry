# Problem 1: Flavor SU(3) relation (10 points)

## Thematics

The SU(3) flavor symmetry is an approximate symmetry of the strong interaction. It is useful to understand the relation between different particle interactions.

Consider the differential decay rate of the reference reaction:

$Λ_c^+ → p π⁺ K⁻$

Identify decays that are related to this reference reaction via SU(3) symmetry. Sketch how the differential decay rates appear in their kinematic phase space.

The decay matrix elements of the Λ_c^+ → p π⁺ K⁻ reaction can be written as:

$𝓜(m_{12}^2, m_{23}^2) = \sum_R g_{\text{prod}}^{(R)} · \mathcal{A}_R(m_{ij}^2) · g_{\text{dec}}^{(R)}$

Neglect spin effects and consider only dominant resonance contributions.

Indices:

- 0: $Λ_c^+$
- 1: p
- 2: $π^+$
- 3: $K^-$

Where:

- **$g_{\text{prod}}^{(R)}$** is the coupling for Λ_c^+ → (baryon or meson resonance) + spectator
- **$g_{\text{dec}}^{(R)}$** is the coupling for the resonance decay to a two-body final state

## Resonance Amplitudes and Kinematic Space

The resonance amplitude is parametrized by the Breit-Wigner function:

$$
\mathcal{A}_R(m_{ij}^2) = \frac{1}{m_R^2 - m_{ij}^2 - i m_R \Gamma_R}
$$

where:

- $ m_R $: mass of the resonance $ R $
- $ \Gamma_R $: width of the resonance $ R $
- $ m_{ij}^2 $: invariant mass of the two-body subsystem $ ij $

The kinematic space is bounded by the Kibble function:

$$
\phi(m_{12}^2, m_{23}^2) = \lambda(\lambda(m_{12}^2, m_3^2, m_0^2), \lambda(m_{23}^2, m_1^2, m_0^2), \lambda(m_{31}^2, m_2^2, m_0^2))
$$

with the standard Källén function:

$$
\lambda(a, b, c) = a^2 + b^2 + c^2 - 2(ab + bc + ca)
$$

and:

$$
m_{31}^2 = m_0^2 + m_1^2 + m_2^2 + m_3^2 - m_{12}^2 - m_{23}^2
$$

### Table 1: Resonance Parameters in Λ_c^+ → p K⁻ π⁺ Decays

| Resonance             | $ m_R $ (GeV)     | $ \Gamma_R $ (GeV)    | $ g_{\text{prod}}^{(R)} $         | $ g_{\text{dec}}^{(R)} $ |
|-------------------    |------------------ |-----------------------|-----------------------------------|--------------------------|
| $ K^*(892) $          | 0.896             | 0.047                 | 1.0                               | 1.0                      |
| $ \Lambda^*(1520) $   | 1.518             | 0.015                 | 3.2582 + 1.7589i                  | 1.0                      |
| $ \Delta^*(1232) $    | 1.232             | 0.117                 | 0.665593 + 1.08922i               | 1.0                      |

## Hints on the Procedure

1. **Multiplets**  
   Start from quark content. Place hadrons (initial, final states and resonances) into SU(3) multiplets as discussed in _Thomson Chapter 9_ (octet, decuplet, etc.).  
   **(3 pts)**

2. **Ladder Operators**  
   Use $ I_{\pm}, U_{\pm}, V_{\pm} $ on the quark states to generate flavor partners for all involved hadrons.  
   - List decay reactions related to the reference one via SU(3) symmetry. **(1 pt)**  
   - List their decay chains. **(1 pt)**

3. **Couplings**  
   Perform the SU(3) mapping for production and decay of each subchannel resonance.  
   - Read parameters of the resonances from the Particle Data Group ([pdgLive](https://pdglive.lbl.gov/Viewer.action))
   - Express the couplings using SU(3) symmetry  
   **(2 pts)**

4. **Amplitude → Dalitz Plot**  
   Given the resonances and couplings:  
   - Produce the Dalitz plot **(2 pts)**  
   - Comment on the resonance bands’ strength and position **(1 pt)**
