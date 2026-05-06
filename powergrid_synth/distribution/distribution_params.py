"""
Parameter definitions for the Schweitzer et al. (2017) distribution feeder
synthesis algorithm.

All statistical distribution parameters from Table III of the paper are
encoded here, along with cable library entries and clipping function
parameters. The :class:`DistributionSynthParams` dataclass bundles them
into a single configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Individual distribution parameter containers
# ---------------------------------------------------------------------------

@dataclass
class NegBinomParams:
    """Negative Binomial distribution parameters for hop-distance assignment."""
    r: float = 3.14
    p: float = 0.41


@dataclass
class MixtureGammaParams:
    """Mixture of two Gamma distributions for degree assignment."""
    pi: float = 0.85
    a1: float = 1.49
    b1: float = 0.65
    a2: float = 4.42
    b2: float = 1.67


@dataclass
class BetaParams:
    """Beta distribution parameters (used for fractions)."""
    alpha: float = 1.0
    beta: float = 1.0


@dataclass
class MixturePoissonParams:
    """Mixture of two Poisson distributions for intermediate node hop."""
    pi: float = 0.67
    mu1: float = 1.54
    mu2: float = 4.82


@dataclass
class MixtureNormalParams:
    """Mixture of two Normal distributions for injection node norm. hop."""
    pi: float = 0.73
    mu1: float = 0.14
    sigma1: float = 0.10
    mu2: float = 0.60
    sigma2: float = 0.18


@dataclass
class NormalParams:
    """Normal distribution parameters."""
    mu: float = 0.0
    sigma: float = 0.053


@dataclass
class TLocationScaleParams:
    """t-Location-Scale distribution for load deviation from uniform."""
    mu: float = 0.0
    sigma: float = 0.0026
    nu: float = 1.06


@dataclass
class ExponentialParams:
    """Exponential distribution parameters (e.g., I_est / I_nom ratio)."""
    mu: float = 0.31


@dataclass
class ModifiedCauchyParams:
    """Modified Cauchy distribution for cable lengths."""
    x0: float = 0.119   # km
    gamma: float = 0.159  # km


@dataclass
class PowerLawClip:
    """Power-law clipping function: g_dmax(h) = a * h^b."""
    a: float = 10.58
    b: float = -0.75


@dataclass
class ExponentialClip:
    """Exponential clipping function for max cable length: g_max(h) = a * e^(b*h)."""
    a: float = 5.27   # km
    b: float = 0.0765


@dataclass
class CurrentThreshold:
    """Max nominal current threshold for cables far from the source."""
    h_min: int = 8       # hops at which threshold activates
    i_nom_max: float = 450.0  # Amperes


@dataclass
class CableLibraryEntry:
    """A single cable type in the library."""
    name: str
    r_ohm_per_km: float
    x_ohm_per_km: float
    c_nf_per_km: float
    max_i_ka: float
    frequency: float  # relative frequency of occurrence (0–1)


# ---------------------------------------------------------------------------
# Default cable library (representative MV underground cables from Dutch data)
# ---------------------------------------------------------------------------

DEFAULT_CABLE_LIBRARY: List[CableLibraryEntry] = [
    CableLibraryEntry("Cu50",   0.387, 0.094, 280.0, 0.170, 0.05),
    CableLibraryEntry("Cu95",   0.193, 0.085, 330.0, 0.245, 0.05),
    CableLibraryEntry("Al150",  0.206, 0.079, 370.0, 0.295, 0.20),
    CableLibraryEntry("Al240",  0.125, 0.075, 410.0, 0.380, 0.35),
    CableLibraryEntry("Al400",  0.078, 0.072, 460.0, 0.480, 0.25),
    CableLibraryEntry("Al630",  0.049, 0.069, 510.0, 0.590, 0.10),
]

# ---------------------------------------------------------------------------
# Default empirical power factor CDF (from Table IV of the paper)
# Pairs of (power_factor, cumulative_probability).
# ---------------------------------------------------------------------------

DEFAULT_PF_CDF: List[tuple] = [
    (0.85, 0.02),
    (0.90, 0.10),
    (0.92, 0.20),
    (0.94, 0.35),
    (0.95, 0.50),
    (0.96, 0.65),
    (0.97, 0.80),
    (0.98, 0.92),
    (0.99, 0.97),
    (1.00, 1.00),
]


# ---------------------------------------------------------------------------
# Aggregate parameter container
# ---------------------------------------------------------------------------

@dataclass
class DistributionSynthParams:
    """All parameters for the Schweitzer distribution feeder generator.

    Default values are taken from Table III and the text of
    Schweitzer et al. (2017).
    """
    # Step 1: Node generation
    hop_dist: NegBinomParams = field(default_factory=NegBinomParams)
    pf_cdf: List[tuple] = field(default_factory=lambda: list(DEFAULT_PF_CDF))

    # Step 2: Feeder connection
    degree_dist: MixtureGammaParams = field(default_factory=MixtureGammaParams)
    degree_clip: PowerLawClip = field(default_factory=PowerLawClip)

    # Step 3a: Intermediate nodes
    intermediate_frac: BetaParams = field(
        default_factory=lambda: BetaParams(alpha=1.64, beta=15.77)
    )
    intermediate_hop: MixturePoissonParams = field(
        default_factory=MixturePoissonParams
    )

    # Step 3b: Injection (generation) nodes
    injection_frac: BetaParams = field(
        default_factory=lambda: BetaParams(alpha=0.92, beta=20.53)
    )
    injection_hop: MixtureNormalParams = field(
        default_factory=MixtureNormalParams
    )
    injection_deviation: NormalParams = field(default_factory=NormalParams)

    # Step 3c: Load (consumption) nodes
    load_deviation: TLocationScaleParams = field(
        default_factory=TLocationScaleParams
    )

    # Step 4: Cable type
    cable_library: List[CableLibraryEntry] = field(
        default_factory=lambda: list(DEFAULT_CABLE_LIBRARY)
    )
    current_ratio: ExponentialParams = field(default_factory=ExponentialParams)
    current_threshold: CurrentThreshold = field(
        default_factory=CurrentThreshold
    )

    # Step 5: Cable length
    cable_length: ModifiedCauchyParams = field(
        default_factory=ModifiedCauchyParams
    )
    length_clip: ExponentialClip = field(default_factory=ExponentialClip)
