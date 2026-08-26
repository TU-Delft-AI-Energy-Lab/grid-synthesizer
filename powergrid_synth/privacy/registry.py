"""
registry — declarative inventory of every reference-derived parameter.

This module is the single source of truth for what PowerGridSynth reads from a
reference network.  Each entry is a :class:`ParameterDescriptor` naming one
family of information, the shape it takes, how it may be perturbed, and which
modes it supports.

The registry spans both synthesis pipelines.  Descriptors are tagged with a
:class:`Domain`, and each pipeline resolves only descriptors of its own domain,
so a transmission run never touches a distribution parameter and vice versa.

Design note — why descriptors hold no extractor callable
--------------------------------------------------------
Extraction stays at its existing call sites in the two ``synthesize`` modules.
The registry supplies metadata and the perturbation contract; the pipeline asks
:func:`should_extract` before calling an extractor and hands the result to the
engine afterwards.  ``mode="off"`` therefore works by *not calling* the
extractor at all, leaving the downstream allocator to use the public built-in
table it already falls back to when no reference value is supplied.  This keeps
the public-fallback behaviour in one place rather than duplicating it here.

Spec coverage: FR-01, FR-02, FR-13, FR-13a
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


class Domain(str, Enum):
    """Which synthesis pipeline a parameter belongs to (FR-13a)."""

    TRANSMISSION = "transmission"
    DISTRIBUTION = "distribution"


class Shape(str, Enum):
    """How a parameter is laid out, which selects the engine's perturbation path."""

    SCALAR = "scalar"          # one or more independent named scalar fields
    PROB_VEC = "prob_vec"      # a vector of probabilities
    INTEGER = "integer"        # a single integer count
    DEGREE_SEQ = "degree_seq"  # a per-level sequence of connection counts


class Mode(str, Enum):
    """What the pipeline does with a parameter (FR-02)."""

    OFF = "off"          # never read from the reference; use the public fallback
    PERTURB = "perturb"  # read from the reference and add noise
    RAW = "raw"          # read from the reference and use unchanged


# Transform names resolved against powergrid_synth.privacy.noise by the engine.
TRANSFORM_IDENTITY = "identity"
TRANSFORM_LOG = "log"
TRANSFORM_LOGIT = "logit"
TRANSFORM_AFFINE_LOGIT = "affine_logit"

# Smallest permitted value for any scale/shape parameter.  Without a floor a
# strong perturbation can drive sigma or a gamma shape to zero, collapsing the
# distribution to a point mass (spec Risk 7).
SHAPE_PARAMETER_FLOOR = 1e-3


@dataclass(frozen=True)
class FieldSpec:
    """One named scalar field inside a parameter family.

    A family such as a generator cost model or a negative-binomial hop
    distribution carries several scalars whose valid domains differ, so each
    field declares its own transform and guard rails.

    Args:
        key: Dictionary key or dataclass attribute holding this field.
        transform: One of the ``TRANSFORM_*`` names.  Selects the space in
            which Laplace noise is added.
        noise_k: Data-independent noise constant for this field.  The Laplace
            scale is ``strength * noise_k`` in transformed space (FR-04).
        floor: Lower bound applied after inversion, or ``None``.  Used for
            scale and shape parameters that must stay strictly positive.
        clip: ``(low, high)`` bound applied after inversion, or ``None``.
        affine_bounds: ``(low, high)`` interval for ``affine_logit``.  Required
            when ``transform`` is ``affine_logit``, ignored otherwise.
    """

    key: str
    transform: str = TRANSFORM_IDENTITY
    noise_k: float = 1.0
    floor: Optional[float] = None
    clip: Optional[Tuple[float, float]] = None
    affine_bounds: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class ParameterDescriptor:
    """Declarative description of one reference-derived parameter family.

    Args:
        name: Stable identifier.  Doubles as the config key and the RNG stream
            name, so changing it changes the noise draw and breaks
            reproducibility against earlier runs.
        domain: Which pipeline consumes this parameter (FR-13a).
        shape: Selects the engine's perturbation path.
        description: Plain-language summary, reproduced in the run report.
        supported_modes: Modes this parameter accepts.  Requesting any other
            mode is a hard error (FR-08).
        default_mode: Mode used when the config does not mention this
            parameter.  Chosen so that an absent config reproduces current
            behaviour exactly (FR-12).
        fallback_note: What is used instead when the mode is ``off``.  Recorded
            for the run report and the audit trail.
        fields: Per-field specs for ``SCALAR`` families.  Empty for the other
            shapes, which carry a single value.
        transform: Transform for non-scalar shapes.
        noise_k: Noise constant for non-scalar shapes (FR-04).
        zero_mask: Preserve structural zeros exactly for ``PROB_VEC``.  A zero
            means the reference genuinely has none of something; adding noise
            there would invent behaviour the reference does not exhibit.
        renormalise: Rescale a ``PROB_VEC`` to sum to one after perturbation.
            Set for true simplexes, cleared for independent per-bin
            conditional probabilities.
        min_value: Lower clamp for ``INTEGER`` shapes.
    """

    name: str
    domain: Domain
    shape: Shape
    description: str
    supported_modes: frozenset
    default_mode: Mode
    fallback_note: str = ""
    fields: Tuple[FieldSpec, ...] = ()
    transform: str = TRANSFORM_IDENTITY
    noise_k: float = 1.0
    zero_mask: bool = False
    renormalise: bool = False
    min_value: Optional[int] = None

    def __post_init__(self) -> None:
        if self.default_mode not in self.supported_modes:
            raise ValueError(
                f"Descriptor {self.name!r}: default_mode {self.default_mode.value!r} "
                f"is not in supported_modes "
                f"{sorted(mode.value for mode in self.supported_modes)}."
            )
        if self.shape is Shape.SCALAR and not self.fields:
            raise ValueError(
                f"Descriptor {self.name!r}: SCALAR shape requires at least one FieldSpec."
            )
        for field_spec in self.fields:
            if (
                field_spec.transform == TRANSFORM_AFFINE_LOGIT
                and field_spec.affine_bounds is None
            ):
                raise ValueError(
                    f"Descriptor {self.name!r}, field {field_spec.key!r}: "
                    "affine_logit requires affine_bounds."
                )


# Mode sets used repeatedly below.
ALL_MODES = frozenset({Mode.OFF, Mode.PERTURB, Mode.RAW})
# Some quantities must never be read from a supplied reference at all, however
# the operator configures the run.  Restricting the mode set makes that a
# configuration error rather than a default that can be overridden.
OFF_ONLY = frozenset({Mode.OFF})
OFF_OR_RAW = frozenset({Mode.OFF, Mode.RAW})
PERTURB_OR_RAW = frozenset({Mode.PERTURB, Mode.RAW})
RAW_ONLY = frozenset({Mode.RAW})






# ---------------------------------------------------------------------------
# Transmission descriptors — spec Appendix A.1 and A.2
# ---------------------------------------------------------------------------

_TRANSMISSION_DESCRIPTORS = (
    ParameterDescriptor(
        name="degrees_by_level",
        domain=Domain.TRANSMISSION,
        shape=Shape.DEGREE_SEQ,
        description="Connection-count pattern within each voltage level.",
        supported_modes=PERTURB_OR_RAW,
        default_mode=Mode.RAW,
        fallback_note="No public fallback: reference-mode topology cannot be built without it.",
        transform=TRANSFORM_LOGIT,
    ),
    ParameterDescriptor(
        name="node_count",
        domain=Domain.TRANSMISSION,
        shape=Shape.INTEGER,
        description="Number of buses at each voltage level.",
        supported_modes=PERTURB_OR_RAW,
        default_mode=Mode.RAW,
        fallback_note="No public fallback: implied by the generated topology.",
        # 4.5 buses reproduces the previous default of
        # node_count_scale_multiplier (3.0) x node_count_bias (1.5).
        noise_k=4.5,
        min_value=2,
    ),
    ParameterDescriptor(
        name="diameters_by_level",
        domain=Domain.TRANSMISSION,
        shape=Shape.INTEGER,
        description="Network span (diameter) within each voltage level.",
        supported_modes=PERTURB_OR_RAW,
        default_mode=Mode.RAW,
        fallback_note="No public fallback: required by the topology generator.",
        # Spans are small integers (single digits to low tens), so the bus-count
        # constant would swamp them.  A strength of 1.0 gives a Laplace scale of
        # 2 hops, which is a visible displacement without flattening the level.
        noise_k=2.0,
        # A span of 1 is the smallest meaningful extent for a level with more
        # than one bus.  A span of 0 means the level is a single bus or empty,
        # and is preserved rather than perturbed -- see the call site.
        min_value=1,
    ),
    ParameterDescriptor(
        name="transformer_degrees",
        domain=Domain.TRANSMISSION,
        shape=Shape.DEGREE_SEQ,
        description="Transformer connection pattern between voltage levels.",
        supported_modes=RAW_ONLY,
        default_mode=Mode.RAW,
        fallback_note="No public fallback: required to couple voltage levels.",
    ),
    ParameterDescriptor(
        name="base_kv_map",
        domain=Domain.TRANSMISSION,
        shape=Shape.SCALAR,
        description="Nominal voltage levels of the reference grid, in kV.",
        supported_modes=OFF_OR_RAW,
        default_mode=Mode.RAW,
        fallback_note="Generic level map {0: 380, 1: 110, 2: 20, 3: 0.4, 4: 0.12} kV.",
        # Present so the family validates as SCALAR; perturbation is out of
        # scope because arbitrary noise would produce non-standard voltages.
        fields=(FieldSpec("kv", TRANSFORM_LOG),),
    ),
    # --- Appendix A.2: dormant extractors, default off -----------------
)


# ---------------------------------------------------------------------------
# Distribution descriptors — spec Appendix A.3
# ---------------------------------------------------------------------------

_DISTRIBUTION_DESCRIPTORS = (
    ParameterDescriptor(
        name="dist.hop_dist",
        domain=Domain.DISTRIBUTION,
        shape=Shape.SCALAR,
        description="Negative-binomial hop-distance distribution for feeder depth.",
        supported_modes=ALL_MODES,
        default_mode=Mode.RAW,
        fallback_note="Schweitzer et al. (2017) NegBinomParams defaults.",
        fields=(
            FieldSpec("r", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
            FieldSpec("p", TRANSFORM_LOGIT),
        ),
    ),
    ParameterDescriptor(
        name="dist.degree_dist",
        domain=Domain.DISTRIBUTION,
        shape=Shape.SCALAR,
        description="Mixture-of-gammas degree distribution for feeder branching.",
        supported_modes=ALL_MODES,
        default_mode=Mode.RAW,
        fallback_note="Schweitzer et al. (2017) MixtureGammaParams defaults.",
        fields=(
            FieldSpec("pi", TRANSFORM_LOGIT),
            FieldSpec("a1", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
            FieldSpec("b1", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
            FieldSpec("a2", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
            FieldSpec("b2", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
        ),
    ),
    ParameterDescriptor(
        name="dist.degree_clip",
        domain=Domain.DISTRIBUTION,
        shape=Shape.SCALAR,
        description="Power-law clipping of maximum degree against hop distance.",
        supported_modes=ALL_MODES,
        default_mode=Mode.RAW,
        fallback_note="Schweitzer et al. (2017) PowerLawClip defaults.",
        fields=(
            FieldSpec("a", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
            # The exponent is negative by construction, so it cannot take a
            # log transform; identity noise keeps it on the real line.
            FieldSpec("b", TRANSFORM_IDENTITY, noise_k=0.2),
        ),
    ),
    ParameterDescriptor(
        name="dist.intermediate_frac",
        domain=Domain.DISTRIBUTION,
        shape=Shape.SCALAR,
        description="Beta distribution for the fraction of intermediate nodes.",
        supported_modes=ALL_MODES,
        default_mode=Mode.RAW,
        fallback_note="Schweitzer et al. (2017) BetaParams(1.64, 15.77).",
        fields=(
            FieldSpec("alpha", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
            FieldSpec("beta", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
        ),
    ),
    ParameterDescriptor(
        name="dist.injection_frac",
        domain=Domain.DISTRIBUTION,
        shape=Shape.SCALAR,
        description="Beta distribution for the fraction of injection (generation) nodes.",
        supported_modes=ALL_MODES,
        default_mode=Mode.RAW,
        fallback_note="Schweitzer et al. (2017) BetaParams(0.92, 20.53).",
        fields=(
            FieldSpec("alpha", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
            FieldSpec("beta", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
        ),
    ),
    ParameterDescriptor(
        name="dist.load_deviation",
        domain=Domain.DISTRIBUTION,
        shape=Shape.SCALAR,
        description="t-location-scale distribution of load deviation from the feeder mean.",
        supported_modes=ALL_MODES,
        default_mode=Mode.RAW,
        fallback_note="Schweitzer et al. (2017) TLocationScaleParams defaults.",
        fields=(
            FieldSpec("mu", TRANSFORM_IDENTITY, noise_k=0.01),
            FieldSpec("sigma", TRANSFORM_LOG, floor=1e-6),
            # Degrees of freedom: below ~1 the distribution loses its mean.
            FieldSpec("nu", TRANSFORM_LOG, floor=0.5),
        ),
    ),
    ParameterDescriptor(
        name="dist.cable_length",
        domain=Domain.DISTRIBUTION,
        shape=Shape.SCALAR,
        description="Modified-Cauchy distribution of cable segment lengths, in km.",
        supported_modes=ALL_MODES,
        default_mode=Mode.RAW,
        fallback_note="Schweitzer et al. (2017) ModifiedCauchyParams defaults.",
        fields=(
            FieldSpec("x0", TRANSFORM_LOG, floor=1e-4),
            FieldSpec("gamma", TRANSFORM_LOG, floor=1e-4),
        ),
    ),
    ParameterDescriptor(
        name="dist.length_clip",
        domain=Domain.DISTRIBUTION,
        shape=Shape.SCALAR,
        description="Exponential clipping of maximum cable length against hop distance.",
        supported_modes=ALL_MODES,
        default_mode=Mode.RAW,
        fallback_note="Schweitzer et al. (2017) ExponentialClip defaults.",
        fields=(
            FieldSpec("a", TRANSFORM_LOG, floor=SHAPE_PARAMETER_FLOOR),
            FieldSpec("b", TRANSFORM_LOG, floor=1e-6),
        ),
    ),
    ParameterDescriptor(
        name="dist.input_model",
        domain=Domain.DISTRIBUTION,
        shape=Shape.SCALAR,
        description="Kernel-density model of feeder size and loading.",
                # Not extracted by the pipeline: setting this to raw or perturb
        # produces a byte-identical grid, so only off has a truthful
        # meaning. The consuming allocator still accepts the argument this
        # would feed, so wiring it is a contained change if wanted.
        supported_modes=OFF_ONLY,
        default_mode=Mode.OFF,
        fallback_note="Caller-supplied feeder counts and sizes.",
        # Declared so the family validates; a kernel density estimate is not a
        # scalar and its perturb mode is out of scope.
        fields=(FieldSpec("bandwidth", TRANSFORM_LOG),),
    ),
)


REGISTRY: Dict[str, ParameterDescriptor] = {
    descriptor.name: descriptor
    for descriptor in _TRANSMISSION_DESCRIPTORS + _DISTRIBUTION_DESCRIPTORS
}


def get_descriptor(name: str) -> ParameterDescriptor:
    """Return the descriptor registered under *name*.

    Args:
        name: Registered parameter name.

    Returns:
        The matching :class:`ParameterDescriptor`.

    Raises:
        KeyError: If *name* is not registered.  The message lists every valid
            name so a typo in a config file is immediately actionable (FR-07).
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown parameter {name!r}. Registered parameters: "
            f"{', '.join(sorted(REGISTRY))}."
        ) from None


def descriptors_for_domain(domain: Domain) -> Tuple[ParameterDescriptor, ...]:
    """Return every descriptor belonging to *domain*, ordered by name (FR-13a).

    Args:
        domain: The pipeline whose parameters are wanted.

    Returns:
        Tuple of descriptors, sorted by name for stable reporting.
    """
    return tuple(
        sorted(
            (d for d in REGISTRY.values() if d.domain is domain),
            key=lambda descriptor: descriptor.name,
        )
    )
