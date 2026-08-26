"""
Descriptors used only by the tests.

No probability-vector parameter is registered today: the shapes the engine
supports are deliberately wider than the current parameter set, so that adding
a parameter is a registry change rather than an engine change.  Declaring the
vector descriptors here exercises those paths without tying the engine tests to
whichever parameters happen to be registered.
"""

from __future__ import annotations

from powergrid_synth.privacy.registry import (
    ALL_MODES,
    TRANSFORM_LOGIT,
    Domain,
    Mode,
    ParameterDescriptor,
    Shape,
)

#: Independent per-bin conditional probabilities: structural zeros are
#: preserved and the vector is not forced onto a simplex.
CONDITIONAL_VECTOR = ParameterDescriptor(
    name="test.conditional_vector",
    domain=Domain.TRANSMISSION,
    shape=Shape.PROB_VEC,
    description="Per-bin conditional probability, for engine tests.",
    supported_modes=ALL_MODES,
    default_mode=Mode.RAW,
    transform=TRANSFORM_LOGIT,
    zero_mask=True,
)

#: A true simplex: perturbation must leave it summing to one.
SIMPLEX_VECTOR = ParameterDescriptor(
    name="test.simplex_vector",
    domain=Domain.TRANSMISSION,
    shape=Shape.PROB_VEC,
    description="Joint distribution over categories, for engine tests.",
    supported_modes=ALL_MODES,
    default_mode=Mode.RAW,
    transform=TRANSFORM_LOGIT,
    renormalise=True,
)
