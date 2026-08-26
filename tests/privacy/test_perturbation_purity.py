"""
Invariants the perturbation engine must hold for every parameter shape.

These are the guarantees a caller relies on regardless of which parameter is
being protected:

* perturbation never mutates the caller's data in place, so a reference
  statistic can be reused, logged, or compared against its perturbed form;
* a seed plus a stream name fully determines the draw, and distinct stream
  names never share one.

The perturbation mathematics itself is covered by ``test_engine.py``, and the
end-to-end wiring by ``test_parity.py``.
"""

from __future__ import annotations

import numpy as np

from powergrid_synth.privacy import engine
from powergrid_synth.privacy.registry import (
    ALL_MODES,
    TRANSFORM_LOGIT,
    Domain,
    Mode,
    ParameterDescriptor,
    Shape,
    get_descriptor,
)
from powergrid_synth.privacy.rng import make_rng

# No probability-vector parameter is registered today, so the shape is
# exercised through a descriptor built here.  Declaring it locally also keeps
# these invariants independent of which parameters happen to be registered.
PROBABILITY_VECTOR = ParameterDescriptor(
    name="test.probability_vector",
    domain=Domain.TRANSMISSION,
    shape=Shape.PROB_VEC,
    description="Per-bin conditional probability, for engine invariant tests.",
    supported_modes=ALL_MODES,
    default_mode=Mode.RAW,
    transform=TRANSFORM_LOGIT,
    zero_mask=True,
)


def test_probability_vector_perturbation_does_not_mutate_its_input():
    """The caller's array must survive unchanged so it can be reused or logged."""
    original = np.linspace(0.1, 0.9, 14)
    snapshot = original.copy()

    engine.perturb_probability_vector(
        PROBABILITY_VECTOR, original, strength=0.5, rng=make_rng(1, "stream")
    )

    assert np.array_equal(original, snapshot)


def test_scalar_family_perturbation_does_not_mutate_its_input():
    """The same guarantee for scalar families."""
    descriptor = get_descriptor("dist.load_deviation")
    original = {"mu": 5.3, "sigma": 0.9, "nu": 4.0}
    snapshot = dict(original)

    engine.perturb_scalar_family(
        descriptor, original, strength=0.5, rng=make_rng(1, "stream")
    )

    assert original == snapshot


def test_identical_streams_produce_identical_noise():
    """A stream name plus a seed must fully determine the draw."""
    probabilities = np.linspace(0.05, 0.5, 14)

    first = engine.perturb_probability_vector(
        PROBABILITY_VECTOR, probabilities, strength=0.4, rng=make_rng(42, "bus.first")
    )
    second = engine.perturb_probability_vector(
        PROBABILITY_VECTOR, probabilities, strength=0.4, rng=make_rng(42, "bus.first")
    )

    assert np.array_equal(first, second)


def test_distinct_streams_produce_distinct_noise():
    """Different parameters must not share a noise draw."""
    probabilities = np.linspace(0.05, 0.5, 14)

    first = engine.perturb_probability_vector(
        PROBABILITY_VECTOR, probabilities, strength=0.4, rng=make_rng(42, "bus.first")
    )
    second = engine.perturb_probability_vector(
        PROBABILITY_VECTOR, probabilities, strength=0.4, rng=make_rng(42, "bus.second")
    )

    assert not np.array_equal(first, second)
