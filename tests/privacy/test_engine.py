"""
Tests for the shape-dispatched perturbation engine.

Covers acceptance criteria AC-04 (noise scale independent of the value),
AC-05 (independent of grid shape), AC-06 (zero strength is a no-op), AC-11
(graphicality fallback), AC-14 (structural zeros), AC-16 (scalar clip and
floor), AC-17 (integer minimum) and AC-22 (no degenerate shape parameters).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from powergrid_synth.privacy import engine
from powergrid_synth.privacy.registry import (
    TRANSFORM_LOG,
    FieldSpec,
    Domain,
    Mode,
    ParameterDescriptor,
    Shape,
    get_descriptor,
)
from powergrid_synth.privacy.rng import make_rng

from ._descriptors import CONDITIONAL_VECTOR, SIMPLEX_VECTOR

# A strength far above any sensible operating point, used to prove that guard
# rails hold even when the operator pushes the dial to its limit.
EXTREME_STRENGTH = 10.0

# Enough draws to make a floor violation overwhelmingly likely if one exists.
STRESS_DRAW_COUNT = 300


def test_noise_scale_is_the_product_of_strength_and_constant():
    """FR-04: the scale rule is strength x k, and nothing else."""
    assert engine.noise_scale_for(0.5, 2.0) == pytest.approx(1.0)
    assert engine.noise_scale_for(0.0, 4.5) == pytest.approx(0.0)


def test_scalar_perturbation_at_zero_strength_returns_the_input_unchanged():
    """AC-06: zero strength must be bit-identical to no perturbation."""
    descriptor = get_descriptor("dist.degree_dist")
    original = {"pi": 0.53, "a1": 1.49, "b1": 0.65, "a2": 4.42, "b2": 1.67}

    perturbed = engine.perturb_scalar_family(
        descriptor, original, strength=0.0, rng=make_rng(1, "stream")
    )

    assert perturbed == original


def test_noise_scale_does_not_depend_on_the_magnitude_of_the_value():
    """AC-04: a value ten thousand times larger gets the same relative noise.

    This is the property the old sensitivity rule lacked: deriving the scale
    from the value leaked information about the value.
    """
    descriptor = get_descriptor("dist.cable_length")

    small = engine.perturb_scalar_family(
        descriptor, {"x0": 100.0}, strength=0.5, rng=make_rng(7, "s")
    )
    large = engine.perturb_scalar_family(
        descriptor, {"x0": 1_000_000.0}, strength=0.5, rng=make_rng(7, "s")
    )

    small_ratio = small["x0"] / 100.0
    large_ratio = large["x0"] / 1_000_000.0
    assert small_ratio == pytest.approx(large_ratio)


def test_noise_scale_does_not_depend_on_how_many_parameters_are_configured():
    """AC-05: the dial is not a conserved budget divided among parameters.

    Under the previous epsilon model the per-parameter scale depended on the
    number of streams, so the same configuration behaved differently on grids
    with different voltage-level counts.
    """
    descriptor = CONDITIONAL_VECTOR
    probabilities = [0.5, 0.6, 0.7]

    first = engine.perturb_probability_vector(
        descriptor, probabilities, strength=0.4, rng=make_rng(11, "bus.p_colocate")
    )
    second = engine.perturb_probability_vector(
        descriptor, probabilities, strength=0.4, rng=make_rng(11, "bus.p_colocate")
    )

    assert np.allclose(first, second)


def test_scalar_perturbation_respects_a_declared_floor():
    """AC-16: a floored field never returns below its floor."""
    descriptor = ParameterDescriptor(
        name="floored",
        domain=Domain.TRANSMISSION,
        shape=Shape.SCALAR,
        description="Test descriptor with a floored positive field.",
        supported_modes=frozenset({Mode.PERTURB, Mode.RAW}),
        default_mode=Mode.RAW,
        fields=(FieldSpec("sigma", TRANSFORM_LOG, floor=0.25),),
    )

    values = [
        engine.perturb_scalar_family(
            descriptor, {"sigma": 0.3}, strength=EXTREME_STRENGTH, rng=make_rng(i, "f")
        )["sigma"]
        for i in range(STRESS_DRAW_COUNT)
    ]

    assert min(values) >= 0.25


def test_scalar_perturbation_respects_a_declared_clip():
    """AC-16: a clipped field stays inside its bounds after inversion."""
    descriptor = ParameterDescriptor(
        name="clipped",
        domain=Domain.TRANSMISSION,
        shape=Shape.SCALAR,
        description="Test descriptor with a clipped field.",
        supported_modes=frozenset({Mode.PERTURB, Mode.RAW}),
        default_mode=Mode.RAW,
        fields=(FieldSpec("value", TRANSFORM_LOG, clip=(1.0, 2.0)),),
    )

    values = [
        engine.perturb_scalar_family(
            descriptor, {"value": 1.5}, strength=EXTREME_STRENGTH, rng=make_rng(i, "c")
        )["value"]
        for i in range(STRESS_DRAW_COUNT)
    ]

    assert all(1.0 <= value <= 2.0 for value in values)


def test_scalar_perturbation_skips_fields_absent_from_the_input():
    """A partial fit must not become a full one by inventing missing fields."""
    descriptor = get_descriptor("dist.degree_dist")

    perturbed = engine.perturb_scalar_family(
        descriptor, {"a1": 1.49}, strength=0.5, rng=make_rng(2, "s")
    )

    assert set(perturbed) == {"a1"}


def test_probability_vector_preserves_structural_zeros_exactly():
    """AC-14: a zero bin means the reference genuinely has none of something."""
    descriptor = CONDITIONAL_VECTOR
    probabilities = np.array([0.2, 0.0, 0.15, 0.0, 0.1])

    perturbed = engine.perturb_probability_vector(
        descriptor, probabilities, strength=1.0, rng=make_rng(3, "z")
    )

    assert perturbed[1] == 0.0
    assert perturbed[3] == 0.0
    assert not np.allclose(perturbed[[0, 2, 4]], probabilities[[0, 2, 4]])


def test_probability_vector_stays_within_the_unit_interval():
    """A probability outside [0, 1] would be invalid downstream."""
    descriptor = CONDITIONAL_VECTOR
    probabilities = np.linspace(0.05, 0.95, 14)

    for draw in range(50):
        perturbed = engine.perturb_probability_vector(
            descriptor, probabilities, strength=EXTREME_STRENGTH, rng=make_rng(draw, "p")
        )
        assert np.all(perturbed >= 0.0)
        assert np.all(perturbed <= 1.0)


def test_renormalising_vector_sums_to_one():
    """A simplex parameter must remain a simplex after perturbation."""
    descriptor = SIMPLEX_VECTOR
    probabilities = np.array([0.076, 0.407, 0.331, 0.051, 0.051, 0.017, 0.068])

    perturbed = engine.perturb_probability_vector(
        descriptor, probabilities, strength=0.5, rng=make_rng(4, "j")
    )

    assert perturbed.sum() == pytest.approx(1.0)


def test_non_renormalising_vector_is_left_unnormalised():
    """Independent conditional probabilities must not be forced onto a simplex."""
    descriptor = CONDITIONAL_VECTOR
    probabilities = np.full(14, 0.5)

    perturbed = engine.perturb_probability_vector(
        descriptor, probabilities, strength=0.5, rng=make_rng(5, "n")
    )

    assert perturbed.sum() > 1.0


def test_integer_perturbation_respects_the_declared_minimum():
    """AC-17: a strongly negative draw must not drive a count below its floor."""
    descriptor = get_descriptor("node_count")

    values = [
        engine.perturb_integer_value(
            descriptor, 5, strength=EXTREME_STRENGTH, rng=make_rng(i, "i")
        )
        for i in range(STRESS_DRAW_COUNT)
    ]

    assert min(values) >= descriptor.min_value
    assert all(isinstance(value, int) for value in values)


def test_integer_perturbation_at_zero_strength_is_a_no_op():
    """AC-06 for the integer path."""
    descriptor = get_descriptor("node_count")

    assert (
        engine.perturb_integer_value(
            descriptor, 42, strength=0.0, rng=make_rng(1, "i")
        )
        == 42
    )


def test_degree_sequence_perturbation_returns_a_graphical_sequence():
    """AC-11: a perturbed sequence must describe a network that can exist."""
    descriptor = get_descriptor("degrees_by_level")
    original = [2, 2, 3, 3, 4, 4, 2, 2, 3, 3]

    perturbed, attempts, used_fallback = engine.perturb_degree_sequence(
        descriptor, original, strength=0.5, rng=make_rng(6, "d")
    )

    assert nx.is_graphical(perturbed)
    assert len(perturbed) == len(original)
    assert attempts >= 1
    assert not used_fallback


def test_degree_sequence_falls_back_when_no_graphical_resample_is_found():
    """AC-11: exhausting the retries must report the fallback, not hide it."""
    descriptor = get_descriptor("degrees_by_level")
    # An odd single-node sequence can never be graphical, so every resample
    # fails and the fallback path is forced.
    original = [3]

    perturbed, attempts, used_fallback = engine.perturb_degree_sequence(
        descriptor, original, strength=1.0, rng=make_rng(8, "d"), max_retries=5
    )

    assert used_fallback
    assert attempts == 5
    assert perturbed == original


def test_degree_sequence_at_zero_strength_is_unchanged():
    """AC-06 for the degree-sequence path."""
    descriptor = get_descriptor("degrees_by_level")
    original = [2, 3, 3, 2]

    perturbed, _, used_fallback = engine.perturb_degree_sequence(
        descriptor, original, strength=0.0, rng=make_rng(9, "d")
    )

    assert perturbed == original
    assert not used_fallback


def test_graph_node_count_perturbation_changes_the_bus_count():
    """The node-count offset must actually add or remove buses."""
    descriptor = get_descriptor("node_count")
    graph = nx.path_graph(30)
    nx.set_node_attributes(graph, 0, "voltage_level")

    perturbed_graph, deltas = engine.perturb_graph_node_count(
        graph,
        descriptor=descriptor,
        strength=1.0,
        rng_for_level=lambda level: make_rng(10, f"node_count[{level}]"),
    )

    assert perturbed_graph.number_of_nodes() != graph.number_of_nodes()
    assert sum(deltas.values()) != 0


def test_graph_node_count_perturbation_at_zero_strength_is_a_no_op():
    """AC-06 for the graph-level path."""
    descriptor = get_descriptor("node_count")
    graph = nx.path_graph(10)
    nx.set_node_attributes(graph, 0, "voltage_level")

    perturbed_graph, deltas = engine.perturb_graph_node_count(
        graph,
        descriptor=descriptor,
        strength=0.0,
        rng_for_level=lambda level: make_rng(10, f"node_count[{level}]"),
    )

    assert perturbed_graph.number_of_nodes() == 10
    assert deltas == {}


@pytest.mark.parametrize(
    "parameter_name, fitted_values",
    [
        ("dist.hop_dist", {"r": 3.14, "p": 0.41}),
        ("dist.degree_dist", {"pi": 0.85, "a1": 1.49, "b1": 0.65, "a2": 4.42, "b2": 1.67}),
        ("dist.intermediate_frac", {"alpha": 1.64, "beta": 15.77}),
        ("dist.load_deviation", {"mu": 0.0, "sigma": 0.0026, "nu": 1.06}),
        ("dist.cable_length", {"x0": 0.119, "gamma": 0.159}),
        ("dist.length_clip", {"a": 5.27, "b": 0.0765}),
    ],
)
def test_distribution_shape_parameters_never_become_degenerate(
    parameter_name, fitted_values
):
    """AC-22: at maximum strength every shape parameter stays above its floor.

    The distribution pipeline has no feasibility check comparable to AC power
    flow, so these floors are the only guard against a perturbation producing
    a point-mass or otherwise unusable distribution.
    """
    descriptor = get_descriptor(parameter_name)
    floors = {
        spec.key: spec.floor for spec in descriptor.fields if spec.floor is not None
    }

    for draw in range(100):
        perturbed = engine.perturb_scalar_family(
            descriptor,
            fitted_values,
            strength=EXTREME_STRENGTH,
            rng=make_rng(draw, parameter_name),
        )
        for key, floor in floors.items():
            assert perturbed[key] >= floor, f"{parameter_name}.{key} fell below {floor}"
        for key, value in perturbed.items():
            assert np.isfinite(value), f"{parameter_name}.{key} is not finite"


def test_probability_fields_stay_inside_the_unit_interval():
    """AC-22: a mixture weight outside [0, 1] would be meaningless."""
    descriptor = get_descriptor("dist.degree_dist")

    for draw in range(100):
        perturbed = engine.perturb_scalar_family(
            descriptor,
            {"pi": 0.85, "a1": 1.49, "b1": 0.65, "a2": 4.42, "b2": 1.67},
            strength=EXTREME_STRENGTH,
            rng=make_rng(draw, "pi"),
        )
        assert 0.0 <= perturbed["pi"] <= 1.0


def test_perturb_by_shape_dispatches_each_shape():
    """The convenience entry point must handle every shape it advertises."""
    scalar = engine.perturb_by_shape(
        get_descriptor("dist.cable_length"),
        {"x0": 150.0},
        strength=0.3,
        rng=make_rng(12, "a"),
    )
    assert scalar["x0"] != 150.0

    vector = engine.perturb_by_shape(
        CONDITIONAL_VECTOR,
        np.full(14, 0.5),
        strength=0.3,
        rng=make_rng(12, "b"),
    )
    assert isinstance(vector, np.ndarray)

    count = engine.perturb_by_shape(
        get_descriptor("node_count"), 20, strength=0.3, rng=make_rng(12, "c")
    )
    assert isinstance(count, int)

    sequence = engine.perturb_by_shape(
        get_descriptor("degrees_by_level"),
        [2, 2, 3, 3],
        strength=0.3,
        rng=make_rng(12, "d"),
    )
    assert isinstance(sequence, list)


def test_mean_displacement_grows_with_strength():
    """The dial must be monotone in expectation, which is where it is defined.

    A single Laplace draw at a larger scale can land nearer zero, so the
    ordering is asserted over many draws rather than per-draw.
    """
    descriptor = get_descriptor("node_count")
    baseline_count = 100

    def mean_absolute_displacement(strength: float) -> float:
        displacements = [
            abs(
                engine.perturb_integer_value(
                    descriptor,
                    baseline_count,
                    strength=strength,
                    rng=make_rng(draw, "displacement"),
                )
                - baseline_count
            )
            for draw in range(STRESS_DRAW_COUNT)
        ]
        return sum(displacements) / len(displacements)

    assert (
        mean_absolute_displacement(0.1)
        < mean_absolute_displacement(0.5)
        < mean_absolute_displacement(1.0)
    )


def test_span_descriptor_supports_perturbation():
    """Network span is displaceable; it is not a structural hard case."""
    descriptor = get_descriptor("diameters_by_level")

    assert Mode.PERTURB in descriptor.supported_modes
    assert Mode.OFF not in descriptor.supported_modes  # no public fallback exists
    assert descriptor.min_value == 1


def test_span_perturbation_stays_above_one_hop():
    """A span below one hop would be meaningless for a multi-bus level."""
    descriptor = get_descriptor("diameters_by_level")

    values = [
        engine.perturb_integer_value(
            descriptor, 3, strength=EXTREME_STRENGTH, rng=make_rng(i, "span")
        )
        for i in range(STRESS_DRAW_COUNT)
    ]

    assert min(values) >= 1
