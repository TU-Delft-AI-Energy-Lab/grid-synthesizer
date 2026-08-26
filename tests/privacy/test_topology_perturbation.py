"""
Migrated tests for topology perturbation.

The original suite for this area tested the epsilon budget and the stream
accounting, both removed with the differential-privacy framing.  What survives
is kept here:

* the node-count offset must be symmetric, adding buses about as often as it
  removes them, so it does not systematically inflate or shrink the grid;
* perturbation must not mutate the caller's sequences or graph in place.

Graphicality, floors and dispatch are covered by ``test_engine.py``; the
end-to-end wiring by ``test_parity.py``.
"""

from __future__ import annotations

import networkx as nx
import pytest

from powergrid_synth.privacy import engine
from powergrid_synth.privacy.registry import get_descriptor
from powergrid_synth.privacy.rng import make_rng

# Enough draws for the symmetry check to be meaningful without being slow.
SYMMETRY_DRAW_COUNT = 400

# A symmetric mechanism should split close to evenly; this bound is loose
# enough not to flake and tight enough to catch a one-sided offset.
SYMMETRY_TOLERANCE = 0.20


def test_node_count_offsets_are_symmetric():
    """The offset must not systematically grow or shrink the grid.

    A one-sided offset would be a fidelity bug and a privacy one: a consistent
    direction is itself information about the mechanism.
    """
    descriptor = get_descriptor("node_count")
    baseline_count = 200

    offsets = [
        engine.perturb_integer_value(
            descriptor, baseline_count, strength=1.0, rng=make_rng(draw, "symmetry")
        )
        - baseline_count
        for draw in range(SYMMETRY_DRAW_COUNT)
    ]

    increases = sum(1 for offset in offsets if offset > 0)
    decreases = sum(1 for offset in offsets if offset < 0)
    moved = increases + decreases
    assert moved > 0

    assert abs(increases - decreases) / moved < SYMMETRY_TOLERANCE


def test_degree_sequence_perturbation_does_not_mutate_its_input():
    """The caller's sequence must survive so the original stays available."""
    descriptor = get_descriptor("degrees_by_level")
    original = [2, 2, 3, 3, 4, 4]
    snapshot = list(original)

    engine.perturb_degree_sequence(
        descriptor, original, strength=0.5, rng=make_rng(1, "stream")
    )

    assert original == snapshot


def test_degree_sequence_perturbation_preserves_length():
    """Bus count is perturbed separately; the sequence step must not change it."""
    descriptor = get_descriptor("degrees_by_level")
    original = [2, 2, 3, 3, 4, 4, 2, 2]

    perturbed, _, _ = engine.perturb_degree_sequence(
        descriptor, original, strength=0.5, rng=make_rng(2, "stream")
    )

    assert len(perturbed) == len(original)


def test_graph_node_count_perturbation_does_not_mutate_the_input_graph():
    """The caller's graph must be left intact; a copy is returned."""
    descriptor = get_descriptor("node_count")
    graph = nx.path_graph(30)
    nx.set_node_attributes(graph, 0, "voltage_level")
    original_node_count = graph.number_of_nodes()

    engine.perturb_graph_node_count(
        graph,
        descriptor=descriptor,
        strength=1.0,
        rng_for_level=lambda level: make_rng(3, f"node_count[{level}]"),
    )

    assert graph.number_of_nodes() == original_node_count


def test_node_count_perturbation_never_empties_a_voltage_level():
    """A level reduced to nothing would break downstream allocation."""
    descriptor = get_descriptor("node_count")

    for draw in range(50):
        graph = nx.path_graph(4)
        nx.set_node_attributes(graph, 0, "voltage_level")

        perturbed_graph, _ = engine.perturb_graph_node_count(
            graph,
            descriptor=descriptor,
            strength=10.0,
            rng_for_level=lambda level, draw=draw: make_rng(draw, f"lvl{level}"),
        )

        assert perturbed_graph.number_of_nodes() >= descriptor.min_value


def test_added_buses_carry_their_voltage_level():
    """A bus without a level would be invisible to later allocation steps."""
    descriptor = get_descriptor("node_count")
    graph = nx.path_graph(20)
    nx.set_node_attributes(graph, 2, "voltage_level")

    perturbed_graph, deltas = engine.perturb_graph_node_count(
        graph,
        descriptor=descriptor,
        strength=2.0,
        rng_for_level=lambda level: make_rng(9, f"node_count[{level}]"),
    )

    assert all(
        "voltage_level" in data for _, data in perturbed_graph.nodes(data=True)
    )
    if deltas.get(2, 0) > 0:
        assert perturbed_graph.number_of_nodes() > 20
