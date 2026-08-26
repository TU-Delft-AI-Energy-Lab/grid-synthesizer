"""
engine — shape-dispatched perturbation of reference-derived parameters.

The engine applies Laplace noise to a value according to its
:class:`~powergrid_synth.privacy.registry.ParameterDescriptor`.  Four shapes
are handled, each by its own function, and every one follows the same rule:

    Laplace scale = strength * noise_k

where ``noise_k`` is a constant declared on the descriptor.  The scale never
depends on the value being perturbed.  This is deliberate and is the central
correctness property of this module: a scale derived from the data itself
leaks information about the data and makes the same configuration behave
differently on different grids.

Noise is added in a transformed space chosen so the result is always valid
after inversion — logit for probabilities, log for strictly positive scale and
shape parameters, identity for quantities already on the real line.

Spec coverage: FR-04, FR-05, FR-16, FR-17, FR-18, FR-19
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .noise import Transform, affine_logit, identity, log, logit, perturb, perturb_integer
from .registry import (
    TRANSFORM_AFFINE_LOGIT,
    TRANSFORM_IDENTITY,
    TRANSFORM_LOG,
    TRANSFORM_LOGIT,
    FieldSpec,
    ParameterDescriptor,
    Shape,
)

logger = logging.getLogger(__name__)

# Keeps a probability strictly inside (0, 1) so the logit transform stays finite.
PROBABILITY_EPSILON = 1e-6

# Attempts allowed when resampling a degree sequence before giving up and
# using the unperturbed sequence (FR-19).
DEFAULT_MAX_GRAPHICALITY_RETRIES = 50


def resolve_transform(field_spec: FieldSpec) -> Transform:
    """Return the :class:`Transform` named by a field spec.

    Args:
        field_spec: The field whose transform is wanted.

    Returns:
        The matching forward/inverse transform pair.

    Raises:
        ValueError: If the transform name is not recognised.
    """
    if field_spec.transform == TRANSFORM_IDENTITY:
        return identity
    if field_spec.transform == TRANSFORM_LOG:
        return log
    if field_spec.transform == TRANSFORM_LOGIT:
        return logit
    if field_spec.transform == TRANSFORM_AFFINE_LOGIT:
        low, high = field_spec.affine_bounds
        return affine_logit(low, high)
    raise ValueError(
        f"Unknown transform {field_spec.transform!r} on field {field_spec.key!r}."
    )


def noise_scale_for(strength: float, noise_k: float) -> float:
    """Compute the Laplace scale from strength and the descriptor's constant.

    This is the whole of the scale rule (FR-04).  Note what is absent: the
    value being perturbed plays no part.

    Args:
        strength: Non-negative perturbation strength from the configuration.
        noise_k: Data-independent constant declared on the descriptor or field.

    Returns:
        Laplace scale in the transform's space.
    """
    return float(strength) * float(noise_k)


def perturb_scalar_family(
    descriptor: ParameterDescriptor,
    values: Dict[str, float],
    *,
    strength: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Perturb the named scalar fields of one parameter family (FR-17).

    Each field is transformed into its own space, displaced by Laplace noise,
    inverted, and then clamped by whatever floor and clip the field declares.
    Fields absent from *values* are skipped rather than invented, so a partial
    fit does not become a full one.

    Args:
        descriptor: Descriptor whose ``fields`` drive the perturbation.
        values: Mapping of field key to fitted value.  Not modified.
        strength: Perturbation strength.  Zero returns a copy unchanged.
        rng: Generator supplying the noise draws.

    Returns:
        A new mapping with the same keys as *values*, perturbed where a
        matching field spec exists.

    Example:
        >>> perturbed = perturb_scalar_family(
        ...     descriptor, {"sigma_cp0": 0.9}, strength=0.4, rng=rng
        ... )
    """
    perturbed = dict(values)
    if strength == 0.0:
        # FR-05: strength zero must be bit-identical to no perturbation, so
        # return before any transform round-trip can perturb the low bits.
        return perturbed

    for field_spec in descriptor.fields:
        if field_spec.key not in perturbed:
            continue
        original_value = perturbed[field_spec.key]
        if original_value is None:
            continue

        transform = resolve_transform(field_spec)
        scale = noise_scale_for(strength, field_spec.noise_k)

        prepared_value = float(original_value)
        if field_spec.transform == TRANSFORM_LOG:
            # The log transform is undefined at or below zero; a fitted value
            # that small is already degenerate, so lift it to the floor first.
            minimum = field_spec.floor or PROBABILITY_EPSILON
            prepared_value = max(prepared_value, minimum)
        elif field_spec.transform == TRANSFORM_LOGIT:
            prepared_value = float(
                np.clip(prepared_value, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
            )

        noisy_value = float(
            perturb(
                prepared_value,
                transform=transform,
                scale=scale,
                rng=rng,
                clip=field_spec.clip,
            )
        )

        if field_spec.floor is not None:
            noisy_value = max(noisy_value, field_spec.floor)

        perturbed[field_spec.key] = noisy_value

    return perturbed


def perturb_probability_vector(
    descriptor: ParameterDescriptor,
    probabilities: Sequence[float],
    *,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb a vector of probabilities in logit space (FR-16).

    Structural zeros are preserved exactly when the descriptor sets
    ``zero_mask``.  A zero bin means the reference genuinely has none of
    something in that bin; adding noise there would invent behaviour the
    reference does not exhibit, which is a fidelity error rather than a
    privacy gain.

    Args:
        descriptor: Descriptor supplying ``noise_k``, ``zero_mask`` and
            ``renormalise``.
        probabilities: The vector to perturb.  Not modified.
        strength: Perturbation strength.  Zero returns a copy unchanged.
        rng: Generator supplying the noise draws.

    Returns:
        Perturbed vector as a float array of the same length.
    """
    values = np.asarray(probabilities, dtype=float).copy()
    if strength == 0.0:
        return values

    is_structural_zero = (
        values == 0.0 if descriptor.zero_mask else np.zeros(len(values), dtype=bool)
    )
    scale = noise_scale_for(strength, descriptor.noise_k)

    for index in np.where(~is_structural_zero)[0]:
        clamped = float(
            np.clip(values[index], PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        )
        values[index] = float(
            np.clip(perturb(clamped, transform=logit, scale=scale, rng=rng), 0.0, 1.0)
        )

    if descriptor.renormalise:
        # Renormalise over the non-structural bins only, so preserved zeros
        # neither absorb nor donate probability mass.
        total = values[~is_structural_zero].sum()
        if total > 0:
            values[~is_structural_zero] /= total
        else:
            n_active = int((~is_structural_zero).sum())
            if n_active:
                values[~is_structural_zero] = 1.0 / n_active

    values[is_structural_zero] = 0.0
    return values


def perturb_integer_value(
    descriptor: ParameterDescriptor,
    value: int,
    *,
    strength: float,
    rng: np.random.Generator,
) -> int:
    """Perturb an integer count by a rounded Laplace draw (FR-18).

    Args:
        descriptor: Descriptor supplying ``noise_k`` and ``min_value``.
        value: The count to perturb.
        strength: Perturbation strength.  Zero returns *value* unchanged.
        rng: Generator supplying the noise draw.

    Returns:
        Perturbed count, never below the descriptor's ``min_value``.
    """
    if strength == 0.0:
        return int(value)

    scale = noise_scale_for(strength, descriptor.noise_k)
    noisy_value = perturb_integer(int(value), scale=scale, rng=rng)
    if descriptor.min_value is not None:
        noisy_value = max(noisy_value, descriptor.min_value)
    return int(noisy_value)


def perturb_degree_sequence(
    descriptor: ParameterDescriptor,
    degree_sequence: Sequence[int],
    *,
    strength: float,
    rng: np.random.Generator,
    max_retries: int = DEFAULT_MAX_GRAPHICALITY_RETRIES,
) -> Tuple[List[int], int, bool]:
    """Perturb a degree sequence, keeping it realisable as a network (FR-19).

    The sequence is turned into a histogram over observed degrees, the bin
    probabilities are perturbed in logit space and renormalised, and a fresh
    sequence of the same length is drawn from the result.  The draw is checked
    against the Erdos-Gallai criterion; if it fails, the draw is repeated.

    On exhausting *max_retries* the original sequence is returned unchanged.
    This is deliberate: repairing an invalid sequence with ad hoc rules would
    leave a recognisable signature an adversary could exploit, which is worse
    than an occasional unperturbed level, provided the fallback is reported.

    Args:
        descriptor: Descriptor supplying ``noise_k``.
        degree_sequence: Connection counts for one voltage level.
        strength: Perturbation strength.  Zero returns a copy unchanged.
        rng: Generator supplying the noise draws.
        max_retries: Attempts before falling back.

    Returns:
        Tuple of ``(sequence, attempts_used, used_fallback)``.
    """
    original_sequence = list(degree_sequence)
    if strength == 0.0 or not original_sequence:
        return original_sequence, 0, False

    degree_values, counts = np.unique(
        np.asarray(original_sequence, dtype=int), return_counts=True
    )
    bin_probabilities = counts.astype(float) / counts.sum()
    scale = noise_scale_for(strength, descriptor.noise_k)
    sequence_length = len(original_sequence)

    for attempt in range(max_retries):
        noisy_probabilities = bin_probabilities.copy()
        for index in range(len(noisy_probabilities)):
            clamped = float(
                np.clip(
                    noisy_probabilities[index],
                    PROBABILITY_EPSILON,
                    1.0 - PROBABILITY_EPSILON,
                )
            )
            noisy_probabilities[index] = float(
                np.clip(
                    perturb(clamped, transform=logit, scale=scale, rng=rng), 0.0, 1.0
                )
            )

        total = noisy_probabilities.sum()
        if total > 0:
            noisy_probabilities /= total
        else:
            noisy_probabilities[:] = 1.0 / len(noisy_probabilities)

        candidate = rng.choice(
            degree_values, size=sequence_length, p=noisy_probabilities
        ).tolist()
        if nx.is_graphical(candidate):
            return candidate, attempt + 1, False

    logger.warning(
        "Degree-sequence perturbation found no graphical sequence in %d attempts; "
        "using the unperturbed sequence for this level.",
        max_retries,
    )
    return original_sequence, max_retries, True


def perturb_graph_node_count(
    graph: nx.Graph,
    *,
    descriptor: ParameterDescriptor,
    strength: float,
    rng_for_level,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """Add or remove buses at each voltage level of a generated graph.

    Applied after topology generation and largest-connected-component
    filtering rather than before.  Filtering is non-linear, so noise applied
    beforehand would be distorted by it; applying it afterwards perturbs the
    graph an adversary actually observes.

    Removals take the least-connected buses first, which minimises the risk of
    severing the network.  Additions attach a new single-connection bus to a
    randomly chosen existing bus at the same level, and newly added buses
    become eligible attachment points for later additions.

    Args:
        graph: Generated topology whose nodes carry ``voltage_level``.
        descriptor: The ``node_count`` descriptor.
        strength: Perturbation strength.  Zero returns the graph unchanged.
        rng_for_level: Callable taking a level index and returning a
            :class:`numpy.random.Generator`, so each level draws from its own
            stream.

    Returns:
        Tuple of ``(perturbed_graph, per_level_delta)``.
    """
    if strength == 0.0:
        return graph, {}

    nodes_by_level: Dict[int, List] = {}
    for node, data in graph.nodes(data=True):
        level = int(data.get("voltage_level", 0))
        nodes_by_level.setdefault(level, []).append(node)

    perturbed_graph = graph.copy()
    per_level_delta: Dict[int, int] = {}

    for level in sorted(nodes_by_level):
        nodes_at_level = nodes_by_level[level]
        current_count = len(nodes_at_level)
        rng = rng_for_level(level)

        target_count = perturb_integer_value(
            descriptor, current_count, strength=strength, rng=rng
        )
        delta = target_count - current_count
        if delta == 0:
            per_level_delta[level] = 0
            continue

        if delta < 0:
            minimum_level_size = descriptor.min_value or 2
            n_to_remove = min(-delta, current_count - minimum_level_size)
            if n_to_remove <= 0:
                per_level_delta[level] = 0
                continue
            surviving = [n for n in nodes_at_level if n in perturbed_graph]
            by_ascending_degree = sorted(
                (perturbed_graph.degree(node), node) for node in surviving
            )
            perturbed_graph.remove_nodes_from(
                [node for _, node in by_ascending_degree[:n_to_remove]]
            )
            per_level_delta[level] = -n_to_remove
        else:
            existing_ids = set(perturbed_graph.nodes())
            next_id = max(existing_ids) + 1 if existing_ids else 0
            attachment_pool = [n for n in nodes_at_level if n in perturbed_graph]
            n_added = 0
            for _ in range(delta):
                while next_id in existing_ids:
                    next_id += 1
                perturbed_graph.add_node(next_id, voltage_level=level)
                if attachment_pool:
                    neighbour = attachment_pool[
                        int(rng.integers(0, len(attachment_pool)))
                    ]
                    perturbed_graph.add_edge(next_id, neighbour)
                    attachment_pool.append(next_id)
                existing_ids.add(next_id)
                next_id += 1
                n_added += 1
            per_level_delta[level] = n_added

    return perturbed_graph, per_level_delta


def perturb_by_shape(
    descriptor: ParameterDescriptor,
    value,
    *,
    strength: float,
    rng: np.random.Generator,
):
    """Dispatch to the perturbation function matching the descriptor's shape.

    Convenience entry point for callers holding a value and a descriptor
    without wanting to branch on shape themselves.  The graph-level node-count
    operation is not reachable here; call :func:`perturb_graph_node_count`
    directly for that.

    Args:
        descriptor: Descriptor for the parameter.
        value: The value to perturb, matching the descriptor's shape.
        strength: Perturbation strength.
        rng: Generator supplying the noise draws.

    Returns:
        The perturbed value, of the same type as *value*.

    Raises:
        ValueError: If the descriptor's shape has no dispatch path here.
    """
    if descriptor.shape is Shape.SCALAR:
        return perturb_scalar_family(descriptor, value, strength=strength, rng=rng)
    if descriptor.shape is Shape.PROB_VEC:
        return perturb_probability_vector(
            descriptor, value, strength=strength, rng=rng
        )
    if descriptor.shape is Shape.INTEGER:
        return perturb_integer_value(descriptor, value, strength=strength, rng=rng)
    if descriptor.shape is Shape.DEGREE_SEQ:
        sequence, _, _ = perturb_degree_sequence(
            descriptor, value, strength=strength, rng=rng
        )
        return sequence
    raise ValueError(
        f"No perturbation path for shape {descriptor.shape!r} "
        f"on parameter {descriptor.name!r}."
    )


def measure_inflation_bias(
    reference_graph: nx.Graph,
    n_samples: int = 50,
    *,
    seed: Optional[int] = None,
) -> Dict[int, float]:
    """Measure how much the topology generator inflates per-level bus counts.

    Generates *n_samples* grids from *reference_graph* without perturbation and
    returns the mean per-level difference between generated and input counts.
    Useful for calibrating the ``node_count`` noise constant against a
    particular reference grid.

    Args:
        reference_graph: Graph whose nodes carry ``voltage_level``.
        n_samples: Number of grids to generate.
        seed: Top-level seed for reproducibility.

    Returns:
        Mapping of level index to mean inflation in buses.
    """
    from ..core.input_extractor import extract_topology_params_from_graph
    from ..transmission.generator import PowerGridGenerator

    params = extract_topology_params_from_graph(reference_graph)
    degrees_by_level = params["degrees_by_level"]
    n_levels = len(degrees_by_level)
    input_counts = [len(degrees_by_level[i]) for i in range(n_levels)]
    inflation_per_level: List[List[float]] = [[] for _ in range(n_levels)]

    seed_generator = np.random.default_rng(seed)
    for _ in range(n_samples):
        trial_seed = int(seed_generator.integers(0, 2**31))
        generated = PowerGridGenerator(seed=trial_seed).generate_grid(
            degrees_by_level=degrees_by_level,
            diameters_by_level=params["diameters_by_level"],
            transformer_degrees=params["transformer_degrees"],
            keep_lcc=False,
        )
        for level in range(n_levels):
            level_size = sum(
                1
                for _, data in generated.nodes(data=True)
                if data.get("voltage_level") == level
            )
            inflation_per_level[level].append(float(level_size - input_counts[level]))

    return {
        level: float(np.mean(inflation_per_level[level])) for level in range(n_levels)
    }
