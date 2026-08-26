"""
Tests for the parameter registry.

Covers acceptance criteria AC-01 (registry completeness), AC-18 (domain
counts) and AC-19 (domain isolation) of the modular perturbation layer spec.
"""

from __future__ import annotations

import pytest

from powergrid_synth.privacy.registry import (
    REGISTRY,
    TRANSFORM_AFFINE_LOGIT,
    Domain,
    FieldSpec,
    Mode,
    ParameterDescriptor,
    Shape,
    descriptors_for_domain,
    get_descriptor,
)

# Every parameter family the spec requires the registry to cover.
EXPECTED_TRANSMISSION_PARAMETERS = {
    "degrees_by_level",
    "node_count",
    "diameters_by_level",
    "transformer_degrees",
    "base_kv_map",
}

EXPECTED_DISTRIBUTION_PARAMETERS = {
    "dist.hop_dist",
    "dist.degree_dist",
    "dist.degree_clip",
    "dist.intermediate_frac",
    "dist.injection_frac",
    "dist.load_deviation",
    "dist.cable_length",
    "dist.length_clip",
    "dist.input_model",
}


def test_registry_covers_every_expected_transmission_parameter():
    """AC-01: the registry contains exactly the expected transmission families."""
    registered = {
        descriptor.name for descriptor in descriptors_for_domain(Domain.TRANSMISSION)
    }
    assert registered == EXPECTED_TRANSMISSION_PARAMETERS


def test_registry_covers_every_expected_distribution_parameter():
    """AC-18: the registry contains exactly the expected distribution families."""
    registered = {
        descriptor.name for descriptor in descriptors_for_domain(Domain.DISTRIBUTION)
    }
    assert registered == EXPECTED_DISTRIBUTION_PARAMETERS


def test_every_descriptor_is_tagged_with_exactly_one_domain():
    """AC-19: domain tagging partitions the registry with no overlap."""
    transmission = set(descriptors_for_domain(Domain.TRANSMISSION))
    distribution = set(descriptors_for_domain(Domain.DISTRIBUTION))

    assert not transmission & distribution
    assert len(transmission) + len(distribution) == len(REGISTRY)


def test_every_descriptor_declares_a_supported_default_mode():
    """A default the parameter does not support would be unreachable."""
    for descriptor in REGISTRY.values():
        assert descriptor.default_mode in descriptor.supported_modes


def test_every_off_capable_parameter_names_its_public_fallback():
    """Switching a parameter off is meaningless without a documented fallback."""
    for descriptor in REGISTRY.values():
        if Mode.OFF in descriptor.supported_modes:
            assert descriptor.fallback_note, (
                f"{descriptor.name} supports 'off' but names no public fallback."
            )


def test_scalar_descriptors_declare_at_least_one_field():
    """A scalar family with no fields would perturb nothing."""
    for descriptor in REGISTRY.values():
        if descriptor.shape is Shape.SCALAR:
            assert descriptor.fields


def test_noise_constants_are_positive():
    """A non-positive constant would silently disable perturbation (FR-04)."""
    for descriptor in REGISTRY.values():
        assert descriptor.noise_k > 0
        for field_spec in descriptor.fields:
            assert field_spec.noise_k > 0, f"{descriptor.name}.{field_spec.key}"


def test_get_descriptor_rejects_unknown_name_and_lists_valid_ones():
    """FR-07: a typo must produce an actionable error, not a silent miss."""
    with pytest.raises(KeyError) as error:
        get_descriptor("degrees_by_levle")

    message = str(error.value)
    assert "degrees_by_levle" in message
    assert "degrees_by_level" in message


def test_descriptor_rejects_default_mode_outside_supported_modes():
    """A descriptor that cannot reach its own default is a construction error."""
    with pytest.raises(ValueError, match="default_mode"):
        ParameterDescriptor(
            name="broken",
            domain=Domain.TRANSMISSION,
            shape=Shape.PROB_VEC,
            description="Invalid descriptor used to check validation.",
            supported_modes=frozenset({Mode.RAW}),
            default_mode=Mode.PERTURB,
        )


def test_descriptor_rejects_affine_logit_without_bounds():
    """affine_logit is undefined without an interval to map onto."""
    with pytest.raises(ValueError, match="affine_bounds"):
        ParameterDescriptor(
            name="broken",
            domain=Domain.TRANSMISSION,
            shape=Shape.SCALAR,
            description="Invalid descriptor used to check validation.",
            supported_modes=frozenset({Mode.RAW}),
            default_mode=Mode.RAW,
            fields=(FieldSpec("value", TRANSFORM_AFFINE_LOGIT),),
        )


def test_structural_hard_cases_do_not_advertise_perturb_support():
    """Two structural quantities still have no perturbation mechanism.

    Nominal voltages cannot take arbitrary noise without producing
    non-standard levels. The transformer pattern is a *pair* of bipartite
    degree sequences bound by a conservation law -- every transformer counted
    from one level is the same transformer counted from the other -- so
    displacing each side independently would break the equality and ask the
    generator for a bipartite graph that cannot exist. Realising it needs a
    Gale-Ryser check, which the engine does not have.

    The network span is deliberately absent from this list: it is a plain
    integer target and is perturbable.
    """
    for name in ("base_kv_map", "transformer_degrees"):
        assert Mode.PERTURB not in get_descriptor(name).supported_modes


