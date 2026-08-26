"""
Tests for perturbation configuration loading, validation and resolution.

Covers acceptance criteria AC-03 (override resolution), AC-07 (loud failure on
bad configuration) and AC-15 (TOML via the standard library) of the modular
perturbation layer spec.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from powergrid_synth.privacy.registry import REGISTRY, Domain, Mode
from powergrid_synth.privacy.settings import (
    MAX_STRENGTH,
    PerturbationConfigError,
    PerturbationSettings,
    coerce,
    load,
    resolve,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_PATH = REPOSITORY_ROOT / "perturbation.example.toml"


def test_resolve_without_config_covers_every_registered_parameter():
    """The report must be complete, so every parameter resolves even unmentioned."""
    settings = resolve(None)
    assert set(settings.per_parameter) == set(REGISTRY)


def test_global_strength_applies_to_unmentioned_parameters():
    """AC-03: the global value reaches every parameter set to perturb.

    Every parameter defaults to raw, so a mode must be set for the strength to
    have anything to act on. What this checks is that the *global* value is
    picked up without being repeated per parameter.
    """
    settings = resolve(
        {
            "strength": 0.7,
            "parameters": {
                "degrees_by_level": {"mode": "perturb"},
                "node_count": {"mode": "perturb"},
            },
        }
    )

    assert settings.strength_of("degrees_by_level") == pytest.approx(0.7)
    assert settings.strength_of("node_count") == pytest.approx(0.7)


def test_per_parameter_strength_overrides_the_global_value():
    """AC-03: an override wins, and does not disturb its neighbours."""
    settings = resolve(
        {
            "strength": 0.4,
            "parameters": {
                "diameters_by_level": {"mode": "perturb", "strength": 1.0},
                "degrees_by_level": {"mode": "perturb"},
            },
        }
    )

    assert settings.strength_of("diameters_by_level") == pytest.approx(1.0)
    assert settings.strength_of("degrees_by_level") == pytest.approx(0.4)


def test_strength_is_zeroed_outside_perturb_mode():
    """Strength is meaningless when not perturbing; zeroing simplifies callers."""
    settings = resolve(
        {"strength": 0.8, "parameters": {"diameters_by_level": {"mode": "raw"}}}
    )

    assert settings.strength_of("diameters_by_level") == 0.0
    assert not settings.should_perturb("diameters_by_level")


def test_off_mode_reports_that_extraction_should_be_skipped():
    """AC-02 precondition: 'off' must tell the pipeline not to read the value."""
    settings = resolve({"parameters": {"base_kv_map": {"mode": "off"}}})

    assert settings.mode_of("base_kv_map") is Mode.OFF
    assert not settings.should_extract("base_kv_map")


def test_unknown_parameter_name_is_rejected_with_valid_names_listed():
    """FR-07: a silently ignored key is a silent loss of protection."""
    with pytest.raises(PerturbationConfigError) as error:
        resolve({"parameters": {"degrees_by_levell": {"mode": "raw"}}})

    message = str(error.value)
    assert "degrees_by_levell" in message
    assert "degrees_by_level" in message


def test_unsupported_mode_is_rejected_and_names_the_supported_modes():
    """FR-08: asking for perturb on a hard case must fail loudly."""
    with pytest.raises(PerturbationConfigError) as error:
        resolve({"parameters": {"base_kv_map": {"mode": "perturb"}}})

    message = str(error.value)
    assert "base_kv_map" in message
    assert "off" in message and "raw" in message


def test_unrecognised_mode_name_is_rejected():
    """A misspelled mode must not fall through to a default."""
    with pytest.raises(PerturbationConfigError, match="is not a mode"):
        resolve({"parameters": {"degrees_by_level": {"mode": "pertrub"}}})


@pytest.mark.parametrize("bad_strength", [-1.0, MAX_STRENGTH + 0.1])
def test_out_of_range_strength_is_rejected(bad_strength):
    """FR-09: a strength outside the permitted range is an error."""
    with pytest.raises(PerturbationConfigError, match="between"):
        resolve({"strength": bad_strength})


@pytest.mark.parametrize("bad_strength", ["strong", True, None])
def test_non_numeric_strength_is_rejected(bad_strength):
    """FR-09: strength must be a number; booleans are not numbers here."""
    with pytest.raises(PerturbationConfigError, match="must be a number"):
        resolve({"strength": bad_strength})


def test_unknown_top_level_key_is_rejected():
    """A stray key usually means a misplaced setting, not an intention."""
    with pytest.raises(PerturbationConfigError, match="Unknown top-level key"):
        resolve({"strenght": 0.4})


def test_unknown_key_inside_a_parameter_table_is_rejected():
    """Only mode and strength are meaningful inside a parameter table."""
    with pytest.raises(PerturbationConfigError, match="unknown key"):
        resolve({"parameters": {"degrees_by_level": {"mode": "raw", "epsilon": 0.5}}})


def test_non_integer_seed_is_rejected():
    """A float seed would be silently truncated by the generator."""
    with pytest.raises(PerturbationConfigError, match="seed"):
        resolve({"seed": 1.5})


def test_active_for_domain_returns_only_that_pipelines_parameters():
    """AC-19: each pipeline sees only its own parameters."""
    settings = resolve(None)

    transmission = settings.active_for_domain(Domain.TRANSMISSION)
    distribution = settings.active_for_domain(Domain.DISTRIBUTION)

    assert all(name.startswith("dist.") for name in distribution)
    assert not any(name.startswith("dist.") for name in transmission)
    assert len(transmission) + len(distribution) == len(REGISTRY)


def test_example_config_parses_and_resolves_every_parameter():
    """AC-15: the shipped example must load with stdlib tomllib."""
    with EXAMPLE_CONFIG_PATH.open("rb") as handle:
        raw = tomllib.load(handle)

    settings = resolve(raw, source=str(EXAMPLE_CONFIG_PATH))
    assert set(settings.per_parameter) == set(REGISTRY)


def test_load_reads_a_toml_file_from_disk(tmp_path):
    """AC-15: configuration arrives from a file, not only from a mapping."""
    config_file = tmp_path / "perturbation.toml"
    config_file.write_text(
        'strength = 0.25\nseed = 7\n\n[parameters.diameters_by_level]\nmode = "perturb"\n',
        encoding="utf-8",
    )

    settings = load(config_file)

    assert settings.seed == 7
    assert settings.strength_of("diameters_by_level") == pytest.approx(0.25)


def test_load_rejects_a_missing_file(tmp_path):
    """A mistyped path must not silently fall back to defaults."""
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "absent.toml")


def test_load_rejects_malformed_toml(tmp_path):
    """A broken file must fail at load time, before synthesis starts."""
    config_file = tmp_path / "broken.toml"
    config_file.write_text("strength = = 0.4\n", encoding="utf-8")

    with pytest.raises(PerturbationConfigError, match="not valid TOML"):
        load(config_file)


def test_coerce_accepts_the_supported_configuration_forms(tmp_path):
    """The entry points accept a path, a settings object, a mapping, or None."""
    config_file = tmp_path / "perturbation.toml"
    config_file.write_text("strength = 0.1\n", encoding="utf-8")

    assert isinstance(coerce(None), PerturbationSettings)
    assert isinstance(coerce({"strength": 0.2}), PerturbationSettings)
    assert isinstance(coerce(config_file), PerturbationSettings)
    assert isinstance(coerce(str(config_file)), PerturbationSettings)

    already_resolved = resolve({"strength": 0.3})
    assert coerce(already_resolved) is already_resolved


def test_coerce_rejects_an_unsupported_type():
    """An unexpected type usually means a mis-wired call site."""
    with pytest.raises(PerturbationConfigError, match="Unsupported"):
        coerce(42)
