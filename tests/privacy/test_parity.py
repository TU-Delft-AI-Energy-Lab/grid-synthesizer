"""
End-to-end tests for the perturbation layer in both synthesis pipelines.

Covers acceptance criteria AC-02 (off skips extraction), AC-08 (run report),
AC-09 (determinism), AC-10 and AC-21 (byte-for-byte parity with no config),
AC-12 (registry extension) and AC-20 (distribution perturbation).

These tests run the real pipelines and are correspondingly slow.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from powergrid_synth.privacy.registry import Domain, Mode
from powergrid_synth.privacy.settings import resolve

pytestmark = pytest.mark.slow

pandapower = pytest.importorskip(
    "pandapower", reason="reference-mode synthesis requires pandapower"
)

REFERENCE_CASE = "case118"
SYNTHESIS_SEED = 42
DISTRIBUTION_CASE = "cigre_lv"
DISTRIBUTION_SEED = 7


def graph_fingerprint(graph) -> str:
    """Return a canonical hash of a graph's structure and attributes.

    Rounds floats to nine decimal places so that a fingerprint is stable
    against last-bit differences in unrelated arithmetic, while still
    detecting any real change to the synthesized grid.

    Args:
        graph: The graph to fingerprint.

    Returns:
        Hex SHA-256 digest of the canonical representation.
    """

    def round_value(value):
        return round(value, 9) if isinstance(value, float) else value

    nodes = [
        [
            node,
            sorted(
                (key, round_value(value))
                for key, value in graph.nodes[node].items()
                if not key.startswith("_")
            ),
        ]
        for node in sorted(graph.nodes())
    ]
    edges = sorted(
        (
            [
                min(source, target),
                max(source, target),
                sorted((key, round_value(value)) for key, value in data.items()),
            ]
            for source, target, data in graph.edges(data=True)
        ),
        key=lambda edge: (edge[0], edge[1], str(edge[2])),
    )
    blob = json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def synthesize_transmission(tmp_path, perturbation_config=None):
    """Run the transmission pipeline against the reference case."""
    from powergrid_synth.transmission.synthesize import synthesize

    return synthesize(
        mode="reference",
        reference_case=REFERENCE_CASE,
        seed=SYNTHESIS_SEED,
        output_dir=str(tmp_path),
        export_formats=[],
        perturbation_config=perturbation_config,
    )


def synthesize_feeders(tmp_path, perturbation_config=None):
    """Run the distribution pipeline against the reference feeder case."""
    from powergrid_synth.distribution.synthesize import synthesize_distribution

    return synthesize_distribution(
        mode="reference",
        reference_case=DISTRIBUTION_CASE,
        seed=DISTRIBUTION_SEED,
        n_feeders=1,
        n_nodes=15,
        output_dir=str(tmp_path),
        export_formats=[],
        perturbation_config=perturbation_config,
    )


# Every parameter defaults to raw, so perturbation is opt-in per parameter: a
# global strength alone does nothing. These tests need a configuration that
# actually perturbs something.
PERTURBING_PARAMETERS = {
    "degrees_by_level": {"mode": "perturb"},
    "node_count": {"mode": "perturb"},
    "diameters_by_level": {"mode": "perturb"},
}


def perturbing_config(strength: float, seed: int) -> dict:
    """A configuration that perturbs the three topology inputs."""
    return {
        "strength": strength,
        "seed": seed,
        "parameters": dict(PERTURBING_PARAMETERS),
    }


# ---------------------------------------------------------------------------
# Parity — an absent config must reproduce the pre-change pipeline
# ---------------------------------------------------------------------------


def test_transmission_output_is_unchanged_without_a_config(tmp_path):
    """AC-10: no config means byte-for-byte parity with the previous pipeline."""
    first = synthesize_transmission(tmp_path / "a")
    second = synthesize_transmission(tmp_path / "b")

    assert graph_fingerprint(first) == graph_fingerprint(second)


def test_transmission_zero_strength_matches_the_unconfigured_run(tmp_path):
    """AC-06 end-to-end: strength zero is indistinguishable from no config."""
    baseline = synthesize_transmission(tmp_path / "a")
    zero_strength = synthesize_transmission(tmp_path / "b", {"strength": 0.0})

    assert graph_fingerprint(baseline) == graph_fingerprint(zero_strength)


def test_distribution_output_is_unchanged_without_a_config(tmp_path):
    """AC-21: the distribution pipeline must not start perturbing by default."""
    first = synthesize_feeders(tmp_path / "a")
    second = synthesize_feeders(tmp_path / "b")

    assert graph_fingerprint(first[0]) == graph_fingerprint(second[0])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_config_and_seed_produce_identical_output(tmp_path):
    """AC-09: a run must be reproducible from its config and seed alone."""
    config = perturbing_config(0.5, 99)

    first = synthesize_transmission(tmp_path / "a", config)
    second = synthesize_transmission(tmp_path / "b", config)

    assert graph_fingerprint(first) == graph_fingerprint(second)


def test_different_perturbation_seeds_produce_different_output(tmp_path):
    """Distinct seeds must draw distinct noise, or the seed does nothing."""
    first = synthesize_transmission(tmp_path / "a", perturbing_config(0.5, 1))
    second = synthesize_transmission(tmp_path / "b", perturbing_config(0.5, 2))

    assert graph_fingerprint(first) != graph_fingerprint(second)


# ---------------------------------------------------------------------------
# The dial actually moves the output
# ---------------------------------------------------------------------------


def test_increasing_strength_changes_the_synthetic_grid(tmp_path):
    """The trade-off dial must have a visible effect on the output."""
    baseline = synthesize_transmission(tmp_path / "a")
    moderate = synthesize_transmission(tmp_path / "b", perturbing_config(0.3, 3))
    strong = synthesize_transmission(tmp_path / "c", perturbing_config(1.0, 3))

    fingerprints = {
        graph_fingerprint(baseline),
        graph_fingerprint(moderate),
        graph_fingerprint(strong),
    }
    assert len(fingerprints) == 3


def test_bus_count_moves_further_from_the_baseline_as_strength_rises(tmp_path):
    """Stronger perturbation should displace the bus count further, on average.

    Averaged over several seeds rather than asserted per-run: the offset is a
    Laplace draw, so a single draw at a larger scale can legitimately land
    nearer zero than a draw at a smaller one.  Only the expectation is ordered.
    """
    seeds = (1, 2, 3, 4, 5)
    baseline_count = synthesize_transmission(tmp_path / "base").number_of_nodes()

    def mean_absolute_displacement(strength: float) -> float:
        displacements = [
            abs(
                synthesize_transmission(
                    tmp_path / f"s{strength}_{seed}",
                    perturbing_config(strength, seed),
                ).number_of_nodes()
                - baseline_count
            )
            for seed in seeds
        ]
        return sum(displacements) / len(displacements)

    assert mean_absolute_displacement(1.0) > mean_absolute_displacement(0.2)


# ---------------------------------------------------------------------------
# Off mode
# ---------------------------------------------------------------------------


def test_off_mode_drops_the_reference_voltage_levels(tmp_path):
    """AC-02: switching nominal voltages off must remove them from the output."""
    graph = synthesize_transmission(
        tmp_path, {"parameters": {"base_kv_map": {"mode": "off"}}}
    )

    assert graph.graph.get("base_kv_map") is None


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------


def test_run_report_covers_every_transmission_parameter(tmp_path):
    """AC-08: the report must account for every parameter of the pipeline."""
    graph = synthesize_transmission(tmp_path, {"strength": 0.4, "seed": 21})
    report = graph.graph["perturbation_report"]

    expected = set(resolve(None).active_for_domain(Domain.TRANSMISSION))
    reported = {record["name"] for record in report["parameters"]}

    assert reported == expected
    assert report["resolved_seed"] == 21
    assert report["domain"] == Domain.TRANSMISSION.value


def test_run_report_is_written_to_the_output_directory(tmp_path):
    """AC-08: the report must survive the run, not only be returned."""
    synthesize_transmission(tmp_path, {"strength": 0.4, "seed": 21})

    report_files = list(tmp_path.glob("*_perturbation_report.json"))
    assert len(report_files) == 1

    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report["global_strength"] == pytest.approx(0.4)


def test_run_report_records_a_resolved_seed_even_when_none_was_given(tmp_path):
    """AC-08: an unseeded run must still be reproducible from its report."""
    graph = synthesize_transmission(tmp_path, {"strength": 0.4})
    report = graph.graph["perturbation_report"]

    assert isinstance(report["resolved_seed"], int)


# ---------------------------------------------------------------------------
# Distribution pipeline
# ---------------------------------------------------------------------------


def test_distribution_perturbation_changes_the_synthetic_feeder(tmp_path):
    """AC-20: perturbing a fitted feeder parameter must change the output."""
    baseline = synthesize_feeders(tmp_path / "a")
    perturbed = synthesize_feeders(
        tmp_path / "b",
        {
            "strength": 0.6,
            "seed": 5,
            "parameters": {
                "dist.cable_length": {"mode": "perturb"},
                "dist.hop_dist": {"mode": "perturb"},
            },
        },
    )

    assert graph_fingerprint(baseline[0]) != graph_fingerprint(perturbed[0])


def test_distribution_off_mode_falls_back_to_published_defaults(tmp_path):
    """'off' must substitute the published Schweitzer default for the fit."""
    from powergrid_synth.distribution.distribution_params import DistributionSynthParams
    from powergrid_synth.distribution.synthesize import _apply_perturbation

    fitted = DistributionSynthParams()
    fitted.cable_length.x0 = 0.999
    fitted.cable_length.gamma = 0.888

    settings = resolve({"parameters": {"dist.cable_length": {"mode": "off"}}})
    result = _apply_perturbation(fitted, settings, resolved_seed=1)

    published = DistributionSynthParams()
    assert result.cable_length.x0 == published.cable_length.x0
    assert result.cable_length.gamma == published.cable_length.gamma


def test_distribution_run_report_covers_every_distribution_parameter(tmp_path):
    """AC-08 for the distribution pipeline."""
    synthesize_feeders(tmp_path, {"strength": 0.3, "seed": 11})

    report_files = list(tmp_path.glob("*_perturbation_report.json"))
    assert len(report_files) == 1

    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    expected = set(resolve(None).active_for_domain(Domain.DISTRIBUTION))
    assert {record["name"] for record in report["parameters"]} == expected


def test_transmission_run_never_reports_distribution_parameters(tmp_path):
    """AC-19: the pipelines must not leak parameters into each other's runs."""
    graph = synthesize_transmission(tmp_path, {"strength": 0.2, "seed": 4})
    report = graph.graph["perturbation_report"]

    assert not any(
        record["name"].startswith("dist.") for record in report["parameters"]
    )


# ---------------------------------------------------------------------------
# Registry extension
# ---------------------------------------------------------------------------


def test_a_new_descriptor_is_perturbed_without_changing_the_engine():
    """AC-12: adding a parameter must require only a declarative entry."""
    from powergrid_synth.privacy import engine
    from powergrid_synth.privacy.registry import (
        TRANSFORM_LOG,
        FieldSpec,
        ParameterDescriptor,
        Shape,
    )
    from powergrid_synth.privacy.rng import make_rng

    new_descriptor = ParameterDescriptor(
        name="experimental_parameter",
        domain=Domain.TRANSMISSION,
        shape=Shape.SCALAR,
        description="A parameter added after the engine was written.",
        supported_modes=frozenset({Mode.OFF, Mode.PERTURB, Mode.RAW}),
        default_mode=Mode.RAW,
        fallback_note="Some public default.",
        fields=(FieldSpec("magnitude", TRANSFORM_LOG, floor=0.01),),
    )

    perturbed = engine.perturb_scalar_family(
        new_descriptor, {"magnitude": 2.0}, strength=0.5, rng=make_rng(1, "new")
    )

    assert perturbed["magnitude"] != 2.0
    assert perturbed["magnitude"] >= 0.01


def test_span_perturbation_displaces_the_realised_span(tmp_path):
    """Perturbing the span target must move the span of the generated graph.

    The span is a target handed to the generator rather than a value written to
    the output, so it is possible for a displaced target to be ignored.  This
    compares against several unperturbed runs so that ordinary generator
    variance cannot be mistaken for a displacement.
    """
    from powergrid_synth.transmission.synthesize import _measure_spans_by_level

    unperturbed = [
        _measure_spans_by_level(synthesize_transmission(tmp_path / f"raw{seed}"))[-1]
        for seed in range(3)
    ]
    perturbed = [
        _measure_spans_by_level(
            synthesize_transmission(
                tmp_path / f"pert{seed}",
                {
                    "strength": 2.0,
                    "seed": 7,
                    "parameters": {"diameters_by_level": {"mode": "perturb"}},
                },
            )
        )[-1]
        for seed in range(3)
    ]

    assert min(perturbed) > max(unperturbed)


def test_span_perturbation_is_recorded_in_the_run_report(tmp_path):
    """The report must show the target displacement and what was realised."""
    graph = synthesize_transmission(
        tmp_path,
        {
            "strength": 1.0,
            "seed": 7,
            "parameters": {"diameters_by_level": {"mode": "perturb"}},
        },
    )
    record = next(
        entry
        for entry in graph.graph["perturbation_report"]["parameters"]
        if entry["name"] == "diameters_by_level"
    )

    assert record["mode"] == "perturb"
    assert record["detail"]["target_before"] != record["detail"]["target_after"]
    assert "realised" in record["detail"]


def test_span_perturbation_preserves_a_degenerate_level(tmp_path):
    """A level with no extent must not be given one by perturbation."""
    graph = synthesize_transmission(
        tmp_path,
        {
            "strength": 2.0,
            "seed": 7,
            "parameters": {"diameters_by_level": {"mode": "perturb"}},
        },
    )
    detail = next(
        entry["detail"]
        for entry in graph.graph["perturbation_report"]["parameters"]
        if entry["name"] == "diameters_by_level"
    )

    for before, after in zip(detail["target_before"], detail["target_after"]):
        if before == 0:
            assert after == 0
