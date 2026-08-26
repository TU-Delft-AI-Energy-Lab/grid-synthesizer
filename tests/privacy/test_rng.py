"""
Tests for powergrid_synth.privacy.rng.

§8 requirements:
- same (seed, stream) ⇒ identical draw (once-per-reference-grid guarantee).
- different streams ⇒ independent draws.
- The critical assertion that a parameter perturbed twice under the same seed
  is identical (no per-sample re-draw).
"""

import numpy as np
import pytest

from powergrid_synth.privacy.rng import make_rng, resolve_seed
from powergrid_synth.privacy.noise import perturb, identity


# ---------------------------------------------------------------------------
# resolve_seed
# ---------------------------------------------------------------------------

class TestResolveSeed:

    def test_int_returned_unchanged(self):
        assert resolve_seed(42) == 42
        assert resolve_seed(0) == 0

    def test_none_returns_int(self):
        s = resolve_seed(None)
        assert isinstance(s, int)
        assert s >= 0

    def test_none_gives_fresh_seed_each_call(self):
        s1 = resolve_seed(None)
        s2 = resolve_seed(None)
        # Very unlikely to collide; if they do the test is flaky — acceptable
        assert s1 != s2


# ---------------------------------------------------------------------------
# make_rng — same (seed, stream) ⇒ identical sequence
# ---------------------------------------------------------------------------

class TestMakeRngReproducibility:

    def test_same_seed_stream_identical_draw(self):
        rng1 = make_rng(42, "test_stream")
        rng2 = make_rng(42, "test_stream")
        assert rng1.uniform() == rng2.uniform()

    def test_same_seed_stream_identical_sequence(self):
        draws1 = make_rng(42, "param_mu").standard_normal(20)
        draws2 = make_rng(42, "param_mu").standard_normal(20)
        np.testing.assert_array_equal(draws1, draws2)

    def test_different_seeds_differ(self):
        r1 = make_rng(1, "stream").uniform()
        r2 = make_rng(2, "stream").uniform()
        assert r1 != r2


# ---------------------------------------------------------------------------
# make_rng — different streams ⇒ independent draws
# ---------------------------------------------------------------------------

class TestMakeRngStreamIndependence:

    def test_different_streams_differ(self):
        r_a = make_rng(42, "stream_a").uniform()
        r_b = make_rng(42, "stream_b").uniform()
        assert r_a != r_b

    def test_many_streams_all_distinct(self):
        streams = [f"param_{i}" for i in range(20)]
        draws = [make_rng(99, s).uniform() for s in streams]
        # All 20 draws should be distinct
        assert len(set(draws)) == len(draws), "Some stream draws collide — not independent"

    def test_stream_correlation_near_zero(self):
        """Statistical independence: stream_a and stream_b draws should be uncorrelated."""
        n = 1000
        a = [make_rng(i, "stream_a").uniform() for i in range(n)]
        b = [make_rng(i, "stream_b").uniform() for i in range(n)]
        corr = float(np.corrcoef(a, b)[0, 1])
        assert abs(corr) < 0.07, (
            f"stream_a and stream_b are correlated: ρ={corr:.4f}; expected ~0"
        )


# ---------------------------------------------------------------------------
# Critical: once-per-reference-grid guarantee (§5.1)
#
# A parameter perturbed twice under the same seed must yield the IDENTICAL
# result.  This enforces the rule that within one run, all N synthetic
# samples share the same perturbed parameters (no per-sample re-draw).
# ---------------------------------------------------------------------------

class TestOncePerRunGuarantee:

    def test_same_seed_same_perturbation(self):
        """make_rng(42, 'cp0_mu') called twice produces identical draws."""
        value = 5.0
        rng1 = make_rng(42, "cp0_mu")
        rng2 = make_rng(42, "cp0_mu")
        p1 = perturb(value, transform=identity, scale=1.0, rng=rng1)
        p2 = perturb(value, transform=identity, scale=1.0, rng=rng2)
        assert p1 == p2, (
            f"Same (seed=42, stream='cp0_mu') must yield identical perturbation; "
            f"got p1={p1}, p2={p2}"
        )

    def test_n_samples_share_one_perturbation(self):
        """Within a run, all N samples must use the SAME perturbed parameter.

        Simulate how Plans 1 & 2 will work: perturb the parameter ONCE before
        the synthesis loop, then use the perturbed value for every sample.
        The perturbed value must equal what you'd get from a fresh rng with
        the same seed.
        """
        n_samples = 50
        base_param = 7.0
        run_seed = 1234

        # Step 1: perturb the parameter once (as the synthesizer should do)
        rng = make_rng(run_seed, "mu_param")
        perturbed_once = perturb(base_param, transform=identity, scale=0.5, rng=rng)

        # Step 2: all N samples in this run use perturbed_once
        synthetic_samples = [perturbed_once] * n_samples

        # Step 3: verify each sample matches a fresh rng seeded identically
        reference_rng = make_rng(run_seed, "mu_param")
        reference_value = perturb(base_param, transform=identity, scale=0.5, rng=reference_rng)

        assert all(s == reference_value for s in synthetic_samples), (
            "Not all samples used the same perturbed parameter — per-sample re-draw detected"
        )

    def test_different_run_seeds_differ(self):
        """Two runs with different seeds produce different perturbations."""
        value = 5.0
        p1 = perturb(value, transform=identity, scale=1.0, rng=make_rng(1, "stream"))
        p2 = perturb(value, transform=identity, scale=1.0, rng=make_rng(2, "stream"))
        assert p1 != p2

    def test_none_seed_different_each_call(self):
        """seed=None produces different perturbations across calls (random-per-run)."""
        value = 5.0
        results = [
            perturb(value, transform=identity, scale=1.0, rng=make_rng(None, "stream"))
            for _ in range(20)
        ]
        # At least some should differ (with overwhelming probability)
        assert len(set(results)) > 1, (
            "seed=None should produce fresh noise each run, but all results were identical"
        )


# ---------------------------------------------------------------------------
# Seed-contract integration
# ---------------------------------------------------------------------------

class TestSeedContract:

    def test_fixed_seed_reproducible(self):
        """A seed fixed in the configuration must reproduce the same draw."""
        from powergrid_synth.privacy.settings import resolve

        settings = resolve({"strength": 0.5, "seed": 77})
        rng1 = make_rng(settings.seed, "param")
        rng2 = make_rng(settings.seed, "param")
        assert rng1.uniform() == rng2.uniform()

    def test_none_seed_random(self):
        """An omitted seed must draw fresh entropy on every run."""
        from powergrid_synth.privacy.settings import resolve

        settings = resolve({"strength": 0.5})
        assert settings.seed is None
        # resolve_seed(None) gives a fresh value each time
        s1 = resolve_seed(settings.seed)
        s2 = resolve_seed(settings.seed)
        assert s1 != s2
