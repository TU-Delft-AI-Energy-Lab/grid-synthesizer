"""
Tests for powergrid_synth.privacy.noise.

§8 requirements:
- Round-trip exactness for all four transforms (within 1e-9 for valid inputs).
- perturb reduces to identity when scale=0.
- Vector inputs get independent draws (statistical check with fixed seed).
- perturb_integer returns ints and is unbiased in expectation at large sample.
"""

import numpy as np
import pytest

from powergrid_synth.privacy.noise import (
    Transform,
    affine_logit,
    identity,
    log,
    logit,
    perturb,
    perturb_integer,
)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """inverse(forward(x)) ≈ x within 1e-9 for valid inputs."""

    def test_identity_scalar(self):
        for x in [-100.0, 0.0, 0.5, 42.7]:
            assert abs(identity.inverse(identity.forward(x)) - x) < 1e-9

    def test_identity_array(self):
        x = np.array([-5.0, 0.0, 3.14])
        rt = identity.inverse(identity.forward(x))
        np.testing.assert_allclose(rt, x, atol=1e-9)

    def test_log_scalar(self):
        for x in [0.001, 1.0, 2.718, 1000.0]:
            rt = log.inverse(log.forward(x))
            assert abs(rt - x) < 1e-9, f"log round-trip failed for x={x}: got {rt}"

    def test_log_array(self):
        x = np.array([0.01, 1.0, 10.0, 100.0])
        rt = log.inverse(log.forward(x))
        np.testing.assert_allclose(rt, x, atol=1e-9)

    def test_log_rejects_non_positive(self):
        with pytest.raises(ValueError, match="x > 0"):
            log.forward(0.0)
        with pytest.raises(ValueError, match="x > 0"):
            log.forward(-1.0)
        with pytest.raises(ValueError, match="x > 0"):
            log.forward(np.array([1.0, -0.5]))

    def test_logit_scalar(self):
        for p in [0.05, 0.3, 0.5, 0.7, 0.95]:
            rt = logit.inverse(logit.forward(p))
            assert abs(rt - p) < 1e-9, f"logit round-trip failed for p={p}: got {rt}"

    def test_logit_array(self):
        p = np.array([0.1, 0.4, 0.6, 0.9])
        rt = logit.inverse(logit.forward(p))
        np.testing.assert_allclose(rt, p, atol=1e-9)

    def test_logit_clamps_boundary(self):
        # Should not raise; clamps to ε_clamp internally
        v0 = logit.forward(0.0)
        v1 = logit.forward(1.0)
        assert np.isfinite(v0)
        assert np.isfinite(v1)

    def test_affine_logit_scalar(self):
        t = affine_logit(lo=0.8, hi=1.0)
        for x in [0.82, 0.85, 0.9, 0.95, 0.99]:
            rt = t.inverse(t.forward(x))
            assert abs(rt - x) < 1e-9, f"affine_logit round-trip failed for x={x}"

    def test_affine_logit_array(self):
        t = affine_logit(lo=-1.0, hi=2.0)
        x = np.array([-0.9, 0.0, 0.5, 1.9])
        rt = t.inverse(t.forward(x))
        np.testing.assert_allclose(rt, x, atol=1e-9)

    def test_affine_logit_clamps_boundary(self):
        t = affine_logit(lo=0.0, hi=1.0)
        # Exact boundary should not raise
        assert np.isfinite(t.forward(0.0))
        assert np.isfinite(t.forward(1.0))


# ---------------------------------------------------------------------------
# perturb — scale=0 reduces to identity
# ---------------------------------------------------------------------------

class TestPerturbScaleZero:

    def _rng(self):
        return np.random.default_rng(0)

    def test_scalar_identity(self):
        for x in [0.5, 3.14, -7.2]:
            result = perturb(x, transform=identity, scale=0.0, rng=self._rng())
            assert abs(result - x) < 1e-9

    def test_log_scale_zero(self):
        x = 2.5
        result = perturb(x, transform=log, scale=0.0, rng=self._rng())
        assert abs(result - x) < 1e-9

    def test_logit_scale_zero(self):
        x = 0.4
        result = perturb(x, transform=logit, scale=0.0, rng=self._rng())
        assert abs(result - x) < 1e-9

    def test_affine_logit_scale_zero(self):
        t = affine_logit(lo=0.0, hi=2.0)
        x = 1.3
        result = perturb(x, transform=t, scale=0.0, rng=self._rng())
        assert abs(result - x) < 1e-9

    def test_array_scale_zero(self):
        x = np.array([0.1, 0.5, 0.9])
        result = perturb(x, transform=logit, scale=0.0, rng=self._rng())
        np.testing.assert_allclose(result, x, atol=1e-9)


# ---------------------------------------------------------------------------
# perturb — clip parameter
# ---------------------------------------------------------------------------

class TestPerturbClip:

    def test_clip_enforced(self):
        rng = np.random.default_rng(1)
        # Large scale → result without clip would likely leave [0,1]
        results = [
            perturb(0.5, transform=identity, scale=100.0, rng=rng, clip=(0.0, 1.0))
            for _ in range(50)
        ]
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_no_clip_allows_out_of_range(self):
        rng = np.random.default_rng(2)
        seen_outside = False
        for _ in range(200):
            r = perturb(0.5, transform=identity, scale=10.0, rng=rng)
            if r < 0.0 or r > 1.0:
                seen_outside = True
                break
        assert seen_outside, "Expected some draws outside [0,1] with large scale and no clip"


# ---------------------------------------------------------------------------
# perturb — vector inputs get independent draws
# ---------------------------------------------------------------------------

class TestPerturbVectorIndependence:
    """Statistical check: perturbations of different vector elements are independent."""

    def test_vector_draws_vary(self):
        rng = np.random.default_rng(42)
        x = np.ones(100) * 5.0
        result = perturb(x, transform=identity, scale=1.0, rng=rng)
        # Each element should be independently perturbed — check variance
        assert result.std() > 0.1, "Expected non-zero variance across independent draws"

    def test_vector_draws_independent(self):
        # Draw two adjacent elements many times; they should not be perfectly correlated.
        results = []
        for i in range(500):
            rng = np.random.default_rng(i)
            x = np.array([5.0, 5.0])
            r = perturb(x, transform=identity, scale=1.0, rng=rng)
            results.append(r)
        a = np.array([r[0] for r in results])
        b = np.array([r[1] for r in results])
        corr = np.corrcoef(a, b)[0, 1]
        assert abs(corr) < 0.15, f"Elements are too correlated: ρ={corr:.3f} (expected ~0)"

    def test_scalar_return_type(self):
        rng = np.random.default_rng(0)
        r = perturb(3.0, transform=identity, scale=0.5, rng=rng)
        assert isinstance(r, float)

    def test_array_return_type(self):
        rng = np.random.default_rng(0)
        x = np.array([1.0, 2.0, 3.0])
        r = perturb(x, transform=identity, scale=0.5, rng=rng)
        assert isinstance(r, np.ndarray)
        assert r.shape == x.shape


# ---------------------------------------------------------------------------
# perturb_integer
# ---------------------------------------------------------------------------

class TestPerturbInteger:

    def test_returns_int(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            result = perturb_integer(10, scale=2.0, rng=rng)
            assert isinstance(result, int)

    def test_zero_scale_returns_original(self):
        rng = np.random.default_rng(0)
        assert perturb_integer(42, scale=0.0, rng=rng) == 42

    def test_unbiased_in_expectation(self):
        # Mean offset should be close to 0 over many samples.
        n = 5000
        base = 100
        draws = [
            perturb_integer(base, scale=3.0, rng=np.random.default_rng(i))
            for i in range(n)
        ]
        mean_offset = np.mean(draws) - base
        assert abs(mean_offset) < 0.15, (
            f"perturb_integer appears biased: mean offset = {mean_offset:.4f}"
        )

    def test_nonzero_scale_produces_variation(self):
        draws = {
            perturb_integer(50, scale=5.0, rng=np.random.default_rng(i))
            for i in range(100)
        }
        assert len(draws) > 5, "Expected varied outputs with scale=5"
