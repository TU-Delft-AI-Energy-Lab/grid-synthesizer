"""
noise.py — transform pairs and Laplace mechanism draws.

All perturbation happens in a transformed space where additive noise is
well-behaved, then the inverse maps back and an optional clip is applied.

Transform pairs, as forward / inverse::

    identity      x                        / y
                  (ℝ, used for LogNormal μ)
    log           ln(x)                    / exp(y)
                  (x > 0, e.g. LogNormal σ)
    logit         logit(p)                 / sigmoid(y)
                  (p ∈ (0,1), probabilities)
    affine_logit  logit((x-lo)/(hi-lo))    / lo + (hi-lo)*sigmoid(y)
                  (x ∈ (lo,hi), bounded params)
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Tuple, Union

import numpy as np

_EPS_CLAMP = 1e-6


class Transform(NamedTuple):
    """A (forward, inverse) callable pair for a monotone reparameterisation."""
    forward: Callable
    inverse: Callable


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def _identity_fwd(x):
    return np.asarray(x, dtype=float)


def _identity_inv(y):
    return np.asarray(y, dtype=float)


identity: Transform = Transform(forward=_identity_fwd, inverse=_identity_inv)


# ---------------------------------------------------------------------------
# log / exp  (x > 0 required)
# ---------------------------------------------------------------------------

def _log_fwd(x):
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError(
            "log transform requires x > 0; got values <= 0. "
            "Caller must guarantee positivity before applying this transform."
        )
    return np.log(x)


def _log_inv(y):
    return np.exp(np.asarray(y, dtype=float))


log: Transform = Transform(forward=_log_fwd, inverse=_log_inv)


# ---------------------------------------------------------------------------
# logit / sigmoid  (p ∈ (0,1), clamped at ε_clamp to avoid ±inf)
# ---------------------------------------------------------------------------

def _logit_fwd(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS_CLAMP, 1.0 - _EPS_CLAMP)
    return np.log(p / (1.0 - p))


def _logit_inv(y):
    y = np.asarray(y, dtype=float)
    return 1.0 / (1.0 + np.exp(-y))


logit: Transform = Transform(forward=_logit_fwd, inverse=_logit_inv)


# ---------------------------------------------------------------------------
# affine_logit  (x ∈ (lo, hi), clamped)
# ---------------------------------------------------------------------------

def affine_logit(lo: float, hi: float) -> Transform:
    """Return a Transform for a bounded interval (lo, hi).

    Forward:  logit((x - lo) / (hi - lo))
    Inverse:  lo + (hi - lo) * sigmoid(y)
    """
    def _fwd(x):
        x = np.asarray(x, dtype=float)
        p = np.clip((x - lo) / (hi - lo), _EPS_CLAMP, 1.0 - _EPS_CLAMP)
        return np.log(p / (1.0 - p))

    def _inv(y):
        y = np.asarray(y, dtype=float)
        return lo + (hi - lo) / (1.0 + np.exp(-y))

    return Transform(forward=_fwd, inverse=_inv)


# ---------------------------------------------------------------------------
# Laplace perturbation
# ---------------------------------------------------------------------------

def perturb(
    value,
    *,
    transform: Transform,
    scale: float,
    rng: np.random.Generator,
    clip: Optional[Tuple[float, float]] = None,
) -> Union[float, np.ndarray]:
    """Apply Laplace noise in the transformed space and invert.

    Parameters
    ----------
    value : scalar or array_like
        The parameter to perturb.
    transform : Transform
        (forward, inverse) pair defining the noise space.
    scale : float
        Laplace scale parameter b (not ε). Use ``budget.epsilon_to_scale``
        to convert ε → b before calling this function.
    rng : np.random.Generator
        Pre-seeded generator from ``rng.make_rng``. For vector inputs, each
        element gets an independent draw — renormalisation (for simplex
        constraints) is the caller's responsibility.
    clip : (lo, hi) or None
        Optional clip applied to the result in the original space.

    Returns
    -------
    float or np.ndarray
        Perturbed value in the original space.
    """
    value = np.asarray(value, dtype=float)
    scalar = value.ndim == 0

    z = transform.forward(value)

    if scale == 0.0:
        # Zero scale: noise is identically 0; return the exact round-trip result.
        result = transform.inverse(z)
    else:
        size = z.shape if z.ndim > 0 else None
        noise = rng.laplace(loc=0.0, scale=scale, size=size)
        result = transform.inverse(z + noise)

    if clip is not None:
        result = np.clip(result, clip[0], clip[1])

    if scalar:
        return float(result)
    return np.asarray(result)


# ---------------------------------------------------------------------------
# Discrete Laplace (integer targets)
# ---------------------------------------------------------------------------

def perturb_integer(value: int, *, scale: float, rng: np.random.Generator) -> int:
    """Discrete Laplace perturbation for integer-valued targets.

    v1 implementation: rounds a continuous Laplace draw, which approximates
    the true discrete Laplace (two-sided geometric) distribution and is
    sufficient for the node-count use case (Mechanism B).
    """
    if scale == 0.0:
        return int(value)
    noise = rng.laplace(loc=0.0, scale=scale)
    return int(round(value + noise))
