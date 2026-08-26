"""
rng.py — stream-keyed RNG factory with seed-contract guarantees.

Two determinism properties operate at different scopes:

§5.1  Within-run guarantee (mandatory)
    The noise draw for a given parameter happens ONCE per reference grid per
    run, keyed by (resolved_seed, stream).  It must NOT be re-drawn per
    synthetic sample.  All N synthetic samples in one run share the same
    perturbed parameters.

§5.2  Across-run policy (default: random each run)
    When seed=None, each synthesizer run draws fresh noise.  Pass a fixed int
    to make re-runs from the same reference carry the same perturbation.

⚠️  PRIVACY ASSUMPTION — single-release-per-reference:
    Random-per-run seeding is sound only when you publish at most one batch
    of synthetic grids per reference grid.  See the perturbation config for
    the full statement.
"""

from __future__ import annotations

import hashlib

import numpy as np


def resolve_seed(seed: int | None) -> int:
    """Return a concrete top-level seed (draw fresh entropy when seed is None).

    Call this once at run start and pass the resolved int to ``make_rng`` so
    that the resolved seed can be recorded in the run report for
    reproducibility.
    """
    if seed is None:
        return int(np.random.SeedSequence().entropy)
    return int(seed)


def make_rng(seed: int | None, stream: str) -> np.random.Generator:
    """Return a reproducible, stream-independent Generator.

    Parameters
    ----------
    seed : int or None
        Top-level seed.  ``None`` triggers a fresh random draw (see §5.2).
        Pass ``resolve_seed(seed)`` here after recording the resolved value.
    stream : str
        Logical name for this noise channel (e.g. ``"cp0_mu"``,
        ``"node_count"``, ``"degrees_by_level[level=2]"``).  Different stream
        names yield *independent* generators from the same top-level seed,
        satisfying the within-run once-per-parameter guarantee (§5.1).

    Returns
    -------
    np.random.Generator
        A fresh Generator for this (seed, stream) pair.  Re-creating it with
        the same (seed, stream) returns an identical sequence — callers that
        need the guarantee of §5.1 should create the generator once and pass
        it to ``perturb`` rather than calling ``make_rng`` again for each
        sample.
    """
    top_seed = resolve_seed(seed)

    # Derive a stable, stream-specific child seed via SHA-256.
    stream_hash = int(
        hashlib.sha256(stream.encode()).hexdigest(), 16
    ) % (2 ** 32)

    ss = np.random.SeedSequence([top_seed, stream_hash])
    return np.random.default_rng(ss)
