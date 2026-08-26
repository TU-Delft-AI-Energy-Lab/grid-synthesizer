"""
powergrid_synth.privacy.recovery
==================================
Adversarial recovery estimators consumed by the privacy evaluation notebook.

Each estimator answers the same question from the defender's side: pooling a
number of synthetic grids, how close does an adversary get to the reference
value that was handed to the synthesizer? A recovery error that stays high as
the pool grows is the evidence that a perturbed parameter is protected.

Public API
----------
recover_node_count(graphs, k) -> dict[int, dict]
    Per-level node count mean + 95% CI.

recover_degree_dist(graphs, true_degrees_by_level) -> dict[int, float]
    Per-level KS distance between pooled degree distribution and reference.

recover_degree_dist_full(graphs, true_degrees_by_level) -> float
    Full-graph KS distance (all voltage levels combined).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import scipy.stats as scipy_stats


def recover_node_count(graphs: List, k: int) -> Dict[int, Dict]:
    """
    Estimate node count per voltage level from a pool of synthetic graphs.

    Returns
    -------
    dict[int, dict]
        ``{level: {mean, ci_lo, ci_hi, counts}}``
    """
    counts_by_level: Dict[int, List[int]] = {i: [] for i in range(k)}
    for G in graphs:
        level_tally: Dict[int, int] = {i: 0 for i in range(k)}
        for _, d in G.nodes(data=True):
            lvl = d.get("voltage_level")
            if lvl is not None and lvl in level_tally:
                level_tally[lvl] += 1
        for i in range(k):
            counts_by_level[i].append(level_tally[i])

    result: Dict[int, Dict] = {}
    for lvl, cnts in counts_by_level.items():
        arr = np.array(cnts, dtype=float)
        n = len(arr)
        mean = float(arr.mean())
        if n > 1:
            se = arr.std(ddof=1) / np.sqrt(n)
            ci_lo = float(mean - 1.96 * se)
            ci_hi = float(mean + 1.96 * se)
        else:
            ci_lo = ci_hi = mean
        result[lvl] = {
            "mean":   mean,
            "ci_lo":  ci_lo,
            "ci_hi":  ci_hi,
            "counts": [int(c) for c in cnts],
        }
    return result


def recover_degree_dist(
    graphs: List,
    true_degrees_by_level: List[List[int]],
) -> Dict[int, float]:
    """
    KS distance between pooled empirical degree distribution and the reference
    distribution for each voltage level.

    Returns
    -------
    dict[int, float]
        ``{level: ks_distance}``
    """
    k = len(true_degrees_by_level)
    result: Dict[int, float] = {}
    for lvl in range(k):
        ref_degs = np.array(true_degrees_by_level[lvl], dtype=float)
        pooled: List[int] = []
        for G in graphs:
            nodes_lvl = [n for n, d in G.nodes(data=True) if d.get("voltage_level") == lvl]
            sub = G.subgraph(nodes_lvl)
            pooled.extend(deg for _, deg in sub.degree())
        if pooled and len(ref_degs) > 0:
            stat, _ = scipy_stats.ks_2samp(ref_degs, np.array(pooled, dtype=float))
            result[lvl] = float(stat)
        else:
            result[lvl] = float("nan")
    return result


def recover_degree_dist_full(
    graphs: List,
    true_degrees_by_level: List[List[int]],
) -> float:
    """
    KS distance between the pooled full-graph degree distribution and the
    reference, with all voltage levels combined into a single sequence.

    Returns
    -------
    float
        KS distance (scalar).
    """
    ref_degs = np.array(
        [d for level in true_degrees_by_level for d in level], dtype=float
    )
    pooled: List[int] = []
    for G in graphs:
        pooled.extend(deg for _, deg in G.degree())
    if not pooled or len(ref_degs) == 0:
        return float("nan")
    stat, _ = scipy_stats.ks_2samp(ref_degs, np.array(pooled, dtype=float))
    return float(stat)
