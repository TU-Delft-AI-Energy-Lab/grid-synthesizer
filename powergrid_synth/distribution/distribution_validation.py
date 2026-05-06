"""
Validation and statistical comparison for synthetic distribution feeders.

Provides KL-divergence computation and emergent property validation against
reference feeders, as described in Section IV of Schweitzer et al. (2017).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import stats


def kl_divergence_discrete(
    p_counts: np.ndarray, q_counts: np.ndarray, n_bins: Optional[int] = None
) -> float:
    """Compute KL divergence D_KL(P || Q) for two discrete histograms.

    Parameters
    ----------
    p_counts, q_counts : array-like
        Raw counts or probability arrays. They are normalized internally.
    n_bins : int, optional
        Re-bin both arrays into *n_bins* equally spaced bins.

    Returns
    -------
    float
        KL divergence in nats. Returns ``inf`` if Q has zero probability
        where P is nonzero.
    """
    p = np.asarray(p_counts, dtype=float)
    q = np.asarray(q_counts, dtype=float)

    if n_bins is not None and len(p) != n_bins:
        raise ValueError("n_bins doesn't match array length. Re-bin first.")

    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q

    # Add small epsilon to avoid log(0)
    eps = 1e-12
    q = np.where(q < eps, eps, q)
    p = np.where(p < eps, eps, p)

    return float(np.sum(p * np.log(p / q)))


def extract_hop_distribution(G: nx.Graph) -> np.ndarray:
    """Return hop-distance histogram from a feeder graph."""
    hops = [G.nodes[n]["h"] for n in G.nodes]
    max_h = max(hops)
    hist = np.zeros(max_h + 1)
    for h in hops:
        hist[h] += 1
    return hist


def extract_degree_distribution(G: nx.Graph) -> np.ndarray:
    """Return degree histogram from a feeder graph."""
    degrees = [d for _, d in G.degree()]
    max_d = max(degrees)
    hist = np.zeros(max_d + 1)
    for d in degrees:
        hist[d] += 1
    return hist


def extract_cable_length_distribution(G: nx.Graph, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Return cable-length histogram (counts, bin_edges)."""
    lengths = [G.edges[e]["length_km"] for e in G.edges if "length_km" in G.edges[e]]
    if not lengths:
        return np.array([0]), np.array([0, 1])
    return np.histogram(lengths, bins=n_bins)


def extract_load_per_node(G: nx.Graph) -> np.ndarray:
    """Return array of non-zero load values (MW)."""
    return np.array([
        G.nodes[n]["P_mw"] for n in G.nodes
        if G.nodes[n].get("P_mw", 0) > 0
    ])


def compute_emergent_properties(G: nx.Graph) -> Dict[str, float]:
    """Compute emergent (aggregate) properties of a feeder.

    Returns a dict with:

    - ``n_nodes`` : total number of nodes
    - ``n_edges`` : total number of edges
    - ``max_hop`` : maximum hop distance
    - ``mean_degree`` : mean node degree
    - ``total_load_mw`` : sum of positive P
    - ``total_gen_mw`` : sum of negative P (absolute)
    - ``frac_intermediate`` : fraction of intermediate nodes
    - ``frac_injection`` : fraction of injection nodes
    - ``mean_length_km`` : mean cable length
    - ``max_length_km`` : maximum cable length
    - ``total_length_km`` : total cable length
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    hops = [G.nodes[n]["h"] for n in G.nodes]
    max_hop = max(hops) if hops else 0

    degrees = [d for _, d in G.degree()]
    mean_degree = np.mean(degrees) if degrees else 0.0

    loads = [G.nodes[n].get("P_mw", 0.0) for n in G.nodes]
    total_load = sum(p for p in loads if p > 0)
    total_gen = sum(abs(p) for p in loads if p < 0)

    n_int = sum(1 for n in G.nodes if G.nodes[n].get("node_type") == "intermediate")
    n_inj = sum(1 for n in G.nodes if G.nodes[n].get("node_type") == "injection")
    frac_int = n_int / n_nodes if n_nodes > 0 else 0.0
    frac_inj = n_inj / n_nodes if n_nodes > 0 else 0.0

    lengths = [G.edges[e].get("length_km", 0.0) for e in G.edges]
    mean_len = np.mean(lengths) if lengths else 0.0
    max_len = max(lengths) if lengths else 0.0
    total_len = sum(lengths)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "max_hop": max_hop,
        "mean_degree": float(mean_degree),
        "total_load_mw": total_load,
        "total_gen_mw": total_gen,
        "frac_intermediate": frac_int,
        "frac_injection": frac_inj,
        "mean_length_km": float(mean_len),
        "max_length_km": float(max_len),
        "total_length_km": total_len,
    }


def compare_feeders(
    synthetic: nx.Graph,
    reference: nx.Graph,
) -> Dict[str, float]:
    """Compare synthetic and reference feeders via KL-divergence.

    Returns a dict mapping distribution name → KL divergence value.
    """
    results: Dict[str, float] = {}

    # Hop distance
    h_syn = extract_hop_distribution(synthetic)
    h_ref = extract_hop_distribution(reference)
    max_len = max(len(h_syn), len(h_ref))
    h_syn = np.pad(h_syn, (0, max_len - len(h_syn)))
    h_ref = np.pad(h_ref, (0, max_len - len(h_ref)))
    results["hop_distance_kl"] = kl_divergence_discrete(h_syn, h_ref)

    # Degree
    d_syn = extract_degree_distribution(synthetic)
    d_ref = extract_degree_distribution(reference)
    max_len = max(len(d_syn), len(d_ref))
    d_syn = np.pad(d_syn, (0, max_len - len(d_syn)))
    d_ref = np.pad(d_ref, (0, max_len - len(d_ref)))
    results["degree_kl"] = kl_divergence_discrete(d_syn, d_ref)

    # Cable length
    cl_syn, edges_syn = extract_cable_length_distribution(synthetic)
    cl_ref, edges_ref = extract_cable_length_distribution(reference)
    # Re-histogram on common bins
    len_syn = [synthetic.edges[e]["length_km"] for e in synthetic.edges if "length_km" in synthetic.edges[e]]
    len_ref = [reference.edges[e]["length_km"] for e in reference.edges if "length_km" in reference.edges[e]]
    if len_syn and len_ref:
        lo = min(min(len_syn), min(len_ref))
        hi = max(max(len_syn), max(len_ref))
        bins = np.linspace(lo, hi, 51)
        h_syn, _ = np.histogram(len_syn, bins=bins)
        h_ref, _ = np.histogram(len_ref, bins=bins)
        results["cable_length_kl"] = kl_divergence_discrete(h_syn, h_ref)

    return results


def validate_tree(G: nx.Graph) -> List[str]:
    """Check structural validity of a feeder and return list of issues."""
    issues: List[str] = []

    if not nx.is_connected(G):
        issues.append("Graph is not connected.")

    if nx.number_of_edges(G) != nx.number_of_nodes(G) - 1:
        issues.append(
            f"Not a tree: {nx.number_of_edges(G)} edges "
            f"for {nx.number_of_nodes(G)} nodes."
        )

    # Check source at h=0 exists
    if 0 not in G.nodes or G.nodes[0].get("h") != 0:
        issues.append("Node 0 (source) missing or does not have h=0.")

    # Check monotonic hop increase along each edge
    for u, v in G.edges:
        h_u = G.nodes[u].get("h", -1)
        h_v = G.nodes[v].get("h", -1)
        if abs(h_u - h_v) != 1:
            issues.append(f"Edge ({u},{v}): hop diff = {abs(h_u - h_v)} ≠ 1.")

    return issues
