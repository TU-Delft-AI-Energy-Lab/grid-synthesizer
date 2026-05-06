"""
Conversion utilities for building Schweitzer-format feeder graphs
from pandapower and pypowsybl reference networks.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import networkx as nx
import numpy as np


def pandapower_to_feeders(net: Any) -> list[nx.Graph]:
    """Convert a pandapower network to a list of Schweitzer-format feeder graphs.

    Each connected component with at least three nodes is returned as a
    separate graph annotated with the attributes expected by
    :func:`~powergrid_synth.distribution.fit_params_from_feeders` and
    :class:`~powergrid_synth.distribution.SchweetzerFeederGenerator`:

    * **Node attributes**: ``h`` (hop distance from root), ``P_mw``
      (positive = load, negative = generation), ``node_type``
      (``'load'`` | ``'injection'`` | ``'intermediate'``), ``pf``
      (power factor).
    * **Edge attributes**: ``length_km`` (cable length in km).

    The function handles lines, transformers, and closed bus–bus switches
    so that the full connectivity of the pandapower network is captured.

    Parameters
    ----------
    net : pandapowerNet
        A pandapower network object.

    Returns
    -------
    list[nx.Graph]
        One feeder graph per connected component (≥ 3 nodes).
    """
    G = nx.Graph()

    # --- Nodes ---------------------------------------------------------------
    for bus_idx in net.bus.index:
        G.add_node(int(bus_idx))

    # --- Edges: lines --------------------------------------------------------
    for _, row in net.line.iterrows():
        G.add_edge(
            int(row["from_bus"]),
            int(row["to_bus"]),
            length_km=row["length_km"],
        )

    # --- Edges: transformers (treated as short connections) -------------------
    if hasattr(net, "trafo") and len(net.trafo) > 0:
        for _, row in net.trafo.iterrows():
            G.add_edge(
                int(row["hv_bus"]),
                int(row["lv_bus"]),
                length_km=0.01,
            )

    # --- Edges: closed bus-bus switches --------------------------------------
    if hasattr(net, "switch") and len(net.switch) > 0:
        for _, row in net.switch.iterrows():
            if row.get("closed", True) and row.get("et") == "b":
                G.add_edge(
                    int(row["bus"]),
                    int(row["element"]),
                    length_km=0.001,
                )

    # --- Source buses (ext_grid) ---------------------------------------------
    source_buses = set(int(b) for b in net.ext_grid.bus.values)

    # --- Aggregate loads and generation per bus ------------------------------
    bus_load: dict[int, float] = {}
    bus_load_q: dict[int, float] = {}
    for _, row in net.load.iterrows():
        b = int(row["bus"])
        bus_load[b] = bus_load.get(b, 0.0) + row["p_mw"]
        bus_load_q[b] = bus_load_q.get(b, 0.0) + row["q_mvar"]

    bus_gen: dict[int, float] = {}
    if hasattr(net, "sgen") and len(net.sgen) > 0:
        for _, row in net.sgen.iterrows():
            b = int(row["bus"])
            bus_gen[b] = bus_gen.get(b, 0.0) + row["p_mw"]

    # --- Build feeders per connected component -------------------------------
    feeders: list[nx.Graph] = []
    for comp in nx.connected_components(G):
        if len(comp) < 3:
            continue
        sub = G.subgraph(comp).copy()

        # Identify root: prefer an ext_grid bus inside this component
        roots_in_comp = source_buses & comp
        root = min(roots_in_comp) if roots_in_comp else min(comp)

        # BFS to assign hop distances
        hop = {root: 0}
        for u, v in nx.bfs_edges(sub, root):
            hop[v] = hop[u] + 1
        nx.set_node_attributes(sub, hop, "h")

        # Assign node types and power
        for n in sub.nodes:
            p_load = bus_load.get(n, 0.0)
            p_gen = bus_gen.get(n, 0.0)

            if p_gen > 0 and p_load == 0:
                sub.nodes[n]["node_type"] = "injection"
                sub.nodes[n]["P_mw"] = -p_gen
            elif p_load > 0:
                sub.nodes[n]["node_type"] = "load"
                sub.nodes[n]["P_mw"] = p_load
            else:
                sub.nodes[n]["node_type"] = "intermediate"
                sub.nodes[n]["P_mw"] = 0.0

            q = bus_load_q.get(n, 0.0)
            s = np.sqrt(p_load**2 + q**2)
            sub.nodes[n]["pf"] = (p_load / s) if s > 0 else 0.95

        feeders.append(sub)

    return feeders


def feeder_summary(feeders: Sequence[nx.Graph]) -> list[dict[str, Any]]:
    """Return a summary dict for each feeder graph.

    Parameters
    ----------
    feeders : sequence of nx.Graph
        Feeder graphs produced by :func:`pandapower_to_feeders`.

    Returns
    -------
    list[dict]
        One dict per feeder with keys ``n_nodes``, ``n_edges``,
        ``max_hop``, ``is_tree``, ``total_load_mw``, ``total_gen_mw``,
        ``node_types``.
    """
    from collections import Counter

    summaries = []
    for f in feeders:
        types = Counter(f.nodes[n]["node_type"] for n in f.nodes)
        hops = [f.nodes[n]["h"] for n in f.nodes]
        total_load = sum(f.nodes[n]["P_mw"] for n in f.nodes if f.nodes[n]["P_mw"] > 0)
        total_gen = sum(-f.nodes[n]["P_mw"] for n in f.nodes if f.nodes[n]["P_mw"] < 0)
        summaries.append({
            "n_nodes": f.number_of_nodes(),
            "n_edges": f.number_of_edges(),
            "max_hop": max(hops) if hops else 0,
            "is_tree": nx.is_tree(f),
            "total_load_mw": total_load,
            "total_gen_mw": total_gen,
            "node_types": dict(types),
        })
    return summaries


def pypowsybl_to_feeders(network: Any) -> list[nx.Graph]:
    """Convert a pypowsybl Network to a list of Schweitzer-format feeder graphs.

    This is the pypowsybl counterpart of :func:`pandapower_to_feeders`.
    It extracts buses, lines, two-winding transformers, loads, and
    generators from the pypowsybl Network and builds annotated feeder
    graphs suitable for :func:`~powergrid_synth.distribution.fit_params_from_feeders`
    and :class:`~powergrid_synth.distribution.SchweetzerFeederGenerator`.

    Parameters
    ----------
    network : pypowsybl.network.Network
        A pypowsybl Network object (loaded from CGMES, XIIDM, MATPOWER, etc.).

    Returns
    -------
    list[nx.Graph]
        One feeder graph per connected component (≥ 3 nodes), with the
        same node/edge attributes as :func:`pandapower_to_feeders`.
    """
    try:
        import pypowsybl as _ppl
    except ImportError as exc:
        raise ImportError(
            "pypowsybl is required for pypowsybl_to_feeders. "
            "Install it with: pip install powergrid_synth[export]"
        ) from exc

    G = nx.Graph()

    # --- Buses ---------------------------------------------------------------
    buses = network.get_buses()
    for bus_id in buses.index:
        G.add_node(bus_id)

    # --- Lines ---------------------------------------------------------------
    lines = network.get_lines()
    for line_id, row in lines.iterrows():
        b1, b2 = row["bus1_id"], row["bus2_id"]
        if b1 not in G or b2 not in G:
            continue
        if not (row.get("connected1", True) and row.get("connected2", True)):
            continue
        G.add_edge(b1, b2, length_km=0.1)  # pypowsybl doesn't store line length

    # --- Two-winding transformers (short connections) ------------------------
    trafos = network.get_2_windings_transformers()
    for trafo_id, row in trafos.iterrows():
        b1, b2 = row["bus1_id"], row["bus2_id"]
        if b1 not in G or b2 not in G:
            continue
        if not (row.get("connected1", True) and row.get("connected2", True)):
            continue
        G.add_edge(b1, b2, length_km=0.01)

    # --- Identify source buses (generators with highest output) --------------
    gens = network.get_generators()
    gen_bus_p: dict[str, float] = {}
    for _, row in gens.iterrows():
        bid = row["bus_id"]
        if bid in G and row.get("connected", True):
            gen_bus_p[bid] = gen_bus_p.get(bid, 0.0) + abs(row.get("target_p", 0.0))

    # --- Loads per bus -------------------------------------------------------
    loads = network.get_loads()
    bus_load: dict[str, float] = {}
    bus_load_q: dict[str, float] = {}
    for _, row in loads.iterrows():
        bid = row["bus_id"]
        if bid in G and row.get("connected", True):
            bus_load[bid] = bus_load.get(bid, 0.0) + row.get("p0", 0.0)
            bus_load_q[bid] = bus_load_q.get(bid, 0.0) + row.get("q0", 0.0)

    # --- Build feeders per connected component -------------------------------
    feeders: list[nx.Graph] = []
    for comp in nx.connected_components(G):
        if len(comp) < 3:
            continue
        sub = G.subgraph(comp).copy()

        # Root: prefer generator bus with highest output; else highest-degree bus
        gens_in_comp = {b: p for b, p in gen_bus_p.items() if b in comp}
        if gens_in_comp:
            root = max(gens_in_comp, key=gens_in_comp.get)
        else:
            root = max(comp, key=lambda n: sub.degree(n))

        # BFS to assign hop distances
        hop = {root: 0}
        for u, v in nx.bfs_edges(sub, root):
            hop[v] = hop[u] + 1
        nx.set_node_attributes(sub, hop, "h")

        # Assign node types and power
        for n in sub.nodes:
            p_load = bus_load.get(n, 0.0)
            p_gen = gen_bus_p.get(n, 0.0)

            if p_gen > 0 and p_load == 0:
                sub.nodes[n]["node_type"] = "injection"
                sub.nodes[n]["P_mw"] = -p_gen
            elif p_load > 0:
                sub.nodes[n]["node_type"] = "load"
                sub.nodes[n]["P_mw"] = p_load
            else:
                sub.nodes[n]["node_type"] = "intermediate"
                sub.nodes[n]["P_mw"] = 0.0

            q = bus_load_q.get(n, 0.0)
            s = np.sqrt(p_load**2 + q**2)
            sub.nodes[n]["pf"] = (p_load / s) if s > 0 else 0.95

        feeders.append(sub)

    return feeders
