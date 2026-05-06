"""
Schweitzer et al. (2017) distribution feeder synthesis algorithm.

Implements the five-step pipeline:

1. **Node generation** — Negative Binomial hop distances, power factor CDF.
2. **Feeder connection** — bimodal Gamma degrees, bottom-up predecessor matching.
3. **Node properties** — intermediate / injection / load assignment.
4. **Cable type** — Exponential current-ratio sampling with cable library.
5. **Cable length** — modified Cauchy distribution with hop-dependent clipping.

Returns an annotated :class:`networkx.Graph`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import stats

from .distribution_params import (
    CableLibraryEntry,
    DistributionSynthParams,
)


class SchweetzerFeederGenerator:
    """Generate a single MV radial distribution feeder.

    Parameters
    ----------
    params : DistributionSynthParams, optional
        All distribution parameters. Defaults to Table III values.
    seed : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        params: Optional[DistributionSynthParams] = None,
        seed: Optional[int] = None,
    ):
        self.params = params if params is not None else DistributionSynthParams()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_feeder(
        self,
        n_nodes: int,
        total_load_mw: float,
        total_gen_mw: float = 0.0,
        v_nom_kv: float = 10.0,
        assign_cable_types: bool = True,
        assign_cable_lengths: bool = True,
    ) -> nx.Graph:
        """Generate a complete MV distribution feeder.

        Parameters
        ----------
        n_nodes : int
            Total number of nodes in the feeder (including source and root).
        total_load_mw : float
            Total real-power load (MW) to be distributed across load buses.
        total_gen_mw : float, optional
            Total real-power generation/injection (MW). Default 0.
        v_nom_kv : float, optional
            Nominal MV voltage in kV. Default 10.
        assign_cable_types : bool, optional
            Whether to run Step 4 (cable type assignment). Default True.
        assign_cable_lengths : bool, optional
            Whether to run Step 5 (cable length and impedance assignment).
            Default True.

        Returns
        -------
        nx.Graph
            Annotated graph with node attributes (``h``, ``P_mw``, ``Q_mvar``,
            ``pf``, ``node_type``) and, when enabled, edge attributes
            (``cable_type``, ``length_km``, ``r_ohm``, ``x_ohm``,
            ``max_i_ka``, ``I_est_ka``).
        """
        G = nx.Graph()

        # Step 1: generate nodes with hop distances and power factors
        self._generate_nodes(G, n_nodes)

        # Step 2: connect into a tree
        self._connect_nodes(G)

        # Step 3: assign node properties (load / generation / intermediate)
        self._assign_node_properties(G, total_load_mw, total_gen_mw)

        # Step 4: assign cable types (optional)
        if assign_cable_types:
            self._assign_cable_types(G, v_nom_kv)

        # Step 5: assign cable lengths and compute impedance (optional)
        if assign_cable_lengths:
            self._assign_cable_lengths(G)

        # Store nominal voltage
        G.graph["v_nom_kv"] = v_nom_kv

        return G

    # ------------------------------------------------------------------
    # Step 1: Node generation
    # ------------------------------------------------------------------

    def _generate_nodes(self, G: nx.Graph, n_nodes: int) -> None:
        """Create *n_nodes* nodes with hop-distance and power-factor attrs."""
        p = self.params

        # Source (h=0) and root (h=1) are deterministic
        G.add_node(0, h=0)
        G.add_node(1, h=1)

        # Sample hop distances for the remaining nodes
        hop_rv = stats.nbinom(p.hop_dist.r, p.hop_dist.p)
        hops = hop_rv.rvs(size=n_nodes - 2, random_state=self.rng)
        # Shift: nbinom support starts at 0; feeder nodes start at h=2
        hops = hops + 2

        for i, h_val in enumerate(hops, start=2):
            G.add_node(i, h=int(h_val))

        # Ensure no gaps in hop levels
        self._fill_hop_gaps(G)

        # Assign power factors from the empirical CDF
        pf_vals = np.array([pf for pf, _ in p.pf_cdf])
        pf_probs = np.array([cp for _, cp in p.pf_cdf])
        for node in G.nodes:
            u = self.rng.uniform()
            idx = np.searchsorted(pf_probs, u)
            idx = min(idx, len(pf_vals) - 1)
            G.nodes[node]["pf"] = float(pf_vals[idx])

    def _fill_hop_gaps(self, G: nx.Graph) -> None:
        """Compress hop distances so every level from 0 to max has ≥1 node.

        Gaps are removed by decrementing hop values above each gap, ensuring
        a contiguous sequence of hop levels.
        """
        nodes_by_h: Dict[int, List[int]] = {}
        for n in G.nodes:
            h = G.nodes[n]["h"]
            nodes_by_h.setdefault(h, []).append(n)

        # Build sorted occupied levels and create a mapping to compressed levels
        occupied = sorted(nodes_by_h.keys())
        compress_map = {old_h: new_h for new_h, old_h in enumerate(occupied)}

        for n in G.nodes:
            G.nodes[n]["h"] = compress_map[G.nodes[n]["h"]]

    # ------------------------------------------------------------------
    # Step 2: Feeder connection
    # ------------------------------------------------------------------

    def _connect_nodes(self, G: nx.Graph) -> None:
        """Connect nodes into a radial tree using bimodal-Gamma degrees."""
        p = self.params

        # Collect nodes sorted by hop distance
        nodes = sorted(G.nodes, key=lambda n: G.nodes[n]["h"])
        max_h = max(G.nodes[n]["h"] for n in G.nodes)

        # --- Deterministic degree assignments ---
        # Source and leaf nodes get degree 1
        for n in nodes:
            h = G.nodes[n]["h"]
            if h == 0 or h == max_h:
                G.nodes[n]["d_target"] = 1
            elif h == 1:
                # Root degree = 1 (source connection) + number of h=2 nodes
                n_h2 = sum(1 for n2 in G.nodes if G.nodes[n2]["h"] == 2)
                G.nodes[n]["d_target"] = 1 + n_h2

        # --- Stochastic degree assignments (mixture Gamma) ---
        gp = p.degree_dist
        for n in nodes:
            if "d_target" not in G.nodes[n]:
                h = G.nodes[n]["h"]
                clip = p.degree_clip.a * (h ** p.degree_clip.b) if h > 0 else 1
                clip = max(clip, 1)

                while True:
                    if self.rng.uniform() < gp.pi:
                        d = stats.gamma.rvs(
                            gp.a1, scale=gp.b1, random_state=self.rng
                        )
                    else:
                        d = stats.gamma.rvs(
                            gp.a2, scale=gp.b2, random_state=self.rng
                        )
                    d = max(1, round(d))
                    if d <= clip:
                        break
                G.nodes[n]["d_target"] = d

        # --- Bottom-up connection ---
        for n in nodes:
            G.nodes[n]["d_actual"] = 0

        # Sort descending by hop (furthest first)
        for n in sorted(nodes, key=lambda n: -G.nodes[n]["h"]):
            h = G.nodes[n]["h"]
            if h == 0:
                continue  # source has no predecessor

            # Find viable predecessors at h-1
            predecessors = [
                n2 for n2 in G.nodes
                if G.nodes[n2]["h"] == h - 1
            ]
            if not predecessors:
                continue

            # Pick predecessor with largest degree deficit
            best = min(
                predecessors,
                key=lambda n2: G.nodes[n2]["d_actual"] - G.nodes[n2]["d_target"],
            )
            G.add_edge(n, best)
            G.nodes[n]["d_actual"] += 1
            G.nodes[best]["d_actual"] += 1

    # ------------------------------------------------------------------
    # Step 3: Node properties
    # ------------------------------------------------------------------

    def _assign_node_properties(
        self, G: nx.Graph, total_load_mw: float, total_gen_mw: float
    ) -> None:
        """Assign intermediate / injection / load node types and powers."""
        nodes = list(G.nodes)
        n_total = len(nodes)
        p = self.params

        for n in nodes:
            G.nodes[n]["node_type"] = "load"
            G.nodes[n]["P_mw"] = 0.0
            G.nodes[n]["Q_mvar"] = 0.0

        # 3a: Intermediate nodes
        self._assign_intermediate(G)

        # 3b: Injection (generation) nodes
        if total_gen_mw > 0:
            self._assign_injection(G, total_gen_mw)

        # 3c: Load (consumption) nodes
        self._assign_load(G, total_load_mw)

    def _assign_intermediate(self, G: nx.Graph) -> None:
        """Mark a fraction of nodes as intermediate (zero load)."""
        p = self.params
        n_total = G.number_of_nodes()
        frac = stats.beta.rvs(
            p.intermediate_frac.alpha,
            p.intermediate_frac.beta,
            random_state=self.rng,
        )
        n_int = max(1, round(n_total * frac))  # at least the source

        # Source is always intermediate
        G.nodes[0]["node_type"] = "intermediate"
        remaining = n_int - 1

        # Hop distances from mixture Poisson
        mp = p.intermediate_hop
        max_h = max(G.nodes[n]["h"] for n in G.nodes)
        available = [n for n in G.nodes if n != 0]

        for _ in range(remaining):
            if not available:
                break
            if self.rng.uniform() < mp.pi:
                h_target = self.rng.poisson(mp.mu1)
            else:
                h_target = self.rng.poisson(mp.mu2)
            h_target = int(np.clip(h_target, 0, max_h))

            # Find an available node at or near this hop
            candidates = [n for n in available if G.nodes[n]["h"] == h_target]
            if not candidates:
                candidates = sorted(available, key=lambda n: abs(G.nodes[n]["h"] - h_target))
            chosen = candidates[0]
            G.nodes[chosen]["node_type"] = "intermediate"
            available.remove(chosen)

    def _assign_injection(self, G: nx.Graph, total_gen_mw: float) -> None:
        """Assign power injection to a fraction of nodes."""
        p = self.params
        n_total = G.number_of_nodes()
        frac = stats.beta.rvs(
            p.injection_frac.alpha,
            p.injection_frac.beta,
            random_state=self.rng,
        )
        n_inj = max(1, round(n_total * frac))

        max_h = max(G.nodes[n]["h"] for n in G.nodes)
        available = [
            n for n in G.nodes
            if G.nodes[n]["node_type"] != "intermediate"
        ]

        # Mixture Normal for normalized hop distance
        mn = p.injection_hop
        injection_nodes = []
        for _ in range(n_inj):
            if not available:
                break
            if self.rng.uniform() < mn.pi:
                h_norm = self.rng.normal(mn.mu1, mn.sigma1)
            else:
                h_norm = self.rng.normal(mn.mu2, mn.sigma2)
            h_norm = np.clip(h_norm, 0, 1)
            h_target = int(round(h_norm * max_h))

            candidates = [n for n in available if G.nodes[n]["h"] == h_target]
            if not candidates:
                candidates = sorted(available, key=lambda n: abs(G.nodes[n]["h"] - h_target))
            chosen = candidates[0]
            G.nodes[chosen]["node_type"] = "injection"
            injection_nodes.append(chosen)
            available.remove(chosen)

        # Assign power: uniform + Normal deviation
        nd = p.injection_deviation
        n_inj_actual = len(injection_nodes)
        if n_inj_actual == 0:
            return

        for n in injection_nodes:
            while True:
                eps = self.rng.normal(nd.mu, nd.sigma)
                if 1.0 / n_inj_actual + eps > 0:
                    break
            pf = G.nodes[n]["pf"]
            P = -total_gen_mw * (1.0 / n_inj_actual + eps)
            G.nodes[n]["P_mw"] = P
            G.nodes[n]["Q_mvar"] = P * math.tan(math.acos(pf))

    def _assign_load(self, G: nx.Graph, total_load_mw: float) -> None:
        """Assign positive load to all remaining (non-intermediate, non-inj) nodes."""
        p = self.params
        load_nodes = [
            n for n in G.nodes
            if G.nodes[n]["node_type"] == "load"
        ]
        n_load = len(load_nodes)
        if n_load == 0:
            return

        tls = p.load_deviation
        for n in load_nodes:
            while True:
                eps = stats.t.rvs(
                    tls.nu, loc=tls.mu, scale=tls.sigma,
                    random_state=self.rng,
                )
                if 1.0 / n_load + eps > 0:
                    break
            pf = G.nodes[n]["pf"]
            P = total_load_mw * (1.0 / n_load + eps)
            G.nodes[n]["P_mw"] = P
            G.nodes[n]["Q_mvar"] = P * math.tan(math.acos(pf))

    # ------------------------------------------------------------------
    # Step 4: Cable type assignment
    # ------------------------------------------------------------------

    def _assign_cable_types(self, G: nx.Graph, v_nom_kv: float) -> None:
        """Assign cable types from the library using I_est / I_nom ratio."""
        p = self.params
        library = p.cable_library
        threshold = p.current_threshold

        # Compute downstream power for each node (tree traversal)
        downstream = self._compute_downstream_power(G)

        # Process edges from furthest to closest
        edges_sorted = sorted(
            G.edges,
            key=lambda e: -max(G.nodes[e[0]]["h"], G.nodes[e[1]]["h"]),
        )

        for u, v in edges_sorted:
            # Identify upstream/downstream node
            if G.nodes[u]["h"] < G.nodes[v]["h"]:
                upstream, downstream_node = u, v
            else:
                upstream, downstream_node = v, u

            h = G.nodes[downstream_node]["h"]
            S_down = abs(downstream.get(downstream_node, 0.0))
            I_est = S_down / (math.sqrt(3) * v_nom_kv) if v_nom_kv > 0 else 0.0

            if I_est > 0:
                if self.rng.uniform() < 2.0 / 3.0:
                    # Use max nominal current of downstream node's incident cables
                    # (for the first pass, just pick the median cable)
                    cable = self._pick_cable_by_capacity(
                        I_est * 1.5, h, library, threshold
                    )
                else:
                    # Sample I_est/I_nom from Exponential, solve for I_nom
                    ratio = self.rng.exponential(p.current_ratio.mu)
                    ratio = max(ratio, 0.01)  # avoid division by zero
                    i_nom_target = I_est / ratio
                    cable = self._pick_cable_by_capacity(
                        i_nom_target, h, library, threshold
                    )
            else:
                # Zero-current branch: pick average cable
                cable = library[len(library) // 2]

            G.edges[u, v]["cable_type"] = cable.name
            G.edges[u, v]["max_i_ka"] = cable.max_i_ka
            G.edges[u, v]["r_per_km"] = cable.r_ohm_per_km
            G.edges[u, v]["x_per_km"] = cable.x_ohm_per_km
            G.edges[u, v]["I_est_ka"] = I_est / 1000.0

    def _pick_cable_by_capacity(
        self,
        i_nom_target: float,
        h: int,
        library: List[CableLibraryEntry],
        threshold,
    ) -> CableLibraryEntry:
        """Pick cable from library closest to target I_nom, respecting threshold."""
        # Convert target to kA for comparison
        target_ka = i_nom_target / 1000.0

        # Apply threshold for far-from-source branches
        filtered = library
        if h >= threshold.h_min:
            filtered = [c for c in library if c.max_i_ka <= threshold.i_nom_max / 1000.0]
            if not filtered:
                filtered = library  # fallback

        # Weight by frequency × closeness
        best = min(filtered, key=lambda c: abs(c.max_i_ka - target_ka))
        return best

    def _compute_downstream_power(self, G: nx.Graph) -> Dict[int, float]:
        """Compute total downstream apparent power for each node.

        Returns dict mapping node → downstream S (MVA).
        """
        downstream: Dict[int, float] = {}
        # Process leaves first, then work toward root
        for n in sorted(G.nodes, key=lambda n: -G.nodes[n]["h"]):
            P = G.nodes[n].get("P_mw", 0.0)
            Q = G.nodes[n].get("Q_mvar", 0.0)
            # Add children's downstream power
            children = [
                nb for nb in G.neighbors(n)
                if G.nodes[nb]["h"] > G.nodes[n]["h"]
            ]
            child_P = sum(downstream.get(c, 0.0) for c in children)
            # Store as apparent power magnitude
            total_P = P + child_P
            downstream[n] = total_P
        return downstream

    # ------------------------------------------------------------------
    # Step 5: Cable length assignment
    # ------------------------------------------------------------------

    def _assign_cable_lengths(self, G: nx.Graph) -> None:
        """Assign lengths from modified Cauchy and compute impedance."""
        p = self.params
        cauchy = p.cable_length
        clip = p.length_clip

        for u, v in G.edges:
            h = max(G.nodes[u]["h"], G.nodes[v]["h"])
            l_max = clip.a * math.exp(clip.b * h)
            # Ensure l_max is reasonable; fall back to default if
            # the fitted clip produces an excessively small bound.
            l_max = max(l_max, 0.01)

            max_attempts = 2000
            raw = None
            for _ in range(max_attempts):
                # Sample from modified Cauchy (positive support)
                candidate = stats.cauchy.rvs(
                    loc=cauchy.x0, scale=cauchy.gamma,
                    random_state=self.rng,
                )
                if 0 < candidate <= l_max:
                    raw = candidate
                    break
            if raw is None:
                # Fallback: use the median (x0) clipped to (0, l_max].
                raw = min(max(cauchy.x0, 0.001), l_max)

            G.edges[u, v]["length_km"] = raw
            r_per_km = G.edges[u, v].get("r_per_km", 0.0)
            x_per_km = G.edges[u, v].get("x_per_km", 0.0)
            G.edges[u, v]["r_ohm"] = r_per_km * raw
            G.edges[u, v]["x_ohm"] = x_per_km * raw
