r"""
Transmission-line impedance and capacity allocation.

This module assigns branch impedance components (:math:`R`, :math:`X`) and
thermal-capacity limits (:math:`F_l^{\max}`) to all transmission lines in a
synthetic power grid.  The statistical models follow
`Sadeghian et al. (2018) <https://ieeexplore.ieee.org/document/8585532>`_
(Section IV) and the SynGrid MATLAB function ``sg_line.m``.

Impedance magnitudes are drawn from a LogNormal distribution; line angles
from a Lévy stable distribution.  A DCPF-based swapping heuristic
associates low impedance with high-flow lines, reproducing empirical
correlations.  Capacity factors (gauge ratios) are assigned through a
2-D empirical PMF table.

.. todo::
   `Reed et al. (2006) <https://www.tandfonline.com/doi/full/10.1081/STA-120037438>`_
   argues that impedance magnitudes may follow a clipped
   double-Pareto-logNormal distribution, which could be explored.
"""

import numpy as np
import networkx as nx
from scipy.stats import levy_stable
from typing import Dict, Tuple, List
from ..core.reference_data import get_reference_stats
from ..core.dcpf import DCPowerFlow

class TransmissionLineAllocator:
    r"""Allocate impedance and capacity limits to transmission lines.

    The algorithm follows `Sadeghian et al. (2018)
    <https://ieeexplore.ieee.org/document/8585532>`_ and the SynGrid
    MATLAB toolbox (``sg_line.m``, ``sg_flow_lim.m``):

    1. **Impedance generation** — magnitudes :math:`Z` from
       LogNormal(:math:`\mu`, :math:`\sigma`), angles :math:`\varphi`
       from a Lévy stable distribution
       :math:`S(\alpha_s, \beta_s, \gamma_s, \delta_s)`.  Then
       :math:`X = Z \sin\varphi`, :math:`R = Z \cos\varphi`.
    2. **DCPF-based swapping** — sort impedances ascending and flows
       descending, then randomly swap ~20–30 % of assignments to
       introduce variance while preserving the negative correlation
       between impedance and flow.
    3. **(Optional) Topology refinement** — iteratively add
       low-impedance lines between max angle-difference bus pairs
       and remove weak high-:math:`X` lines until the angle spread
       is below a size-dependent threshold.
    4. **Capacity assignment** — gauge ratios
       :math:`\beta_l = F_l / F_l^{\max}` from
       Exponential(:math:`\mu_\beta`) with overload injection;
       assigned via 2-D table ``Tab_2D_FlBeta``.
       Capacity limits: :math:`F_l^{\max} = F_l / \beta_l`.

    Parameters
    ----------
    graph : networkx.Graph
        Power grid graph with nodal generation/load attributes.
    ref_sys_id : int, optional
        Reference system (1 = NYISO-2935, 2 = WECC-16994).  Default 1.

    Attributes
    ----------
    stab_params : list of float
        Lévy stable parameters :math:`[\alpha_s, \beta_s, \gamma_s, \delta_s]`.
    tab_fl_beta : numpy.ndarray
        2-D empirical PMF table for flow–beta assignment.
    mu_beta : float
        Mean of the exponential distribution for :math:`\beta`.
    overload_b : float
        Fraction of lines assigned overload (:math:`\beta > 1`).
    """

    def __init__(self, graph: nx.Graph, ref_sys_id: int = 1):
        self.graph = graph
        self.ref_sys_id = ref_sys_id
        
        try:
            self.stats = get_reference_stats(ref_sys_id)
        except ValueError:
             self.stats = get_reference_stats(1)
             
        self.stab_params = self.stats.get('stab', [1.374, -0.838, 2.965, 85.801])
        self.tab_fl_beta = self.stats['Tab_2D_FlBeta']
        self.mu_beta = self.stats.get('mu_beta', 0.27)
        self.overload_b = self.stats.get('Overload_b', 0.00083)

    def _generate_phi(self, num_lines: int) -> np.ndarray:
        r"""Generate line angles from a Lévy stable distribution.

        Draws :math:`\varphi \sim S(\alpha_s, \beta_s, \gamma_s, \delta_s)`
        and clips to :math:`[0.01, 89.99]` degrees.  Out-of-range samples
        are resampled up to 20 times before a hard clip.

        Parameters
        ----------
        num_lines : int
            Number of transmission lines.

        Returns
        -------
        numpy.ndarray, shape (num_lines,)
            Line angle in degrees for each branch.
        """
        alpha, beta, gamma, delta = self.stab_params
        
        phi = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=num_lines)
        
        # Ensure phi is within logical bounds [0, 90] degrees mostly
        mask = (phi > 90) | (phi < 0)
        max_retries = 20
        count = 0
        while np.any(mask) and count < max_retries:
            n_resample = np.sum(mask)
            new_vals = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=n_resample)
            phi[mask] = new_vals
            mask = (phi > 90) | (phi < 0)
            count += 1
            
        # Hard clip fallback if retries exhausted
        phi = np.clip(phi, 0.01, 89.99)
        return phi

    def _generate_beta(self, num_lines: int) -> np.ndarray:
        r"""Generate gauge ratios from an exponential distribution.

        Draws :math:`\beta \sim \mathrm{Exp}(\mu_\beta)` and resamples
        values exceeding 1.0.  A fraction ``overload_b`` of lines are then
        injected with :math:`\beta \in (1.0, 1.2]` to model bottleneck
        / overloaded lines (Sadeghian et al., 2018, Sec. IV).

        Parameters
        ----------
        num_lines : int
            Number of transmission lines.

        Returns
        -------
        numpy.ndarray, shape (num_lines,)
            Sorted gauge-ratio values.
        """
        beta = np.random.exponential(scale=self.mu_beta, size=num_lines)
        # Handle outliers (beta > 1)
        mask = beta > 1.0
        max_retries = 10
        count = 0
        while np.any(mask) and count < max_retries:
            n_resample = np.sum(mask)
            beta[mask] = np.random.exponential(scale=self.mu_beta, size=n_resample)
            mask = beta > 1.0
            count += 1
            
        # Overload injection (simulating bottleneck lines)
        n_overload = int(round(num_lines * self.overload_b))
        if n_overload > 0:
            indices = np.random.choice(num_lines, n_overload, replace=False)
            # Overload_bet = 1 + 0.2 * rand
            beta[indices] = 1.0 + 0.2 * np.random.rand(n_overload)
            
        # Avoid zero or near-zero betas (division by zero protection)
        small_mask = beta < 1e-4
        n_small = np.sum(small_mask)
        if n_small > 0:
            beta[small_mask] = 0.01 + 0.099 * np.random.rand(n_small)
        
        return np.sort(beta)

    def _assign_betas(self, flows: np.ndarray, betas: np.ndarray) -> np.ndarray:
        r"""Assign gauge ratios to lines via 2-D bin matching.

        Uses the empirical PMF ``Tab_2D_FlBeta`` to reproduce the joint
        distribution :math:`f(\bar{F}_l, \beta_l)` from the reference
        system.  Lines are sorted by normalised flow and betas by value,
        then paired randomly within matching bins (high-to-low traversal).

        Parameters
        ----------
        flows : numpy.ndarray, shape (m, 2)
            ``[line_index, normalised_flow]``.
        betas : numpy.ndarray, shape (m,)
            Sorted gauge-ratio values from :meth:`_generate_beta`.

        Returns
        -------
        numpy.ndarray, shape (m, 3)
            ``[line_index, normalised_flow, beta]``.
        """
        num_lines = len(flows)
        if num_lines == 0: return np.array([])

        tab_n = np.round(self.tab_fl_beta * num_lines).astype(int)
        
        # Adjust for rounding errors to match total lines
        diff = num_lines - np.sum(tab_n)
        if diff != 0:
            flat_idx = np.argmax(tab_n)
            r, c = np.unravel_index(flat_idx, tab_n.shape)
            tab_n[r, c] += diff

        # Targets per Flow class (Rows) and Beta class (Cols)
        n_flow_targets = np.sum(tab_n, axis=1) 
        n_beta_targets = np.sum(tab_n, axis=0) 
        
        # Sort flows (input is [idx, val])
        sorted_flows = flows[flows[:, 1].argsort()]
        
        # Binning Flows
        flow_bins = []
        curr = 0
        for count in n_flow_targets:
            end = curr + count
            flow_bins.append(list(sorted_flows[curr:end]))
            curr = end
            
        # Binning Betas (already sorted)
        beta_bins = [] 
        curr = 0
        for count in n_beta_targets:
            end = curr + count
            beta_bins.append(list(betas[curr:end]))
            curr = end
            
        # Assignment Loop: Rows(Flows) x Cols(Betas)
        final_assignment = [] # [LineIdx, FlowVal, BetaVal]
        
        # Iterate Rows (Flows) - typically high to low in MATLAB (14:-1:1)
        for r in range(len(n_flow_targets)-1, -1, -1):
            # Iterate Cols (Betas) - typically high to low in MATLAB (16:-1:1)
            for c in range(len(n_beta_targets)-1, -1, -1):
                count = tab_n[r, c]
                if count > 0:
                    f_bin = flow_bins[r]
                    b_bin = beta_bins[c]
                    
                    take = min(count, len(f_bin), len(b_bin))
                    
                    for _ in range(take):
                        if f_bin and b_bin:
                            # Random sample without replacement
                            f_idx = np.random.randint(0, len(f_bin))
                            flow_data = f_bin.pop(f_idx) # [LineIdx, FlowVal]
                            
                            b_idx = np.random.randint(0, len(b_bin))
                            beta_val = b_bin.pop(b_idx)
                            
                            final_assignment.append([flow_data[0], flow_data[1], beta_val])

        # Handle leftovers if any (due to bin mismatch/emptying)
        rem_flows = [f for bin in flow_bins for f in bin]
        rem_betas = [b for bin in beta_bins for b in bin]
        
        for i in range(min(len(rem_flows), len(rem_betas))):
             final_assignment.append([rem_flows[i][0], rem_flows[i][1], rem_betas[i]])
             
        return np.array(final_assignment)

    def _refine_topology(self):
        r"""Refine grid topology to reduce phase-angle spread.

        Iteratively tightens the electrical diameter of the network
        (ported from ``sg_flow_lim.m``):

        1. Compute the DCPF and measure
           :math:`\Delta\theta_{\max} = \max(\theta) - \min(\theta)`.
        2. If :math:`\Delta\theta_{\max} > TT + 2` with
           :math:`TT = 10^{0.3196 \log_{10} N + 0.8324}`,
           add a low-impedance edge between the bus pair with the
           largest angle difference.
        3. Remove a random high-:math:`X` edge (top 20 %) whose
           end-point degrees are both :math:`\ge 3`, preserving
           graph connectivity.
        4. Repeat for up to 10 iterations.
        """
        n_nodes = self.graph.number_of_nodes()
        # Target threshold for angle difference
        tt_target = 10**(0.3196 * np.log10(n_nodes) + 0.8324)
        
        max_iter = 10 # Safety limit to prevent infinite loops
        
        for _ in range(max_iter):
            dcpf = DCPowerFlow(self.graph)
            flows, angles = dcpf.run() 
            
            theta_rad = np.array(list(angles.values()))
            node_keys = list(angles.keys())
            
            # Convert to degrees
            theta_deg = np.degrees(theta_rad)
            if len(theta_deg) == 0: break 
            
            angle_spread = np.max(theta_deg) - np.min(theta_deg)
            
            # Condition: max(abs(DD)) > TT_target + 2
            if angle_spread <= tt_target + 2:
                break # Grid is sufficiently compact electrically
                
            # --- 1. Add Strengthening Line ---
            # Find pair with max difference
            min_idx = np.argmin(theta_deg)
            max_idx = np.argmax(theta_deg)
            min_node = node_keys[min_idx]
            max_node = node_keys[max_idx]
            
            # Add edge if not exists
            if not self.graph.has_edge(min_node, max_node):
                # Low impedance parameters from MATLAB code
                r_new = 0.001 + np.random.rand() * 0.001
                x_new = 0.002 + np.random.rand() * 0.003
                z_new = np.sqrt(r_new**2 + x_new**2)
                
                self.graph.add_edge(min_node, max_node, r=r_new, x=x_new, z=z_new)
            
            # --- 2. Remove Weak Line ---
            # Candidates: Select from lines with high X (Impedance)
            edges = list(self.graph.edges(data=True))
            if not edges: break

            # Sort by X descending (High X = Weak/Long line)
            sorted_edges = sorted(edges, key=lambda e: e[2].get('x', 0), reverse=True)
            
            n_edges = len(sorted_edges)
            br_sl = 0.8 # Select top 20%
            n_candidates = int(n_edges * (1.0 - br_sl))
            n_candidates = max(n_candidates, 1)
            
            # Candidates are the 'tail' of the list in MATLAB (high X)
            candidates = sorted_edges[:n_candidates]
            
            # Filter by degree to avoid isolating nodes
            valid_candidates = []
            min_deg = 3 if n_nodes >= 40 else 2
            
            for u, v, d in candidates:
                if self.graph.degree(u) >= min_deg and self.graph.degree(v) >= min_deg:
                    valid_candidates.append((u, v, d))
            
            if valid_candidates:
                # Pick random candidate to remove
                idx = np.random.randint(0, len(valid_candidates))
                u_rem, v_rem, d_rem = valid_candidates[idx]
                
                self.graph.remove_edge(u_rem, v_rem)
                
                # Verify Grid Integrity
                # MATLAB: checks success flag. Here we check connectivity.
                if not nx.is_connected(self.graph):
                    # Revert if disconnected
                    self.graph.add_edge(u_rem, v_rem, **d_rem)

    def allocate(self, refine_topology: bool = False) -> Dict[Tuple[int, int], float]:
        r"""Run the full transmission-line allocation pipeline.

        Executes the seven-step procedure:

        1. Draw impedance magnitudes
           :math:`Z \sim \text{LogNormal}(\mu, \sigma)`, clipped to
           [0.001, 0.5] p.u.
        2. Generate angles :math:`\varphi` (Lévy stable), compute
           :math:`X = Z\sin\varphi`, :math:`R = Z\cos\varphi`.
        3. Iterative DCPF swapping: sort :math:`Z` ascending / flows
           descending, randomly swap ~20–30 % of assignments.
        4. (Optional) Topology refinement via :meth:`_refine_topology`.
        5. Final DCPF to obtain converged flows.
        6. Generate and assign gauge ratios (:math:`\beta`) via
           :meth:`_generate_beta` and :meth:`_assign_betas`.
        7. Set capacity limits:
           :math:`F_l^{\max} = F_l / \beta_l` with a minimum-capacity
           fallback (5 + 100 · rand MW when :math:`\le 2`).

        Parameters
        ----------
        refine_topology : bool, optional
            If ``True``, run topology refinement after step 3.  Default
            is ``False``.

        Returns
        -------
        dict
            Mapping ``(u, v)`` edge tuple to capacity limit (MW).
        """
        edges = list(self.graph.edges())
        m_lines = len(edges)
        n_nodes = self.graph.number_of_nodes()
        
        if m_lines == 0: return {}

        # 1. Initial Impedance (Zpr)
        # LogNormal distribution
        zpr_pars = self.stats.get('Zpr_pars', [-2.38, 1.99, 1.99])
        mu_len = zpr_pars[0]
        sigma_len = zpr_pars[1]
        
        zpr = np.random.lognormal(mu_len, sigma_len, m_lines)
        # physical practice in the per-unit system, and for stability 
        zpr = np.clip(zpr, 0.001, 0.5) 
        
        # 2. Generate Phi and Initial X, R
        phi = self._generate_phi(m_lines)
        
        for i, (u, v) in enumerate(edges):
            # X = Z * sin(phi), R = Z * cos(phi)
            # Zpr is magnitude
            x_val = zpr[i] * np.sin(np.deg2rad(phi[i]))
            r_val = zpr[i] * np.cos(np.deg2rad(phi[i]))
            self.graph[u][v]['x'] = max(x_val, 1e-5)
            self.graph[u][v]['r'] = max(r_val, 1e-5)
            self.graph[u][v]['z'] = zpr[i] 
            self.graph[u][v]['edge_idx'] = i

        # 3. Iterative Refinement
        iterations = 2 if n_nodes >= 300 else 1
        dcpf = DCPowerFlow(self.graph)
        
        for _ in range(iterations):
            # Calculate Flow
            flows, _ = dcpf.run()
            
            # Map flows to current edges
            flow_vals = np.zeros(m_lines)
            for i, (u, v) in enumerate(edges):
                flow_vals[i] = abs(flows.get((u, v), 0.0))
            
            # Sort Z magnitudes (Ascending)
            current_z = np.array([self.graph[u][v]['z'] for u, v in edges])
            sorted_z = np.sort(current_z) 
            
            # Sort Flows (Descending)
            # We want High Flow to get Low Z
            flow_indices = np.argsort(-flow_vals) 
            
            # Create a pool of Z values ordered by flow rank
            # z_pool[0] is smallest Z, assigned to flow_indices[0] (highest flow)
            z_pool = sorted_z.copy()
            
            # Swapping Logic
            if n_nodes > 1200:
                as_param = 0.3
                an_param = 0.8
            else:
                as_param = 0.2
                an_param = 0.2
                
            n_swap = int(round(as_param * m_lines))
            n_neib = int(round(an_param * m_lines))
            
            for _ in range(n_swap):
                # Pick random index
                idx1 = np.random.randint(0, m_lines)
                if idx1 == 0: idx1 = 1 
                
                # Pick neighbor
                xch_disf = int(np.floor(n_neib * np.random.rand()))
                xch_dis = min(xch_disf, (m_lines - 1 - idx1))
                idx2 = idx1 + xch_dis
                
                # Swap Z values
                if idx2 < m_lines:
                    z_pool[idx1], z_pool[idx2] = z_pool[idx2], z_pool[idx1]
            
            # Assign Z back to lines
            for rank_i, line_idx in enumerate(flow_indices):
                z_val = z_pool[rank_i]
                u, v = edges[line_idx]
                
                # Recalculate X, R using original Phi structure
                p = phi[line_idx]
                self.graph[u][v]['z'] = z_val
                self.graph[u][v]['x'] = max(z_val * np.sin(np.deg2rad(p)), 1e-5)
                self.graph[u][v]['r'] = max(z_val * np.cos(np.deg2rad(p)), 1e-5)

        # 4. Topology Refinement (Optional)
        if refine_topology:
            self._refine_topology()
            # Refresh edge list after topology changes
            edges = list(self.graph.edges())
            m_lines = len(edges)

        # 5. Final Flow Calculation
        flows, _ = dcpf.run()
        flow_vals = np.array([abs(flows.get((u, v), 0.0)) for u, v in edges])
        max_flow = np.max(flow_vals) if len(flow_vals) > 0 and np.max(flow_vals) > 0 else 1.0
        
        # 6. Generate & Assign Capacity Factors (Beta)
        betas = self._generate_beta(m_lines)
        
        # Normalize flows
        norm_flow_data = []
        for i, val in enumerate(flow_vals):
            norm_flow_data.append([i, val / max_flow])
        norm_flow_data = np.array(norm_flow_data)
        
        # Assign Betas
        assigned = self._assign_betas(norm_flow_data, betas)
        
        # 7. Calculate & Set Limits
        line_caps = {}
        for row in assigned:
            idx = int(row[0])
            beta = row[2]
            u, v = edges[idx]
            
            flow_mw = flow_vals[idx]
            if beta < 1e-3: beta = 1e-3
            limit = flow_mw / beta
            
            # Min capacity check (MATLAB: <=2 -> 5 + 100*rand)
            if limit <= 2.0: 
                limit = 5.0 + 100 * np.random.rand() 
            
            self.graph[u][v]['capacity'] = limit
            line_caps[(u, v)] = limit
            
        return line_caps