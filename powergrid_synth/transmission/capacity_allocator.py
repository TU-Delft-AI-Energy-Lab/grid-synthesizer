"""
Assigns generation capacities to generator buses using a statistical approach
based on Elyas et al. (2017), "On the Statistical Settings of Generation and
Load in a Synthetic Grid Modeling" (https://arxiv.org/abs/1706.09294).

The methodology uses the exponential distribution of individual generation
capacities and the non-trivial correlation between generation capacity and
nodal degree observed in realistic grids. An analogous approach for load
allocation is implemented in :mod:`load_allocator`.

Ported from the MATLAB SynGrid toolbox (``sg_refsys_stat.m``).
"""

import numpy as np
import networkx as nx
import math
from typing import List, Dict, Tuple, Optional
from ..core.reference_data import get_reference_stats

class CapacityAllocator:
    """
    Assigns generation capacities (PgMax) to generator buses in the grid.

    Implements the three-stage methodology from Elyas et al. (2017):

    1. **Total generation:** Compute the aggregate generation capacity from a
       scaling law fitted to realistic grids:
       ``log Pg_tot(N) = -0.21 * log^2(N) + 2.06 * log(N) + 0.66``.
    2. **Individual capacities:** Sample N_g individual capacities from an
       exponential distribution (with ~1% super-large outliers to capture the
       observed heavy tail).
    3. **Correlated assignment:** Assign capacities to generator buses using a
       14x14 empirical 2D probability table (``Tab_2D_Pgmax``) that encodes
       the joint distribution of normalized capacity and normalized nodal
       degree. This preserves the observed Pearson correlation
       rho(P_bar, k_bar) in [0.25, 0.5] between capacity and degree.

    Reference systems with pre-computed ``Tab_2D_Pgmax`` tables are available
    for NYISO-2935 (id=1), WECC-16944 (id=2), and a third system (id=3).
    A heuristic diagonal-bias table (id=0) is provided as a fallback.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph with ``'bus_type'`` node attributes already set
        (by :class:`BusTypeAllocator`).
    ref_sys_id : int
        Reference system for statistical tables (0=heuristic, 1=NYISO-2935,
        2=WECC-16944, 3=additional reference).

    References
    ----------
    .. [1] S. H. Elyas, Z. Wang, R. J. Thomas, "On the Statistical Settings
       of Generation and Load in a Synthetic Grid Modeling," arXiv:1706.09294,
       2017.
    """

    def __init__(self, graph: nx.Graph, ref_sys_id: int = 1):
        """
        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph whose nodes carry a ``'bus_type'`` attribute
            (``'Gen'``, ``'Load'``, or ``'Conn'``).  This attribute must be
            set before instantiation, typically via :class:`BusTypeAllocator`.
        ref_sys_id : int, optional
            Selects which reference system statistics to use for the 2D
            probability table ``Tab_2D_Pgmax``:

            - 0 — heuristic diagonal-bias table (no real-grid data).
            - 1 — NYISO-2935 (default).
            - 2 — WECC-16944.
            - 3 — additional reference system.
        """
        self.graph = graph
        self.ref_sys_id = ref_sys_id
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        
        # Identify Generator Buses (Assuming 'bus_type' attribute exists)
        # 1=Gen, 2=Load, 3=Conn
        # If attribute missing, defaults to empty list (user must run BusTypeAllocator first)
        self.gen_buses = [n for n, d in graph.nodes(data=True) if d.get('bus_type') == 'Gen']
        self.n_gen = len(self.gen_buses)

    def _generate_heuristic_tab_2d(self) -> np.ndarray:
        """
        Generate a heuristic 14x14 joint-probability table as a fallback.

        When no reference system data is selected (``ref_sys_id=0``), this
        method produces a synthetic table using a Gaussian-decay diagonal
        bias: ``weight(r, c) = exp(-0.5 * |r - c|)``, where rows represent
        capacity classes and columns represent degree classes (both ordered
        low-to-high).  The matrix is normalized to sum to 1.

        This simulates the positive correlation between nodal degree and
        generation capacity observed in realistic grids, without relying on
        empirical data from a specific reference system.

        Returns
        -------
        np.ndarray
            A 14x14 probability matrix (sums to 1).
        """
        # Create a matrix with some diagonal weight to simulate correlation
        # Rows = Capacity Classes (low to high), Cols = Degree Classes (low to high)
        matrix = np.zeros((14, 14))
        for r in range(14):
            for c in range(14):
                # Distance from diagonal
                dist = abs(r - c)
                # higher probability near diagonal
                matrix[r, c] = np.exp(-0.5 * dist) 
        
        # Normalize to sum to 1
        return matrix / np.sum(matrix)

    def _get_default_tab_2d(self) -> np.ndarray:
        """
        Return the ``Tab_2D_Pgmax`` table for the selected reference system.

        The table is a 14x14 matrix representing the empirical joint PDF
        ``Pr((P_bar_gn_max, k_bar_n) in A)`` discretized into 14 capacity
        classes (rows, low-to-high) and 14 nodal-degree classes (columns,
        low-to-high).  Tables for real reference systems are loaded from
        :func:`reference_data.get_reference_stats` (ported from the MATLAB
        SynGrid file ``sg_refsys_stat.m``).

        Returns
        -------
        np.ndarray
            A 14x14 probability matrix.
        """
        # Option 0: Use the heuristic generator (previous default)
        if self.ref_sys_id == 0:
            return self._generate_heuristic_tab_2d()

        try:
            stats = get_reference_stats(self.ref_sys_id)
            return stats['Tab_2D_Pgmax']
        except ValueError as e:
            print(f"Warning: {e} Defaulting to Reference System 1.")
            stats = get_reference_stats(1)
            return stats['Tab_2D_Pgmax']

    def _initial_generation_distribution(self, total_gen: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Sample individual generation capacities from the empirical distribution.

        Following Elyas et al. (2017), more than 99% of generation units in
        realistic grids follow an exponential distribution.  About 1% have
        extremely large capacities that fall outside the exponential range.

        The procedure is:

        1. Draw ``N_g`` samples from ``Exp(mu)`` where ``mu = total_gen / N_g``.
        2. Replace ~1% of the samples with "super-large" values drawn
           uniformly from ``[max(P), 3 * max(P)]``.
        3. If the sum deviates more than 5% above or 10% below
           ``total_gen``, rescale all values proportionally.
        4. Normalize by the maximum capacity.

        Parameters
        ----------
        total_gen : float
            Target total generation capacity (MW).

        Returns
        -------
        p_caps : np.ndarray
            Raw (possibly rescaled) capacity values, shape ``(N_g,)``.
        max_r_pgmax : float
            Maximum capacity value, used for normalization.
        normalized_r_pgmax : np.ndarray
            Capacities normalized to [0, 1] by dividing by ``max_r_pgmax``.
        """
        if self.n_gen == 0:
            return np.array([]), 0.0

        # Mean capacity per generator
        mu = total_gen / self.n_gen
        
        # Base distribution: Exponential
        p_caps = np.random.exponential(scale=mu, size=self.n_gen)
        
        # Super large capacities (approx 1%)
        n_super = int(round(0.01 * self.n_gen))
        if n_super > 0:
            max_p = np.max(p_caps)
            # Uniform random [1*max, 3*max] (based on: max(P) + 2*rand*max(P))
            p_super = max_p + (2.0 * np.random.rand(n_super) * max_p)
            
            # Replace random indices with super large values
            indices = np.random.choice(self.n_gen, size=n_super, replace=False)
            p_caps[indices] = p_super

        # Scaling check
        current_sum = np.sum(p_caps)
        if current_sum > 1.05 * total_gen or current_sum < 0.90 * total_gen:
            p_caps = p_caps * (total_gen / current_sum)
            
        max_r_pgmax = np.max(p_caps)
        normalized_r_pgmax = p_caps / max_r_pgmax
        
        return p_caps, max_r_pgmax, normalized_r_pgmax

    def _assignment_logic(self, norm_deg_pairs: np.ndarray, norm_caps: np.ndarray, tab_2d: np.ndarray) -> np.ndarray:
        """
        Assign normalized capacities to buses via 2D-binning (Stage 3).

        This implements the correlated assignment algorithm from Elyas et al.
        (2017) that preserves the empirical joint distribution of normalized
        capacity and normalized nodal degree.  The steps are:

        1. **Scale 2D table to counts:** multiply ``tab_2d`` by ``N_g`` and
           round to integer targets ``n_rc`` for each (capacity-class *r*,
           degree-class *c*) cell.  Adjust rounding so that the total equals
           ``N_g`` exactly.
        2. **Compute marginals:** column sums give degree-bin targets;
           row sums give capacity-bin targets.
        3. **Bin by degree:** sort generators by normalized degree, partition
           into 14 degree bins according to column-sum targets.
        4. **Bin by capacity:** sort normalized capacities, partition into 14
           capacity bins according to row-sum targets.
        5. **Nested assignment loop:** for each degree bin *c* = 1..14 and
           each capacity bin *r* = 14..1 (high to low): randomly sample
           ``n_rc`` capacities from capacity bin *r* (without replacement)
           and assign them to unassigned generators in degree bin *c*.
        6. **Reassemble** all bins into the output array.

        Parameters
        ----------
        norm_deg_pairs : np.ndarray
            Shape ``(N_g, 2)`` — columns are ``[NodeID, NormalizedDegree]``.
        norm_caps : np.ndarray
            Shape ``(N_g,)`` — normalized capacity values in [0, 1].
        tab_2d : np.ndarray
            A 14x14 joint-probability matrix (rows=capacity classes,
            columns=degree classes).

        Returns
        -------
        np.ndarray
            Shape ``(N_g, 3)`` — columns are
            ``[NodeID, NormalizedDegree, NormalizedCapacity]``.
        """
        ng = len(norm_caps)
        
        # 1. Calculate target counts for the 2D table
        tab_2d_ng = np.round(tab_2d * ng).astype(int)
        
        # Fix rounding errors to match exactly ng
        current_sum = np.sum(tab_2d_ng)
        diff = ng - current_sum
        
        if diff != 0:
            # Add/Sub from the max element to assume least disruption
            # Flatten find max linear index
            flat_idx = np.argmax(tab_2d_ng)
            # unravel
            r_idx, c_idx = np.unravel_index(flat_idx, tab_2d_ng.shape)
            tab_2d_ng[r_idx, c_idx] += diff

        # 2. Binning Targets
        # Column sums = targets for Degree bins
        n_k_targets = np.sum(tab_2d_ng, axis=0) # 1x14
        # Row sums = targets for Capacity bins
        n_g_targets = np.sum(tab_2d_ng, axis=1) # 1x14
        
        # 3. Sort and Bin Nodes by Degree
        # Sort by degree (column 1)
        sorted_nodes = norm_deg_pairs[norm_deg_pairs[:, 1].argsort()]
        
        k_bins = []
        current_idx = 0
        for count in n_k_targets:
            end_idx = current_idx + count
            k_bins.append(sorted_nodes[current_idx:end_idx].copy())
            current_idx = end_idx
            
        # 4. Sort and Bin Capacities
        # Sort capacities ascending
        sorted_caps = np.sort(norm_caps)
        
        g_bins = []
        current_idx = 0
        for count in n_g_targets:
            end_idx = current_idx + count
            # Keep as list for easy popping
            g_bins.append(list(sorted_caps[current_idx:end_idx]))
            current_idx = end_idx

        # 5. Assignment Loop (The 2D allocation)
        # Iterate Columns (Degree bins) 1..14
        for k_idx in range(14): # kk
            # Iterate Rows (Capacity bins) 14..1 (High to Low)
            # MATLAB: for gg=14:-1:1
            for g_idx in range(13, -1, -1): # gg
                count = tab_2d_ng[g_idx, k_idx]
                if count > 0:
                    # Take 'count' capacities from G_bin[g_idx]
                    # Since G_bin is sorted, random sampling minimizes bias within the bin
                    caps_to_assign = []
                    source_bin = g_bins[g_idx]
                    
                    # Safety check if bin ran out due to rounding weirdness (shouldn't happen with logic above)
                    take = min(count, len(source_bin))
                    
                    for _ in range(take):
                        # Random sample without replacement strategy from the bin
                        # pop(0) takes smallest, pop(-1) takes largest in that bin
                        # MATLAB uses datasample which is random
                        rand_i = np.random.randint(0, len(source_bin))
                        caps_to_assign.append(source_bin.pop(rand_i))
                    
                    # Assign to nodes in K_bin[k_idx]
                    # We need to fill the 'NormCapacity' column (index 2 in final output, but we construct it)
                    # K_bin currently: [NodeID, NormDeg]
                    # We need to find 'take' nodes in K_bin that don't have capacity yet
                    # Actually, K_bins are partitioned disjointly. We just fill them up.
                    
                    # We add a 3rd column for Capacity to k_bins data structures
                    # Initialize if not present
                    if k_bins[k_idx].shape[1] < 3:
                        # Append column of -1s
                        k_bins[k_idx] = np.hstack((k_bins[k_idx], np.full((len(k_bins[k_idx]), 1), -1.0)))
                        
                    # Find empty slots (-1)
                    empty_slots = np.where(k_bins[k_idx][:, 2] == -1.0)[0]
                    
                    # Fill 'take' slots
                    fill_indices = empty_slots[:take]
                    k_bins[k_idx][fill_indices, 2] = caps_to_assign

        # 6. Reassemble
        # Stack all K bins
        final_list = []
        for bin_arr in k_bins:
            if bin_arr.shape[1] == 3: # valid bin
                final_list.append(bin_arr)
            else:
                # Bin might be empty or untouched?
                pass
                
        if not final_list:
            return np.array([])
            
        gen_setting = np.vstack(final_list)
        return gen_setting

    def allocate(self, tab_2d: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Run the full generation-capacity allocation pipeline.

        Executes the three-stage methodology from Elyas et al. (2017):

        1. Compute total generation capacity from the scaling law:
           ``Pg_tot = 10^(-0.21 * log10(N)^2 + 2.06 * log10(N) + 0.66)``.
        2. Sample individual capacities, normalize them and the generator
           nodal degrees by their respective maxima so that both lie in
           [0, 1]:  ``P_bar = P / max(P)``,  ``k_bar = k / max(k)``.
        3. Assign normalized capacities to generator buses via 2D binning
           using ``Tab_2D_Pgmax``.
        4. Denormalize: ``Pg_max = P_bar * max(P)``.

        Parameters
        ----------
        tab_2d : np.ndarray or None, optional
            Custom 14x14 probability matrix.  If *None*, the default table
            for the selected ``ref_sys_id`` is used.

        Returns
        -------
        dict[int, float]
            Mapping from generator node ID to its assigned capacity
            ``Pg_max`` (MW).
        """
        if self.n_gen == 0:
            print("No generator buses found. Skipping capacity allocation.")
            return {}

        if tab_2d is None:
            tab_2d = self._get_default_tab_2d()

        # 1. Calculate Total Generation
        # Total_Generation = 10^(-0.21*(log10(N))^2+2.06*log10(N)+0.66)
        log10_n = np.log10(self.n_nodes)
        exponent = -0.21 * (log10_n**2) + 2.06 * log10_n + 0.66
        total_gen = 10**exponent
        
        ref_label = "Heuristic (Default)" if self.ref_sys_id == 0 else f"Reference System {self.ref_sys_id}"
        print(f"Allocating Capacity for {self.n_gen} generators.")
        print(f"Total System Capacity Target: {total_gen:.2f} MW using {ref_label}")

        # 2. Get Node Degrees for Generators
        gen_degrees = []
        for n in self.gen_buses:
            gen_degrees.append([n, self.graph.degree(n)])
        gen_degrees = np.array(gen_degrees)
        
        max_node_degree = np.max(gen_degrees[:, 1])
        
        # Normalized Node Degree [NodeID, NormDeg]
        norm_gen_degrees = gen_degrees.copy().astype(float)
        norm_gen_degrees[:, 1] = norm_gen_degrees[:, 1] / max_node_degree

        # 3. Initial Generation Distribution
        # We need actual values P_caps AND normalized values
        p_caps_actual, max_r_pgmax, norm_r_pgmax = self._initial_generation_distribution(total_gen)
        
        # 4. Assignment
        # Returns [NodeID, NormDeg, NormCap]
        norm_assignment = self._assignment_logic(norm_gen_degrees, norm_r_pgmax, tab_2d)
        
        # 5. Denormalize
        # Map NodeID -> NormCap * MaxCap
        final_allocation = {}
        for row in norm_assignment:
            node_id = int(row[0])
            norm_cap = row[2]
            actual_cap = norm_cap * max_r_pgmax
            final_allocation[node_id] = actual_cap
            
        return final_allocation