import numpy as np
import networkx as nx
import math
from typing import List, Dict, Tuple, Optional
from ..core.reference_data import get_reference_stats


class LoadAllocator:
    """
    Assigns active power loads (PL) to load buses in the grid.

    Implements the load-setting methodology from Elyas et al. (2017), which
    mirrors the generation-capacity approach in :class:`CapacityAllocator`:

    1. **Total load:** Compute an aggregate load target, either from a
       deterministic scaling formula or as a fraction (light / medium / heavy)
       of the total installed generation capacity.
    2. **Individual loads:** Sample N_l individual loads from an exponential
       distribution (with ~1% super-large outliers).
    3. **Correlated assignment:** Assign loads to load buses using a 14x14
       empirical 2D probability table (``Tab_2D_load``) that encodes the
       joint distribution of normalized load demand and normalized nodal
       degree.

    Reference systems with pre-computed ``Tab_2D_load`` tables are available
    for NYISO-2935 (id=1), WECC-16944 (id=2), and a third system (id=3).
    A heuristic diagonal-bias table (id=0) is provided as a fallback.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph with ``'bus_type'`` and (for non-deterministic loading
        levels) ``'pg_max'`` node attributes already set.
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
            (``'Gen'``, ``'Load'``, or ``'Conn'``).  For loading levels
            other than ``'D'``, generator nodes must also have a
            ``'pg_max'`` attribute set by :class:`CapacityAllocator`.
        ref_sys_id : int, optional
            Selects which reference system statistics to use for the 2D
            probability table ``Tab_2D_load``:

            - 0 — heuristic diagonal-bias table (no real-grid data).
            - 1 — NYISO-2935 (default).
            - 2 — WECC-16944.
            - 3 — additional reference system.
        """
        self.graph = graph
        self.ref_sys_id = ref_sys_id
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        
        # Identify Load Buses (Bus Type = 2)
        # If attribute missing, defaults to empty list
        self.load_buses = [n for n, d in graph.nodes(data=True) if d.get('bus_type') == 'Load']
        self.n_load = len(self.load_buses)

    def _get_tab_2d_load(self) -> np.ndarray:
        """
        Return the ``Tab_2D_load`` table for the selected reference system.

        The table is a 14x14 matrix representing the empirical joint PDF
        ``Pr((P_bar_ln, k_bar_n) in A)`` discretized into 14 load-demand
        classes (rows, low-to-high) and 14 nodal-degree classes (columns,
        low-to-high).  When ``ref_sys_id=0``, a heuristic table with
        Gaussian-decay diagonal bias ``exp(-0.5 * |r - c|)`` is generated
        instead.

        Returns
        -------
        np.ndarray
            A 14x14 probability matrix (sums to 1).
        """
        # Heuristic fallback (Option 0)
        if self.ref_sys_id == 0:
            # Create a diagonal-heavy matrix similar to the capacity heuristic
            matrix = np.zeros((14, 14))
            for r in range(14):
                for c in range(14):
                    dist = abs(r - c)
                    matrix[r, c] = np.exp(-0.5 * dist) 
            return matrix / np.sum(matrix)

        try:
            stats = get_reference_stats(self.ref_sys_id)
            return stats['Tab_2D_load']
        except ValueError as e:
            print(f"Warning: {e} Defaulting to Reference System 1.")
            stats = get_reference_stats(1)
            return stats['Tab_2D_load']

    def _calculate_total_load(self, loading_level: str, total_gen_capacity: float) -> float:
        """
        Compute the total system load target.

        Four strategies are supported, following Elyas et al. (2017):

        - ``'D'`` — **Deterministic:** scaling formula fitted to realistic
          grids: ``Pl_tot = 10^(-0.2 * log10(N)^2 + 1.98 * log10(N) + 0.58)``.
        - ``'L'`` — **Light loading:** 30–40% of total generation capacity.
        - ``'M'`` — **Medium loading:** 50–60% of total generation capacity.
        - ``'H'`` — **Heavy loading:** 70–80% of total generation capacity.

        Parameters
        ----------
        loading_level : str
            One of ``'D'``, ``'L'``, ``'M'``, ``'H'``.
        total_gen_capacity : float
            Sum of all generator ``pg_max`` values (MW).  Only used when
            ``loading_level`` is not ``'D'``.

        Returns
        -------
        float
            Total system load target (MW).

        Raises
        ------
        ValueError
            If *loading_level* is not one of the four valid options.
        """
        if loading_level == 'D':
            # Formula: 10^(-0.2*(log10(N))^2 + 1.98*log10(N) + 0.58)
            log_n = np.log10(self.n_nodes)
            exponent = -0.2 * (log_n**2) + 1.98 * log_n + 0.58
            return 10**exponent
            
        elif loading_level == 'L':
            factor = 0.3 + np.random.rand() * 0.1
            return total_gen_capacity * factor
            
        elif loading_level == 'M':
            factor = 0.5 + np.random.rand() * 0.1
            return total_gen_capacity * factor
            
        elif loading_level == 'H':
            factor = 0.7 + np.random.rand() * 0.1
            return total_gen_capacity * factor
            
        else:
            raise ValueError("Invalid loading level. Choose 'D', 'L', 'M', or 'H'.")

    def _initial_load_distribution(self, total_load: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Sample individual load demands from the empirical distribution.

        Mirrors :meth:`CapacityAllocator._initial_generation_distribution`.
        More than 99% of load demands follow an exponential distribution;
        ~1% are replaced by super-large outliers drawn uniformly from
        ``[max(P), 3 * max(P)]``.  If the sum deviates more than 5% above
        or 10% below ``total_load``, all values are rescaled proportionally.

        Parameters
        ----------
        total_load : float
            Target aggregate load (MW).

        Returns
        -------
        p_loads : np.ndarray
            Raw (possibly rescaled) load values, shape ``(N_l,)``.
        max_r_pl : float
            Maximum load value, used for normalization.
        normalized_r_pl : np.ndarray
            Loads normalized to [0, 1] by dividing by ``max_r_pl``.
        """
        if self.n_load == 0:
            return np.array([]), 0.0, np.array([])

        mu = total_load / self.n_load
        
        # Base distribution: Exponential
        p_loads = np.random.exponential(scale=mu, size=self.n_load)
        
        # Super large loads (approx 1%)
        n_super = int(round(0.01 * self.n_load))
        if n_super > 0:
            max_p = np.max(p_loads)
            # Uniform random [1*max, 3*max]
            p_super = max_p + (2.0 * np.random.rand(n_super) * max_p)
            
            # Replace random indices
            indices = np.random.choice(self.n_load, size=n_super, replace=False)
            p_loads[indices] = p_super

        # Scaling check
        current_sum = np.sum(p_loads)
        if current_sum > 1.05 * total_load or current_sum < 0.90 * total_load:
            p_loads = p_loads * (total_load / current_sum)
            
        max_r_pl = np.max(p_loads)
        normalized_r_pl = p_loads / max_r_pl
        
        return p_loads, max_r_pl, normalized_r_pl

    def _assignment_logic(self, norm_deg_pairs: np.ndarray, norm_vals: np.ndarray, tab_2d: np.ndarray) -> np.ndarray:
        """
        Assign normalized values to buses via 2D-binning.

        Shared logic with :meth:`CapacityAllocator._assignment_logic`.  See
        that method's docstring for the full algorithm description.  In brief:

        1. Scale the 14x14 probability table to integer target counts.
        2. Derive degree-bin (column sums) and value-bin (row sums) targets.
        3. Sort buses by normalized degree, partition into 14 bins.
        4. Sort normalized values, partition into 14 bins.
        5. For each (degree bin, value bin) pair—iterating degree bins
           1→14, value bins 14→1—randomly assign the target count of
           values to unassigned buses.

        Parameters
        ----------
        norm_deg_pairs : np.ndarray
            Shape ``(N, 2)`` — columns are ``[NodeID, NormalizedDegree]``.
        norm_vals : np.ndarray
            Shape ``(N,)`` — normalized load values in [0, 1].
        tab_2d : np.ndarray
            A 14x14 joint-probability matrix (rows=value classes,
            columns=degree classes).

        Returns
        -------
        np.ndarray
            Shape ``(N, 3)`` — columns are
            ``[NodeID, NormalizedDegree, NormalizedValue]``.
        """
        n_items = len(norm_vals)
        
        # 1. Calculate target counts
        tab_2d_n = np.round(tab_2d * n_items).astype(int)
        
        diff = n_items - np.sum(tab_2d_n)
        if diff != 0:
            flat_idx = np.argmax(tab_2d_n)
            r_idx, c_idx = np.unravel_index(flat_idx, tab_2d_n.shape)
            tab_2d_n[r_idx, c_idx] += diff

        # 2. Binning Targets
        n_k_targets = np.sum(tab_2d_n, axis=0) # Degree bins (Cols)
        n_v_targets = np.sum(tab_2d_n, axis=1) # Value bins (Rows)
        
        # 3. Sort and Bin Nodes by Degree
        sorted_nodes = norm_deg_pairs[norm_deg_pairs[:, 1].argsort()]
        
        k_bins = []
        current_idx = 0
        for count in n_k_targets:
            end_idx = current_idx + count
            k_bins.append(sorted_nodes[current_idx:end_idx].copy())
            current_idx = end_idx
            
        # 4. Sort and Bin Values (Loads)
        sorted_vals = np.sort(norm_vals)
        
        v_bins = []
        current_idx = 0
        for count in n_v_targets:
            end_idx = current_idx + count
            v_bins.append(list(sorted_vals[current_idx:end_idx]))
            current_idx = end_idx

        # 5. Assignment Loop
        for k_idx in range(14): 
            for v_idx in range(13, -1, -1):
                count = tab_2d_n[v_idx, k_idx]
                if count > 0:
                    vals_to_assign = []
                    source_bin = v_bins[v_idx]
                    take = min(count, len(source_bin))
                    
                    for _ in range(take):
                        rand_i = np.random.randint(0, len(source_bin))
                        vals_to_assign.append(source_bin.pop(rand_i))
                    
                    if k_bins[k_idx].shape[1] < 3:
                        k_bins[k_idx] = np.hstack((k_bins[k_idx], np.full((len(k_bins[k_idx]), 1), -1.0)))
                        
                    empty_slots = np.where(k_bins[k_idx][:, 2] == -1.0)[0]
                    fill_indices = empty_slots[:take]
                    k_bins[k_idx][fill_indices, 2] = vals_to_assign

        final_list = []
        for bin_arr in k_bins:
            if bin_arr.shape[1] == 3:
                final_list.append(bin_arr)
                
        if not final_list:
            return np.array([])
            
        return np.vstack(final_list)

    def allocate(self, loading_level: str = 'H') -> Dict[int, float]:
        """
        Run the full load-allocation pipeline.

        Executes the three-stage methodology (analogous to
        :meth:`CapacityAllocator.allocate`):

        1. Compute total system load from ``loading_level`` strategy.
        2. Sample individual loads, normalize them and the load-bus nodal
           degrees by their respective maxima:
           ``P_bar = P / max(P)``, ``k_bar = k / max(k)``.
        3. Assign normalized loads to load buses via 2D binning using
           ``Tab_2D_load``.
        4. Denormalize: ``PL = P_bar * max(P)``.

        Parameters
        ----------
        loading_level : str, optional
            Loading strategy (default ``'H'``):

            - ``'D'`` — deterministic scaling formula.
            - ``'L'`` — light (30–40% of generation capacity).
            - ``'M'`` — medium (50–60% of generation capacity).
            - ``'H'`` — heavy (70–80% of generation capacity).

        Returns
        -------
        dict[int, float]
            Mapping from load node ID to its assigned active power load (MW).
        """
        if self.n_load == 0:
            print("No load buses found. Skipping load allocation.")
            return {}

        # 1. Get Total Generation Capacity from Grid
        gen_caps = [d.get('pg_max', 0.0) for n, d in self.graph.nodes(data=True) if d.get('bus_type') == 'Gen']
        total_gen_capacity = sum(gen_caps)
        
        if total_gen_capacity == 0 and loading_level != 'D':
            print("Warning: Total generation capacity is 0. Switching to 'D' (Deterministic) loading.")
            loading_level = 'D'

        # 2. Calculate Total Load
        total_load = self._calculate_total_load(loading_level, total_gen_capacity)
        
        print(f"Allocating Loads for {self.n_load} load buses.")
        print(f"Total System Load Target: {total_load:.2f} MW (Level: {loading_level})")

        # 3. Get Load Bus Degrees
        load_degrees = []
        for n in self.load_buses:
            load_degrees.append([n, self.graph.degree(n)])
        load_degrees = np.array(load_degrees)
        
        if len(load_degrees) == 0:
            return {}

        max_deg = np.max(load_degrees[:, 1])
        norm_load_degrees = load_degrees.copy().astype(float)
        norm_load_degrees[:, 1] = norm_load_degrees[:, 1] / max_deg

        # 4. Initial Load Distribution
        p_loads_actual, max_r_pl, norm_r_pl = self._initial_load_distribution(total_load)
        
        # 5. Get Reference Table
        tab_2d_load = self._get_tab_2d_load()
        
        # 6. Assignment
        norm_assignment = self._assignment_logic(norm_load_degrees, norm_r_pl, tab_2d_load)
        
        # 7. Denormalize
        final_allocation = {}
        for row in norm_assignment:
            node_id = int(row[0])
            norm_val = row[2]
            actual_val = norm_val * max_r_pl
            final_allocation[node_id] = actual_val
            
        return final_allocation

    def allocate_reactive(
        self,
        active_loads: Dict[int, float],
        pf_min: float = 0.85,
        pf_max: float = 0.97,
    ) -> Dict[int, float]:
        """
        Derive reactive power loads from active loads using power factors.

        For each load bus, a power factor is sampled uniformly from
        ``[pf_min, pf_max]`` (lagging) and the reactive load is computed as
        ``ql = pl * tan(arccos(pf))``.

        Theory reference:
        https://www.phasetophase.nl/book/book_2_9.html#_9.5.2

        Parameters
        ----------
        active_loads : dict[int, float]
            Mapping from load node ID to active power load (MW), as returned
            by :meth:`allocate`.
        pf_min : float, optional
            Minimum power factor (default 0.85).
        pf_max : float, optional
            Maximum power factor (default 0.97).

        Returns
        -------
        dict[int, float]
            Mapping from load node ID to reactive power load (Mvar).
        """
        reactive_loads = {}
        for node_id, pl in active_loads.items():
            pf = np.random.uniform(pf_min, pf_max)
            ql = pl * np.tan(np.arccos(pf))
            reactive_loads[node_id] = ql
        return reactive_loads