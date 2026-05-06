r"""
Generation dispatch for synthetic power grids.

This module implements the generation dispatch algorithm from
`Sadeghian et al. (2018) <https://ieeexplore.ieee.org/document/8585532>`_.
It partitions generator units into three groups — **uncommitted**
(:math:`\alpha = 0`), **partially committed** (:math:`0 < \alpha < 1`),
and **fully committed** (:math:`\alpha \approx 1`) — then assigns
participation factors (dispatch factors) and iteratively balances total
generation against total load.  The dispatch factor is defined as

.. math:: \alpha_i = P_{g_i} / P_{g_i}^{\max}, \qquad i = 1, \ldots, N_G

The correlation between normalised capacity and dispatch factor is
reproduced via a 2-D empirical PMF table (``Tab_2D_Pg``).

Ported from the SynGrid MATLAB function ``sg_gen_dist.m``.
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from ..core.reference_data import get_reference_stats


class GenerationDispatcher:
    r"""Assign active-power dispatch to each generator bus.

    The algorithm follows `Sadeghian et al. (2018)
    <https://ieeexplore.ieee.org/document/8585532>`_ (Section III):

    1. **Uncommitted units** (10–20 %): :math:`\alpha = 0`, selected via
       targets drawn from Uniform[0, 0.6].
    2. **Partially committed units** (40–50 %): selected via exponential
       distribution on capacity; dispatch factors assigned through a
       2-D bin-matching table ``Tab_2D_Pg`` (:math:`14 \times 10`).
    3. **Fully committed units** (remainder): :math:`\alpha = 1`.
    4. **Balancing loop**: iteratively adjusts dispatch to match total
       load within 1 % tolerance.

    Parameters
    ----------
    graph : networkx.Graph
        Power grid graph.  Generator nodes must have ``'bus_type' == 'Gen'``
        and ``'pg_max'`` (MW) attributes.  Load nodes must have ``'pl'`` (MW).
    ref_sys_id : int, optional
        Reference system for statistical tables (1 = NYISO-2935,
        2 = WECC-16994, 3 = additional reference).  Default is 1.

    Attributes
    ----------
    alpha_mod : int
        Loading-level flag from the reference system.  When 0 all alphas
        are Uniform[0, 1]; otherwise 0.5 % receive negative dispatch
        (e.g., pumped-storage hydro).
    mu_committed : float
        Exponential-distribution parameter for committed-unit capacities.
    tab_2d_pg : numpy.ndarray
        2-D empirical PMF table (14 capacity bins × 10 alpha bins).
    """

    def __init__(self, graph: nx.Graph, ref_sys_id: int = 1):
        self.graph = graph
        self.ref_sys_id = ref_sys_id

        # Load stats
        try:
            self.stats = get_reference_stats(ref_sys_id)
        except ValueError:
            print(f"Warning: Invalid ref_sys_id {ref_sys_id}. Defaulting to 1.")
            self.stats = get_reference_stats(1)

        self.alpha_mod = self.stats['Alpha_mod']
        self.mu_committed = self.stats['mu_committed']
        self.tab_2d_pg = self.stats['Tab_2D_Pg']

    def _select_uncommitted(self, norm_pg_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Select generators to be uncommitted (:math:`\alpha = 0`).

        Randomly selects 10–20 % of total generator units.  Target
        capacities are drawn from Uniform[0, 0.6] and the unit whose
        normalised capacity is closest to each target is selected.
        This reproduces the empirical observation that uncommitted units
        tend to be small or medium-size (Sadeghian et al., 2018, Sec. III).

        Parameters
        ----------
        norm_pg_max : numpy.ndarray, shape (n, 2)
            Array with columns ``[bus_id, normalised_capacity]``.

        Returns
        -------
        uncommitted : numpy.ndarray, shape (m, 3)
            Uncommitted units: ``[bus_id, norm_cap, alpha=0]``.
        remaining : numpy.ndarray, shape (n-m, 2)
            Units not selected.
        """
        ng = len(norm_pg_max)
        if ng == 0:
            return np.array([]), norm_pg_max

        # 10% to 20% of the total as uncommitted
        ng_uncomm = int(round(ng * (0.10 + np.random.rand() * 0.10)))
        
        uncommitted = []
        remaining = norm_pg_max.copy()
        
        for _ in range(ng_uncomm):
            if len(remaining) == 0: break
            t_uncomm = 0.6 * np.random.rand() # uniform distribution 0-0.6
            dists = np.abs(remaining[:, 1] - t_uncomm) 
            idx = np.argmin(dists) # select the unit closest to the random number 
            row = remaining[idx]
            uncommitted.append([row[0], row[1], 0.0]) # [ID, NormCap, Alpha=0]
            remaining = np.delete(remaining, idx, axis=0)
            
        return np.array(uncommitted), remaining

    def _select_committed(self, norm_pg_max: np.ndarray, total_units_count: int) -> Tuple[np.ndarray, np.ndarray]:
        r"""Select generators to be partially committed (:math:`0 < \alpha < 1`).

        Selects 40–50 % of *total* generator count.  99 % of these are
        chosen by matching to targets drawn from an exponential distribution
        with parameter :math:`\mu_{\text{committed}}`; the remaining 1 %
        are drawn from the extreme tail Uniform[0.5, 1.0], capturing
        super-large units (Sadeghian et al., 2018, Sec. III-A).

        Parameters
        ----------
        norm_pg_max : numpy.ndarray, shape (n, 2)
            Remaining units after uncommitted selection:
            ``[bus_id, normalised_capacity]``.
        total_units_count : int
            Original total number of generator units (before any selection).

        Returns
        -------
        committed : numpy.ndarray, shape (m, 2)
            Committed units: ``[bus_id, norm_cap]``.
        remaining : numpy.ndarray, shape (n-m, 2)
            Units not selected (will become fully committed).
        """
        ng_comm = int(round(total_units_count * (0.4 + np.random.rand() * 0.1)))
        ng_comm = min(ng_comm, len(norm_pg_max))
        
        if ng_comm <= 0:
            return np.array([]), norm_pg_max
        
        ng_99 = int(round(ng_comm * 0.99))
        ng_01 = ng_comm - ng_99
        
        committed = []
        remaining = norm_pg_max.copy()
        
        for _ in range(ng_99):
            if len(remaining) == 0: break
            t_unit = np.random.exponential(self.mu_committed) # the exp distribution of the commited units' capacities
            idx = np.argmin(np.abs(remaining[:, 1] - t_unit))
            committed.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
        for _ in range(ng_01):
            if len(remaining) == 0: break
            t_comm = 0.5 + 0.5 * np.random.rand() # the extreme value of the 1% commited units' capacities
            idx = np.argmin(np.abs(remaining[:, 1] - t_comm))
            committed.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
        return np.array(committed), remaining

    def _generate_alphas(self, n_comm: int) -> np.ndarray:
        r"""Generate dispatch factors for partially committed units.

        When ``alpha_mod == 0`` (e.g. NYISO), all :math:`\alpha` values are
        drawn from Uniform[0, 1].  When ``alpha_mod != 0`` (e.g. WECC),
        99.5 % are Uniform[0, 1] and 0.5 % are negative, representing
        reverse dispatch such as pumped-storage hydro.

        Parameters
        ----------
        n_comm : int
            Number of committed units requiring :math:`\alpha` values.

        Returns
        -------
        numpy.ndarray, shape (n_comm, 1)
            Dispatch-factor values.
        """
        if n_comm == 0: return np.array([])
        if self.alpha_mod == 0:
            return np.random.rand(n_comm, 1)
        else:
            n_995 = int(round(n_comm * 0.995))
            n_005 = n_comm - n_995
            a1 = np.random.rand(n_995, 1) # 99.5% --- uniform[0, 1]
            if n_005 > 0:
                a2 = -np.random.rand(n_005, 1) # 0.5% --- negative dispatch value 
                return np.vstack((a1, a2))
            return a1

    def _assign_alphas(self, units: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        r"""Assign dispatch factors to committed units via 2-D bin matching.

        Units are sorted by normalised capacity and alphas by value, then
        distributed into bins defined by ``Tab_2D_Pg`` (14 capacity bins
        :math:`\times` 10 alpha bins).  Within each bin, units and alphas
        are paired randomly (high-to-low bin traversal).  Any leftovers
        are paired sequentially as a fallback.

        This reproduces the empirical joint distribution
        :math:`f(\bar{P}_{g}^{\max}, \alpha)` from the reference system
        (Sadeghian et al., 2018, Table I).

        Parameters
        ----------
        units : numpy.ndarray, shape (n, 2)
            ``[bus_id, normalised_capacity]``.
        alphas : numpy.ndarray, shape (n, 1)
            Dispatch-factor values from :meth:`_generate_alphas`.

        Returns
        -------
        numpy.ndarray, shape (m, 3)
            ``[bus_id, normalised_capacity, alpha]``.
        """
        n_items = len(units)
        if n_items == 0: return np.array([])
        
        tab_n = np.round(self.tab_2d_pg * n_items).astype(int)
        diff = n_items - np.sum(tab_n)
        if diff != 0:
            flat_idx = np.argmax(tab_n)
            r, c = np.unravel_index(flat_idx, tab_n.shape)
            tab_n[r, c] += diff
            
        n_cap_targets = np.sum(tab_n, axis=1)
        n_alpha_targets = np.sum(tab_n, axis=0)
        
        sorted_units = units[units[:, 1].argsort()]
        unit_bins = []
        curr = 0
        for count in n_cap_targets:
            end = curr + count
            unit_bins.append(list(sorted_units[curr:end]))
            curr = end
            
        sorted_alphas = np.sort(alphas.flatten())
        alpha_bins = []
        curr = 0
        for count in n_alpha_targets:
            end = curr + count
            alpha_bins.append(list(sorted_alphas[curr:end]))
            curr = end
            
        final_list = []
        for r in range(13, -1, -1):
            for c in range(10):
                count = tab_n[r, c]
                if count > 0:
                    u_bin = unit_bins[r]
                    a_bin = alpha_bins[c]
                    take = min(count, len(u_bin), len(a_bin))
                    for _ in range(take):
                        if u_bin and a_bin:
                            u_idx = np.random.randint(0, len(u_bin))
                            unit = u_bin.pop(u_idx)
                            a_idx = np.random.randint(0, len(a_bin))
                            alpha_val = a_bin.pop(a_idx)
                            final_list.append([unit[0], unit[1], alpha_val])

        # Leftovers fallback
        remaining_units = [u for bin in unit_bins for u in bin]
        remaining_alphas = [a for bin in alpha_bins for a in bin]
        if remaining_units and remaining_alphas:
             min_len = min(len(remaining_units), len(remaining_alphas))
             for i in range(min_len):
                 final_list.append([remaining_units[i][0], remaining_units[i][1], remaining_alphas[i]])

        return np.array(final_list) if final_list else np.array([])

    def dispatch(self) -> Dict[int, float]:
        r"""Run the full generation dispatch pipeline.

        Implements the algorithm of Sadeghian et al. (2018):

        1. Collect generator buses and normalise capacities by
           :math:`P^{\max}_{g_{\max}}`.
        2. Partition generators into **uncommitted**
           (:math:`\alpha = 0`), **partially committed**
           (:math:`0 < \alpha < 1`), and **fully committed**
           (:math:`\alpha = 1`).
        3. Assign dispatch factors to partially committed units via
           the 2-D bin-matching table ``Tab_2D_Pg``.
        4. Iteratively balance total generation against total load
           (1 % tolerance, up to 50 iterations) by scaling committed
           :math:`\alpha` values and toggling uncommitted / full-load
           units on or off.
        5. Convert normalised dispatch back to MW:
           :math:`P_{g_i} = \alpha_i \cdot \bar{P}_{g_i}^{\max} \cdot P^{\max}_{g_{\max}}`.

        Returns
        -------
        dict
            Mapping of generator bus ID to dispatched active power (MW).
        """
        # 1. Prepare Buses into (Id, Gen) and (Id, Load)
        gen_data = []
        for n, d in self.graph.nodes(data=True):
            if d.get('bus_type') == 'Gen':
                gen_data.append([n, d.get('pg_max', 0.0)])
                
        if not gen_data: return {}
        gen_arr = np.array(gen_data)
        max_pg = np.max(gen_arr[:, 1])
        if max_pg == 0: max_pg = 1.0
        
        norm_pg = gen_arr.copy()
        norm_pg[:, 1] = norm_pg[:, 1] / max_pg
        
        total_load = sum(d.get('pl', 0.0) for n, d in self.graph.nodes(data=True) if d.get('bus_type') == 'Load')
        norm_total_load = total_load / max_pg
        
        # 2. Partitioning --- uncommited, partially commited, and fully committed
        uncomm_units, remaining = self._select_uncommitted(norm_pg)
        comm_units_raw, remaining = self._select_committed(remaining, len(gen_arr))
        
        full_units = []
        for row in remaining:
            full_units.append([row[0], row[1], 1.0])
        full_units = np.array(full_units)

        # 3. Alpha Assignment for the partially committed
        if len(comm_units_raw) > 0:
            alphas = self._generate_alphas(len(comm_units_raw))
            comm_units = self._assign_alphas(comm_units_raw, alphas)
        else:
            comm_units = np.array([])
            
        # 4. Balancing Logic
        def calculate_current_total():
            g_u = np.sum(uncomm_units[:, 1] * uncomm_units[:, 2]) if len(uncomm_units) > 0 else 0
            g_c = np.sum(comm_units[:, 1] * comm_units[:, 2]) if len(comm_units) > 0 else 0
            g_f = np.sum(full_units[:, 1] * full_units[:, 2]) if len(full_units) > 0 else 0
            return g_u + g_c + g_f

        for _ in range(50): # Max Iterations
            current_gen = calculate_current_total()
            diff = current_gen - norm_total_load
            
            if abs(diff) < 0.01 * norm_total_load: # 1% Tolerance
                break
            
            if diff > 0: # Excess Gen
                # 1. Try scaling down committed units
                if len(comm_units) > 0:
                    comm_load = np.sum(comm_units[:, 1] * comm_units[:, 2])
                    # We need to reduce by 'diff'. 
                    # new_comm_load = comm_load - diff
                    # ratio = new_comm_load / comm_load
                    if comm_load > 1e-6:
                        ratio = max(0, (comm_load - diff) / comm_load)
                        # Don't drop too drastically in one step to keep stability
                        ratio = max(ratio, 0.5) 
                        comm_units[:, 2] *= ratio
                        continue
                
                # 2. If committed scaling didn't help enough, turn off full units
                if len(full_units) > 0:
                    # Find ON units
                    on_indices = np.where(full_units[:, 2] > 0.01)[0]
                    if len(on_indices) > 0:
                        # Turn off the largest one
                        subset_idx = np.argmax(full_units[on_indices, 1])
                        full_units[on_indices[subset_idx], 2] = 0.0
                        continue
                
                # 3. Last resort: turn off committed units completely
                if len(comm_units) > 0:
                     on_indices = np.where(comm_units[:, 2] > 0.01)[0]
                     if len(on_indices) > 0:
                         comm_units[on_indices[0], 2] = 0.0
            
            else: # Deficit Gen (diff < 0)
                # 1. Try scaling up committed units
                if len(comm_units) > 0:
                    comm_load = np.sum(comm_units[:, 1] * comm_units[:, 2])
                    capacity = np.sum(comm_units[:, 1]) # Max possible if alpha=1
                    headroom = capacity - comm_load
                    
                    if headroom > 1e-6:
                        # We need to increase by abs(diff)
                        # But we can't just multiply alphas linearly because they cap at 1.0
                        # Simple heuristic: multiply by 1.1 or calculated ratio
                        comm_units[:, 2] *= 1.1
                        comm_units[:, 2] = np.minimum(comm_units[:, 2], 1.0)
                        
                        # If we actually gained something, continue
                        new_load = np.sum(comm_units[:, 1] * comm_units[:, 2])
                        if new_load > comm_load + 1e-6:
                            continue

                # 2. Turn on Uncommitted units
                if len(uncomm_units) > 0:
                    off_indices = np.where(uncomm_units[:, 2] < 0.01)[0]
                    if len(off_indices) > 0:
                        # Turn on largest available
                        subset_idx = np.argmax(uncomm_units[off_indices, 1])
                        uncomm_units[off_indices[subset_idx], 2] = 1.0 # Set to Full
                        continue

                # 3. Turn on any Full units that were turned off
                if len(full_units) > 0:
                    off_indices = np.where(full_units[:, 2] < 0.01)[0]
                    if len(off_indices) > 0:
                        full_units[off_indices[0], 2] = 1.0
                        continue
                        
        # 5. Result Export
        result = {}
        for group in [uncomm_units, comm_units, full_units]:
            if len(group) > 0:
                for row in group:
                    gen_mw = row[1] * row[2] * max_pg
                    result[int(row[0])] = gen_mw
        
        return result