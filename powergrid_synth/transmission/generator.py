r"""
Main module for generating synthetic grid topology based on Chung-Lu-Chain start graph model. 

Inputs: 
    
    * Desired same-voltage degrees: $\mathbf{d}^{X_1}, \dots, \mathbf{d}^{X_k}$ for each same-voltage graph $X_i$.
    * Desured same-voltage diameters: $\delta^{X_1},\dots \delta^{X_k}$. 
    * Transformer degrees: $\mathbf{t}[X_i,X_j]$ for each pair $i,j\in\{1,\dots,k\}$
    
Output: 
    Full edge list. 
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional

from .preprocessing import Preprocessor
from .edge_creation import EdgeCreator
from .transformer_edges import TransformerConnector
from ..core.grid_graph import PowerGridGraph


class PowerGridGenerator:
    r"""
    Generative model for an entire power grid graph on *k* voltage levels.

    Implements Algorithm 4 (CLCStars) from `Aksoy et al. (2018)
    <https://doi.org/10.1093/comnet/cny016>`_.
    Phase 1 generates each same-voltage subgraph via the CLC model,
    and Phase 2 inserts transformer edges via the random-star model.

    Parameters
    ----------
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            
        self.preprocessor = Preprocessor()
        self.edge_creator = EdgeCreator()
        self.transformer_connector = TransformerConnector()

    def _generate_subgraphs(self, k: int, 
                          degrees_by_level: List[List[int]], 
                          diameters_by_level: List[int],
                          level_offsets: List[int],
                          level_node_counts: List[int],
                          all_edges: dict):
        r"""
        Generate same-voltage subgraphs (Phase 1) for all *k* levels.

        For each voltage level, runs :meth:`Preprocessor.run_setup` followed
        by :meth:`EdgeCreator.generate_edges`, converting local node IDs to
        global IDs using cumulative offsets.

        Parameters
        ----------
        k : int
            Number of voltage levels.
        degrees_by_level : list of list of int
            Desired degree sequences, one per voltage level.
        diameters_by_level : list of int
            Desired diameters, one per voltage level.
        level_offsets : list of int
            *Mutated in place* — filled with the global node-ID offset for
            each level.
        level_node_counts : list of int
            *Mutated in place* — filled with the actual node count (after
            degree-sequence inflation) for each level.
        all_edges : dict
            *Mutated in place* — ``{(u, v): {'type': 'line'}}`` entries are
            added for each generated edge.

        Example
        -------
        .. code-block:: python
            :linenos:
            
            # Level 0 generated (5 nodes) -> Local IDs: 0, 1, 2, 3, 4
            level_offsets[0] = 0
            current_global_offset becomes 5

            # Level 1 generated (3 nodes) -> Local IDs: 0, 1, 2
            # We shift them by current_global_offset (5):
            # Global IDs: 5, 6, 7
            level_offsets[1] = 5
            current_global_offset becomes 8
        """
        current_global_offset = 0
        for i in range(k):
            print(f"Generating Level {i}...")
            
            d_input = degrees_by_level[i]
            delta_input = diameters_by_level[i]
            
            d_prime, v, D, S = self.preprocessor.run_setup(d_input, delta_input)
            local_edges = self.edge_creator.generate_edges(d_prime, v, D, S)
            
            n_nodes = len(d_prime)
            level_node_counts[i] = n_nodes
            level_offsets[i] = current_global_offset
            
            for u, w in local_edges:
                global_u = int(u) + current_global_offset
                global_w = int(w) + current_global_offset
                edge = tuple(sorted((global_u, global_w)))
                all_edges[edge] = {'type': 'line'}
                
            current_global_offset += n_nodes
            print(f"  -> Level {i} Complete. Nodes: {n_nodes}, Edges: {len(local_edges)}")

    def _generate_transformer_connections(self, k: int,
                                        transformer_degrees: Dict[Tuple[int, int], Tuple[List[int], List[int]]],
                                        level_node_counts: List[int],
                                        level_offsets: List[int],
                                        all_edges: dict):
        r"""
        Insert transformer edges between voltage levels (Phase 2).

        For each pair of levels ``(i, j)``, runs
        :meth:`TransformerConnector.generate_transformer_edges` and converts
        local node IDs to global IDs.

        Parameters
        ----------
        k : int
            Number of voltage levels.
        transformer_degrees : dict
            Mapping ``(i, j) -> (t_i_j, t_j_i)`` of transformer degree lists.
        level_node_counts : list of int
            Actual node counts per level (from Phase 1).
        level_offsets : list of int
            Global node-ID offsets per level (from Phase 1).
        all_edges : dict
            *Mutated in place* — ``{(u, v): {'type': 'transformer'}}`` entries
            are added.
        """
        print("Generating Transformer Connections...")
        for i in range(k):
            for j in range(i + 1, k):
                
                if (i, j) not in transformer_degrees:
                    continue
                
                print(f"  -> Connecting Level {i} <-> Level {j}")
                
                t_i_j_input, t_j_i_input = transformer_degrees[(i, j)]
                
                actual_n_i = level_node_counts[i]
                t_i_j = list(t_i_j_input)
                if len(t_i_j) < actual_n_i:
                    t_i_j.extend([0] * (actual_n_i - len(t_i_j)))
                elif len(t_i_j) > actual_n_i:
                    t_i_j = t_i_j[:actual_n_i]
                
                actual_n_j = level_node_counts[j]
                t_j_i = list(t_j_i_input)
                if len(t_j_i) < actual_n_j:
                    t_j_i.extend([0] * (actual_n_j - len(t_j_i)))
                elif len(t_j_i) > actual_n_j:
                    t_j_i = t_j_i[:actual_n_j]

                trans_edges = self.transformer_connector.generate_transformer_edges(t_i_j, t_j_i)
                
                offset_i = level_offsets[i]
                offset_j = level_offsets[j]
                
                for u_local, v_local in trans_edges:
                    u_global = int(u_local) + offset_i
                    v_global = int(v_local) + offset_j
                    edge = tuple(sorted((u_global, v_global)))
                    all_edges[edge] = {'type': 'transformer'}

    def generate_grid(self, 
                      degrees_by_level: List[List[int]], 
                      diameters_by_level: List[int], 
                      transformer_degrees: Dict[Tuple[int, int], Tuple[List[int], List[int]]],
                      keep_lcc: bool = False) -> PowerGridGraph:
        r"""
        Generate a multi-level power grid graph (CLCStars, Algorithm 4).

        Runs Phase 1 (CLC for each voltage level) and Phase 2 (random-star
        transformer edges for each pair of levels), then combines results.

        Parameters
        ----------
        degrees_by_level : list of list of int
            Desired degree sequences :math:`\mathbf{d}^{X_1},\dots,\mathbf{d}^{X_k}`,
            one per voltage level.
        diameters_by_level : list of int
            Desired diameters :math:`\delta^{X_1},\dots,\delta^{X_k}`,
            one per voltage level.
        transformer_degrees : dict
            Mapping ``(i, j) -> (t_i_j, t_j_i)`` where ``t_i_j`` is the
            transformer degree list for level-*i* nodes toward level *j*,
            and ``t_j_i`` is the reverse.
        keep_lcc : bool, optional
            If True, return only the largest connected component with
            contiguous node IDs. Default is False.

        Returns
        -------
        PowerGridGraph
            The generated grid graph with ``voltage_level`` node attributes
            and ``type`` (``'line'`` or ``'transformer'``) edge attributes.
        """
        k = len(degrees_by_level)
        all_edges = {}
        
        level_offsets = [0] * k
        level_node_counts = [0] * k
        
        print(f"--- Starting Generation for {k} Voltage Levels ---")

        # =========================================================
        # Lines 3-7: Create each same-voltage subgraph
        # =========================================================
        self._generate_subgraphs(k, degrees_by_level, diameters_by_level, 
                               level_offsets, level_node_counts, all_edges)

        # =========================================================
        # Lines 8-13: Insert transformer edges between levels
        # =========================================================
        self._generate_transformer_connections(k, transformer_degrees, 
                                             level_node_counts, level_offsets, all_edges)

        # =========================================================
        # Build Final Graph
        # =========================================================
        G = PowerGridGraph() # Use Custom Class directly
        G.add_edges_from([(u, v, attr) for (u, v), attr in all_edges.items()])
        
        for i in range(k):
            start = level_offsets[i]
            end = start + level_node_counts[i]
            for node_id in range(start, end):
                G.add_node(node_id, voltage_level=i)

        # =========================================================
        # Post-Processing: Largest Connected Component (Optional)
        # =========================================================
        if keep_lcc:
            if G.number_of_nodes() > 0:
                print("Filtering for Largest Connected Component (LCC)...")
                original_n = G.number_of_nodes()
                largest_cc_nodes = max(nx.connected_components(G), key=len)
                
                # We need to preserve the PowerGridGraph class type
                sub_view = G.subgraph(largest_cc_nodes)
                G_new = PowerGridGraph()
                G_new.add_nodes_from(sub_view.nodes(data=True))
                G_new.add_edges_from(sub_view.edges(data=True))
                # Update graph-level attributes
                G_new.graph.update(sub_view.graph)
                
                # Relabel nodes to ensure contiguous indices starting from 0
                mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G_new.nodes()))}
                G = nx.relabel_nodes(G_new, mapping)
                
                new_n = G.number_of_nodes()
                print(f"  -> Kept {new_n} nodes (removed {original_n - new_n} isolated nodes)")
            else:
                print("Warning: Graph is empty, cannot filter for LCC.")
        
        return G