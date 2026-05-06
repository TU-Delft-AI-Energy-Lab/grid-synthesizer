r"""
Transformer edge generation between different-voltage subgraphs.

Implements Algorithm 3 (Stars) from `Aksoy et al. (2018)
<https://doi.org/10.1093/comnet/cny016>`_ (arXiv:1711.11098, Section 4.4).
"""
import numpy as np
import random
from typing import List, Set, Tuple, Dict

class TransformerConnector:
    r"""
    Generate transformer edges between two same-voltage subgraphs.

    Transformer subgraphs in real power grids consist almost entirely of
    disjoint :math:`k`-star graphs.  This class replicates that structure
    by creating random stars from the desired transformer degree sequences,
    with a bipartite Chung-Lu fallback for leftover vertices.

    See Algorithm 3 in `Aksoy et al. (2018)
    <https://doi.org/10.1093/comnet/cny016>`_.
    """

    def generate_transformer_edges(self, t_xy: List[int], t_yx: List[int]) -> List[Tuple[int, int]]:
        r"""
        Generate transformer edges between voltage levels X and Y.

        Implements **Stars**\ (:math:`\mathbf{t}[X,Y],\;\mathbf{t}[Y,X]`)
        :math:`\to E` (Algorithm 3).

        The algorithm proceeds in four stages:

        1. Partition vertices into *centers* (degree :math:`\geq 2`) and
           *leaves* (degree :math:`= 1`).
        2. Build :math:`k`-stars centred at high-degree vertices, consuming
           degree-1 vertices from the opposite level as leaves.
        3. Match remaining degree-1 vertices across levels into single edges.
        4. Apply bipartite Chung-Lu on any leftover vertices whose degrees
           could not be realised via stars.

        Parameters
        ----------
        t_xy : list of int
            Transformer degree for each node in subgraph X toward Y.
            ``t_xy[i]`` is the number of transformer edges desired for
            node *i* in X.
        t_yx : list of int
            Transformer degree for each node in subgraph Y toward X.

        Returns
        -------
        list of tuple[int, int]
            Edge list as ``(u, v)`` where *u* is a local index in X and
            *v* is a local index in Y.

        Notes
        -----
        A sufficient condition for all degrees to be matched exactly (no
        leftover Chung-Lu) is:

        .. math::
            \sum_{i:\,t[X,Y]_i\geq 2} t[X,Y]_i
            \;\leq\;
            |\{j\in Y : t[Y,X]_j = 1\}|

        and symmetrically for the other direction.
        """
        # Line 2: Initialize
        edges = set()
        L_x = [] # Leftover bins for X
        L_y = [] # Leftover bins for Y

        # Line 3-4: Categorize vertices
        # We store them as lists to allow efficient random sampling/shuffling
        # Indices are 0-based relative to the input arrays
        
        # Degree 1 vertices (Open) - "Pools"
        I_x_o = [i for i, d in enumerate(t_xy) if d == 1]
        I_y_o = [i for i, d in enumerate(t_yx) if d == 1]
        
        # Degree >= 2 vertices (Centers)
        I_x_c = [i for i, d in enumerate(t_xy) if d >= 2]
        I_y_c = [i for i, d in enumerate(t_yx) if d >= 2]
        
        # Shuffle lists to simulate "Randomly select" efficiently when we pop()
        random.shuffle(I_x_o)
        random.shuffle(I_y_o)
        random.shuffle(I_x_c)
        random.shuffle(I_y_c)

        # --- Lines 7-23 & 25: Run k-STARS(X, Y) ---
        # Create k-stars centered in X with leaves in Y
        for i in I_x_c:
            req_degree = t_xy[i]
            
            # Line 11: Check if enough leaf vertices in Y
            if len(I_y_o) >= req_degree:
                # Line 12-16: Connect i to multiple j's
                for _ in range(req_degree):
                    j = I_y_o.pop() # Remove leaf vertex from pool
                    edges.add((i, j))
                # i is consumed (removed from I_x_c loop effectively)
            else:
                # Line 18-20: Not enough leaves, put i in leftover
                L_x.append(i)

        # --- Lines 7-23 & 26: Run k-STARS(Y, X) ---
        # Create k-stars centered in Y with leaves in X
        for i in I_y_c:
            req_degree = t_yx[i]
            
            # Check if enough leaf vertices in X
            if len(I_x_o) >= req_degree:
                for _ in range(req_degree):
                    j = I_x_o.pop() # Remove leaf
                    # Edge is (u, v) -> (X-index, Y-index)
                    # Here j is from X, i is from Y
                    edges.add((j, i)) 
            else:
                L_y.append(i)

        # --- Lines 27-32: Insert 1-stars (edges) on remaining degree 1 vertices ---
        # While both pools have elements
        while I_x_o and I_y_o:
            i = I_x_o.pop()
            j = I_y_o.pop()
            edges.add((i, j))

        # --- Line 33: Update Leftovers ---
        # Add remaining unmatched degree-1 nodes to leftovers
        L_x.extend(I_x_o)
        L_y.extend(I_y_o)

        # --- Lines 34-42: Bipartite Chung-Lu on leftovers ---
        if L_x:
            # Line 36: Total edges to insert (m)
            # Sum of desired degrees for nodes in L_x
            m = sum(t_xy[i] for i in L_x)
            
            if m > 0 and L_y:
                # Prepare weights for weighted sampling
                weights_x = [t_xy[i] for i in L_x]
                weights_y = [t_yx[j] for j in L_y]
                
                # Check for zero sum weights (safety against empty or all-zero degree pools)
                if sum(weights_y) > 0 and sum(weights_x) > 0:
                    # Line 37: Loop m times
                    # Efficient sampling: random.choices returns list of size k (m)
                    # We select 'm' pairs roughly proportional to their degrees
                    chosen_xs = random.choices(L_x, weights=weights_x, k=m)
                    chosen_ys = random.choices(L_y, weights=weights_y, k=m)
                    
                    for i, j in zip(chosen_xs, chosen_ys):
                        # Line 40: Discard duplicate edges
                        if (i, j) not in edges:
                            edges.add((i, j))

        return list(edges)