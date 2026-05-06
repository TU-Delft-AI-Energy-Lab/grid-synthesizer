r"""
This module extracts the topology parameters from realistic grids topology, acting as inputs for generate synthetic grids topology, namely

- node degrees per voltage level 
- node diameter per voltage level
- transformer degrees between different-voltage levels
"""

import networkx as nx
from typing import Dict, List, Tuple

def extract_topology_params_from_graph(G: nx.Graph) -> Dict:
    r"""
    Extract CLC model inputs from an existing power grid graph.

    This is "operation mode I", where the generator is configured to mimic
    an existing reference grid.  The function extracts per-level degree
    sequences and diameters (Phase 1 inputs), as well as pairwise
    transformer degree sequences (Phase 2 inputs).

    Parameters
    ----------
    G : networkx.Graph
        Power grid graph with a ``'voltage_level'`` attribute on every node.

    Returns
    -------
    dict
        ``'degrees_by_level'`` : list of list of int
            Intra-level degree sequences (one per voltage level, ordered by
            ascending voltage label).
        ``'diameters_by_level'`` : list of int
            Diameter of the largest connected component of each same-voltage
            subgraph.
        ``'transformer_degrees'`` : dict
            ``{(i, j): (deg_i_to_j, deg_j_to_i)}`` for each pair of voltage
            levels that has at least one transformer edge.
    """
    print("Extracting topology parameters...")
    
    # Get sorted unique voltage levels present in the graph
    # We use a set comprehension to handle missing attributes gracefully if needed
    levels = sorted(list(set(d.get('voltage_level') for n, d in G.nodes(data=True) if 'voltage_level' in d)))
    
    degrees_by_level = []
    diameters_by_level = []
    transformer_degrees = {}
    
    # 1. Per Level Stats
    for lvl in levels:
        nodes = [n for n, d in G.nodes(data=True) if d.get('voltage_level') == lvl]
        subG = G.subgraph(nodes)
        
        # Degree sequence (within the level)
        degs = [d for n, d in subG.degree()]
        degrees_by_level.append(degs)
        
        # Diameter (LCC)
        if len(nodes) > 0:
            if nx.is_connected(subG):
                diam = nx.diameter(subG)
            else:
                largest_cc = max(nx.connected_components(subG), key=len)
                diam = nx.diameter(subG.subgraph(largest_cc))
        else:
            diam = 0
        diameters_by_level.append(diam)
        
    # 2. Transformer Stats (Inter-level connections)
    # We iterate by index to map to the 'degrees_by_level' indices (0, 1, 2...)
    # This assumes the generator expects level indices 0..K-1
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            lvl_i = levels[i]
            lvl_j = levels[j]
            
            nodes_i = [n for n, d in G.nodes(data=True) if d.get('voltage_level') == lvl_i]
            nodes_j = [n for n, d in G.nodes(data=True) if d.get('voltage_level') == lvl_j]
            
            if not nodes_i or not nodes_j: 
                continue
            
            # Degrees in Level i towards Level j
            deg_i_to_j = []
            for n in nodes_i:
                count = sum(1 for neighbor in G.neighbors(n) if neighbor in nodes_j)
                deg_i_to_j.append(count)
                
            # Degrees in Level j towards Level i
            deg_j_to_i = []
            for n in nodes_j:
                count = sum(1 for neighbor in G.neighbors(n) if neighbor in nodes_i)
                deg_j_to_i.append(count)
                
            # Only record if there are actual connections
            if sum(deg_i_to_j) > 0:
                transformer_degrees[(i, j)] = (deg_i_to_j, deg_j_to_i)

    return {
        'degrees_by_level': degrees_by_level,
        'diameters_by_level': diameters_by_level,
        'transformer_degrees': transformer_degrees
    }