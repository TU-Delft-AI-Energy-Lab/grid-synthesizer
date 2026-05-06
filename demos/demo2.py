import numpy as np
import networkx as nx
import sys
import os

# Add the 'src' directory to the path so we can import the package locally
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from powergrid_synth.transmission.generator import PowerGridGenerator
from powergrid_synth.core.visualization import GridVisualizer
from powergrid_synth.transmission.input_configurator import InputConfigurator
from powergrid_synth.core.comparison import GraphComparator

def generate_hierarchical_ws_baseline(target_graph, seed=None):
    """
    Generates a hierarchical baseline where each internal voltage level is 
    modeled as a Watts-Strogatz Small-World graph (instead of pure random).
    
    This is a stronger baseline for power grids, which naturally cluster.
    """
    print("   -> Constructing Hierarchical Watts-Strogatz Baseline...")
    rng = np.random.RandomState(seed)
    H = nx.Graph()
    
    levels = sorted(list(set(nx.get_node_attributes(target_graph, 'voltage_level').values())))
    level_nodes_map = {} 
    
    # 1. Recreate Nodes and Internal Edges (Watts-Strogatz)
    for level in levels:
        nodes = [n for n, d in target_graph.nodes(data=True) if d.get('voltage_level') == level]
        level_nodes_map[level] = set(nodes)
        
        # Add nodes with metadata
        for n in nodes:
            H.add_node(n, voltage_level=level)
            
        # Analyze target level density
        subgraph = target_graph.subgraph(nodes)
        n_edges = subgraph.number_of_edges()
        n_nodes = len(nodes)
        
        if n_nodes > 1 and n_edges > 0:
            # Watts-Strogatz requires parameter 'k' (nearest neighbors)
            # We estimate k from the average degree: k ~ 2 * E / N
            avg_k = 2 * n_edges / n_nodes
            k_ws = int(round(avg_k))
            
            # Constraint check for WS generation:
            # 1. k must be at least 2 (ring structure)
            # 2. k must be less than n_nodes
            if k_ws >= 2 and k_ws < n_nodes:
                # Use standard rewiring probability p=0.1 for small-world property
                try:
                    # Note: WS graph will have exactly N*k/2 edges. This might slightly differ
                    # from target n_edges due to rounding of k, but preserves the "type" of topology.
                    ws_sub = nx.watts_strogatz_graph(n_nodes, k_ws, p=0.1, seed=rng.randint(0, 100000))
                    
                    # Relabel to match original IDs
                    mapping = {i: nodes[i] for i in range(n_nodes)}
                    ws_sub = nx.relabel_nodes(ws_sub, mapping)
                    H.add_edges_from(ws_sub.edges())
                except Exception as e:
                    print(f"      [Warning] Level {level} WS generation failed ({e}), falling back to Random.")
                    rand_sub = nx.gnm_random_graph(n_nodes, n_edges, seed=rng.randint(0, 100000))
                    mapping = {i: nodes[i] for i in range(n_nodes)}
                    rand_sub = nx.relabel_nodes(rand_sub, mapping)
                    H.add_edges_from(rand_sub.edges())
            else:
                # If topology is too sparse (avg degree < 2) or too small for WS ring
                # Fallback to standard G(n,m) random
                rand_sub = nx.gnm_random_graph(n_nodes, n_edges, seed=rng.randint(0, 100000))
                mapping = {i: nodes[i] for i in range(n_nodes)}
                rand_sub = nx.relabel_nodes(rand_sub, mapping)
                H.add_edges_from(rand_sub.edges())

    # 2. Recreate Transformer (Cross-Level) Edges
    # We keep these as random bipartite connections (same as before) because 
    # Watts-Strogatz doesn't have a direct equivalent for bipartite inter-ties.
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            lvl_i = levels[i]
            lvl_j = levels[j]
            
            nodes_i_list = list(level_nodes_map[lvl_i])
            nodes_j_list = list(level_nodes_map[lvl_j])
            nodes_j_set = level_nodes_map[lvl_j]
            
            cross_edges_count = 0
            for u in nodes_i_list:
                for v in target_graph[u]:
                    if v in nodes_j_set:
                        cross_edges_count += 1
            
            if cross_edges_count > 0:
                B = nx.bipartite.gnmk_random_graph(len(nodes_i_list), len(nodes_j_list), cross_edges_count, seed=rng.randint(0, 100000))
                mapping = {}
                for idx in range(len(nodes_i_list)): mapping[idx] = nodes_i_list[idx]
                for idx in range(len(nodes_j_list)): mapping[len(nodes_i_list) + idx] = nodes_j_list[idx]
                
                B = nx.relabel_nodes(B, mapping)
                H.add_edges_from(B.edges())
                
    return H

def main():
    print("--- 1. Configuration: Setting up 5-Level Hierarchy ---")
    
    # Initialize Configurator
    configurator = InputConfigurator(seed=2024)

    # Define Specs for 5 Voltage Levels
    # Same standard config as previous tests
    level_specs = [
        {'n': 50, 'avg_k': 2.0, 'diam': 8, 'dist_type': 'dgln', 'max_k': 20},
        {'n': 100, 'avg_k': 3.0, 'diam': 12, 'dist_type': 'dpl', 'max_k': 20},
        {'n': 200, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'dgln', 'max_k': 20},
        {'n': 400, 'avg_k': 2.2, 'diam': 20, 'dist_type': 'dgln', 'max_k': 20},
        {'n': 800, 'avg_k': 1.8, 'diam': 25, 'dist_type': 'dgln', 'max_k': 20}
    ]

    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (2, 3): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15},
        (3, 4): {'type': 'k-stars', 'c': 0.10, 'gamma': 4.15}
    }

    print("--- 2. Generating Input Parameters ---")
    params = configurator.create_params(level_specs, connection_specs)

    print("--- 3. Generating Grid Topology ---")
    gen = PowerGridGenerator(seed=2024)
    grid_graph = gen.generate_grid(
        degrees_by_level=params['degrees_by_level'],
        diameters_by_level=params['diameters_by_level'],
        transformer_degrees=params['transformer_degrees']
    )
    
    print(f"Synthetic Grid Created: {grid_graph.number_of_nodes()} nodes, {grid_graph.number_of_edges()} edges")

    # print("--- 4. Interactive Visualization ---")
    # print("Opening interactive plot window... (Select layout from the menu)")
    # viz = GridVisualizer()
    # viz.plot_interactive(grid_graph, title="5-Level Synthetic Grid")

    print("--- 5. Comparison with Hierarchical Watts-Strogatz Baseline ---")
    # Generate the Small-World baseline
    ws_baseline = generate_hierarchical_ws_baseline(grid_graph, seed=2024)
    
    print(f"Baseline Generated: {ws_baseline.number_of_nodes()} nodes, {ws_baseline.number_of_edges()} edges")
    
    comparator = GraphComparator(
        synth_graph=grid_graph, 
        ref_graph=ws_baseline, 
        synth_label="Synthetic (CLC)",
        ref_label="Reference (Watts-Strogatz)"
    )
    
    # Run comparison (metrics + plots)
    # Pay attention to 'Avg Clustering' in the report:
    # Synthetic and WS should both be higher than pure random graphs.
    comparator.run_full_comparison(log_scale=True)
    
    print("Test Complete.")

if __name__ == "__main__":
    main()