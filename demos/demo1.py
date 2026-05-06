import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the 'src' directory to the path so we can import the package locally
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from powergrid_synth.transmission.generator import PowerGridGenerator
from powergrid_synth.transmission.input_configurator import InputConfigurator
from powergrid_synth.core.hierarchical_analysis import HierarchicalAnalyzer

def main():
    print("--- 1. Configuration: Setting up 3-Level Hierarchy ---")
    
    # Initialize Configurator
    configurator = InputConfigurator(seed=100)

    # Define 3 voltage levels mimicking a transmission -> sub-transmission -> distribution hierarchy
    level_specs = [
        # Level 0: Transmission (High Connectivity)
        {'n': 20, 'avg_k': 4.0, 'diam': 6, 'dist_type': 'dgln'},
        # Level 1: Sub-Transmission
        {'n': 20, 'avg_k': 3.0, 'diam': 10, 'dist_type': 'dpl'},
        # Level 2: Distribution (More Radial)
        {'n': 10, 'avg_k': 2.0, 'diam': 10, 'dist_type': 'poisson'}
    ]
    
    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15}
    }

    print("--- 2. Generating Input Parameters ---")
    params = configurator.create_params(level_specs, connection_specs)

    print("--- 3. Generating Grid Topology ---")
    gen = PowerGridGenerator(seed=100)
    grid_graph = gen.generate_grid(
        degrees_by_level=params['degrees_by_level'],
        diameters_by_level=params['diameters_by_level'],
        transformer_degrees=params['transformer_degrees'], 
        keep_lcc=True
    )
    
    print(f"Grid Generated: {grid_graph.number_of_nodes()} nodes, {grid_graph.number_of_edges()} edges")

    print("\n--- 4. Running Hierarchical Analysis ---")
    print("This will print topological metrics and plot degree distributions.")
    
    # Initialize the Analyzer
    analyzer = HierarchicalAnalyzer(grid_graph)
    
    # Run the full report
    # - Calculates global metrics (Diameter, Clustering, etc.)
    # - Calculates metrics for EACH voltage level subgraph
    # - Plots degree distributions (Log-Log scale)
    analyzer.run_full_report(log_scale=True)
    
    print("Analysis Complete.")

if __name__ == "__main__":
    main()