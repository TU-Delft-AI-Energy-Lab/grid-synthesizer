import sys
import os
import networkx as nx
from collections import Counter

# Add 'src' to path to ensure we can import the package
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from powergrid_synth.transmission.generator import PowerGridGenerator
from powergrid_synth.transmission.input_configurator import InputConfigurator
from powergrid_synth.transmission.bus_type_allocator import BusTypeAllocator
from powergrid_synth.core.visualization import GridVisualizer

def main():
    print("===========================================================")
    print("   DEMO: 3-Level Grid Generation & Bus Type Assignment")
    print("===========================================================")
    
    # --- 1. Configuration ---
    print("\n[1] Configuring 3-Level Hierarchy...")
    configurator = InputConfigurator(seed=100)
    
    # Define 3 voltage levels 
    level_specs = [
        # Level 0: Transmission (High Connectivity)
        {'n': 20, 'avg_k': 4.0, 'diam': 6, 'dist_type': 'dgln'},
        # Level 1: Sub-Transmission
        {'n': 20, 'avg_k': 3.0, 'diam': 10, 'dist_type': 'dpl'},
        # Level 2: Distribution (More Radial)
        {'n': 10, 'avg_k': 2.0, 'diam': 10, 'dist_type': 'dgln'}
    ]
    
    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15}
    }
    
    params = configurator.create_params(level_specs, connection_specs)
    
    # --- 2. Generation ---
    print("\n[2] Generating Topology...")
    gen = PowerGridGenerator(seed=100)
    grid = gen.generate_grid(
        params['degrees_by_level'], 
        params['diameters_by_level'], 
        params['transformer_degrees'],
        keep_lcc=True
    )
    print(f"    -> Generated Connected Grid: {grid.number_of_nodes()} nodes, {grid.number_of_edges()} edges")
    
    # --- 3. Topology Visualization ---
    viz = GridVisualizer()
    print("\n[3] Visualizing Raw Topology...")
    print("    (Close the plot window to proceed to Bus Type Allocation)")
    viz.plot_interactive(
        grid, 
        title="Step 1: Raw Topology (3 Voltage Levels)",
        figsize=(12, 10)
    )
    
    # --- 4. Bus Type Allocation ---
    print("\n[4] Allocating Bus Types (AIS Optimization)...")
    allocator = BusTypeAllocator(grid, entropy_model=1)
    # We use a moderate iteration count for the demo
    bus_types = allocator.allocate(max_iter=100, population_size=10)
    
    # Assign attributes to the graph nodes
    nx.set_node_attributes(grid, bus_types, name="bus_type")
    
    # Show stats
    counts = Counter(bus_types.values())
    total = sum(counts.values())
    print(f"    -> Assignment Complete:")
    print(f"       Generators: {counts['Gen']} ({counts['Gen']/total:.1%})")
    print(f"       Loads:      {counts['Load']} ({counts['Load']/total:.1%})")
    print(f"       Connectors: {counts['Conn']} ({counts['Conn']/total:.1%})")
    
    # --- 5. Bus Type Visualization ---
    print("\n[5] Visualizing Bus Types & Edge Styles (Interactive)...")
    print("    - Nodes: Red=Gen, Green=Load, Blue=Conn")
    print("    - Edges: Dashed=GG, Solid=LL, Dotted=CC, etc.")
    
    # Call the new interactive method
    viz.plot_interactive_bus_types(
        grid, 
        title="Step 2: Bus Types & Transmission Lines (Interactive)"
    )
    
    print("\nDemo Complete.")

if __name__ == "__main__":
    main()