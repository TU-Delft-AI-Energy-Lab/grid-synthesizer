import sys
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from powergrid_synth.transmission.generator import PowerGridGenerator
from powergrid_synth.transmission.input_configurator import InputConfigurator
from powergrid_synth.transmission.bus_type_allocator import BusTypeAllocator
from powergrid_synth.transmission.capacity_allocator import CapacityAllocator

def main():
    # --- 1. Configuration ---
    print("\n[1] Configuring 3-Level Hierarchy...")
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
    
    params = configurator.create_params(level_specs, connection_specs)
    gen = PowerGridGenerator(seed=100)
    grid = gen.generate_grid(
        params['degrees_by_level'], 
        params['diameters_by_level'], 
        params['transformer_degrees'],
        keep_lcc=True
    )
    
    print(f"Grid: {grid.number_of_nodes()} nodes.")
    
    print("\n--- 2. Assigning Bus Types ---")
    allocator = BusTypeAllocator(grid, entropy_model=1)
    bus_types = allocator.allocate(max_iter=100, population_size=20)
    nx.set_node_attributes(grid, bus_types, name="bus_type")
    
    gen_count = sum(1 for t in bus_types.values() if t == 'Gen')
    print(f"Generators identified: {gen_count}")
    
    print("\n--- 3. Allocating Capacity ---")
    cap_allocator = CapacityAllocator(grid)
    capacities = cap_allocator.allocate()
    
    # Attach to graph
    nx.set_node_attributes(grid, capacities, name="pg_max")
    
    print("\n--- 4. Results Analysis ---")
    total_assigned = sum(capacities.values())
    print(f"Total Capacity Assigned: {total_assigned:.2f} MW")
    
    # Check top 5 generators
    sorted_gens = sorted(capacities.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Generators by Capacity:")
    for node, cap in sorted_gens[:5]:
        print(f"  Node {node}: {cap:.2f} MW (Degree: {grid.degree(node)})")
        
    # Plot Distribution
    caps = list(capacities.values())
    if caps:
        plt.figure(figsize=(10, 6))
        plt.hist(caps, bins=30, color='skyblue', edgecolor='black')
        plt.title("Generator Capacity Distribution")
        plt.xlabel("Capacity (MW)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    main()