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
from powergrid_synth.transmission.load_allocator import LoadAllocator

def main():
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
    nx.set_node_attributes(grid, capacities, name="pg_max")
    
    total_gen = sum(capacities.values())
    print(f"Total Generation: {total_gen:.2f} MW")
    
    print("\n--- 4. Allocating Loads (Heavy Loading) ---")
    load_allocator = LoadAllocator(grid, ref_sys_id=1)
    loads = load_allocator.allocate(loading_level='H')
    
    # Attach to graph (attribute 'pl' for active power load)
    nx.set_node_attributes(grid, loads, name="pl")
    
    total_load = sum(loads.values())
    print(f"Total Load: {total_load:.2f} MW")
    
    print(f"System Loading: {total_load/total_gen:.1%}")

    print("\n--- 5. Results Analysis ---")
    # Plot Distribution
    load_vals = list(loads.values())
    
    if load_vals:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(load_vals, bins=30, color='orange', edgecolor='black')
        plt.title("Load Size Distribution")
        plt.xlabel("Load (MW)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(list(capacities.values()), [0]*len(capacities), alpha=0.5, label='Gen', color='blue')
        plt.scatter(list(loads.values()), [1]*len(loads), alpha=0.5, label='Load', color='orange')
        plt.yticks([0, 1], ['Generators', 'Loads'])
        plt.title("Capacity vs Load Magnitude")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()