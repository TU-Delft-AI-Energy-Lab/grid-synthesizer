import numpy as np
import sys
import os

# Add the 'src' directory to the path so we can import the package locally
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from powergrid_synth.transmission.generator import PowerGridGenerator
from powergrid_synth.core.visualization import GridVisualizer
from powergrid_synth.transmission.input_configurator import InputConfigurator

def main():
    print("--- 1. Configuration: Setting up 3-Level Hierarchy ---")
    
    # Initialize Configurator
    configurator = InputConfigurator(seed=42)

    # Define Specs for 3 Voltage Levels
    level_specs = [
        # Level 0: High Voltage (Backbone)
        {'n': 50, 'avg_k': 3.5, 'diam': 10, 'dist_type': 'dgln', 'max_k': 15},
        
        # Level 1: Medium Voltage (Distribution)
        {'n': 100, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'dpl', 'max_k': 10},
        
        # Level 2: Low Voltage 
        {'n': 150, 'avg_k': 2.0, 'diam': 20, 'dist_type': 'dgln', 'max_k': 10}
    ]

    # Define k-stars connections between levels
    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15}
    }

    print("--- 2. Generating Input Parameters ---")
    params = configurator.create_params(level_specs, connection_specs)

    print("--- 3. Generating Grid Topology ---")
    gen = PowerGridGenerator(seed=42)
    grid_graph = gen.generate_grid(
        degrees_by_level=params['degrees_by_level'],
        diameters_by_level=params['diameters_by_level'],
        transformer_degrees=params['transformer_degrees']
    )
    
    print(f"Grid Generated: {grid_graph.number_of_nodes()} nodes, {grid_graph.number_of_edges()} edges")

    # Initialize Visualizer
    viz = GridVisualizer()

    # --- Visualization 1: Full Interactive Grid ---
    print("\n--- Visualizing Full Grid (Interactive Mode) ---")
    print("Please check the popup window. Use the 'Select Layout' dropdown at the top.")
    viz.plot_interactive(
        grid_graph, 
        title="3-Level Synthetic Grid (Full View)",
        figsize=(14, 10)
    )

    # --- Visualization 2: Per-Level Interactive Plots ---
    print("\n--- Visualizing Individual Voltage Levels (Interactive Mode) ---")
    
    # Level 0
    print("Opening Interactive Plot for Level 0 (High Voltage)...")
    viz.plot_interactive_voltage_level(
        grid_graph, 
        level=0,
        title="Level 0: High Voltage Backbone (Interactive)"
    )

    # Level 1
    print("Opening Interactive Plot for Level 1 (Medium Voltage)...")
    viz.plot_interactive_voltage_level(
        grid_graph, 
        level=1,
        title="Level 1: Medium Voltage Distribution (Interactive)"
    )

    # Level 2
    print("Opening Interactive Plot for Level 2 (Low Voltage)...")
    viz.plot_interactive_voltage_level(
        grid_graph, 
        level=2,
        title="Level 2: Low Voltage Local (Interactive)"
    )

    print("\nTest Complete.")

if __name__ == "__main__":
    main()