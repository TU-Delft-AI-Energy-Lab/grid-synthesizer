import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import os
import sys
import networkx as nx

# Add src to python path to import our library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from powergrid_synth.transmission.generator import PowerGridGenerator
    from powergrid_synth.core.exporter import GridExporter
    from powergrid_synth.transmission.bus_type_allocator import BusTypeAllocator
except ImportError:
    print("Could not import powergrid_synth. Please ensure you are running from the correct directory.")

def extract_topology_params(net):
    """
    Analyzes a Pandapower network to extract topology parameters required 
    for the PowerGridGenerator.
    
    Returns:
        dict: A dictionary containing 'degrees_by_level', 'diameters_by_level', 
              and 'transformer_degrees'.
    """
    print("Extracting topology parameters from real grid...")
    
    # 1. Identify Voltage Levels
    unique_voltages = sorted(net.bus.vn_kv.unique(), reverse=True)
    
    # Map voltage to level index (0, 1, 2...)
    vol_to_level = {v: i for i, v in enumerate(unique_voltages)}
    
    # 2. Build NetworkX Graph from Pandapower
    G = nx.Graph()
    
    # Add nodes with level info
    for idx, row in net.bus.iterrows():
        lvl = vol_to_level[row['vn_kv']]
        G.add_node(idx, voltage_level=lvl, vn_kv=row['vn_kv'])
        
    # Add Lines
    for _, row in net.line.iterrows():
        if row['in_service']:
            G.add_edge(row['from_bus'], row['to_bus'], type='line')
            
    # Add Trafos
    for _, row in net.trafo.iterrows():
        if row['in_service']:
            G.add_edge(row['hv_bus'], row['lv_bus'], type='transformer')
            
    # 3. Extract Parameters per Level
    degrees_by_level = []
    diameters_by_level = []
    
    for i in range(len(unique_voltages)):
        nodes_in_level = [n for n, d in G.nodes(data=True) if d['voltage_level'] == i]
        
        if not nodes_in_level:
            degrees_by_level.append([])
            diameters_by_level.append(0)
            continue
            
        subG = G.subgraph(nodes_in_level)
        
        # Degree sequence
        degrees = [d for n, d in subG.degree()]
        degrees_by_level.append(degrees)
        
        # Diameter
        if len(nodes_in_level) > 0:
            if nx.is_connected(subG):
                diam = nx.diameter(subG)
            else:
                largest_cc = max(nx.connected_components(subG), key=len)
                diam = nx.diameter(subG.subgraph(largest_cc))
        else:
            diam = 0
        diameters_by_level.append(diam)
        
    # 4. Extract Transformer Degrees
    transformer_degrees = {}
    
    for i in range(len(unique_voltages)):
        for j in range(i + 1, len(unique_voltages)):
            nodes_i = [n for n, d in G.nodes(data=True) if d['voltage_level'] == i]
            nodes_j = [n for n, d in G.nodes(data=True) if d['voltage_level'] == j]
            
            if not nodes_i or not nodes_j:
                continue
                
            deg_i_to_j = []
            for n in nodes_i:
                count = sum(1 for neighbor in G.neighbors(n) if neighbor in nodes_j)
                deg_i_to_j.append(count)
                
            deg_j_to_i = []
            for n in nodes_j:
                count = sum(1 for neighbor in G.neighbors(n) if neighbor in nodes_i)
                deg_j_to_i.append(count)
                
            if sum(deg_i_to_j) > 0:
                transformer_degrees[(i, j)] = (deg_i_to_j, deg_j_to_i)

    return {
        'degrees_by_level': degrees_by_level,
        'diameters_by_level': diameters_by_level,
        'transformer_degrees': transformer_degrees
    }

def analyze_pandapower_network(net, name):
    """
    Analyzes a Pandapower network for Topology and Bus Types.
    """
    print(f"\n{'='*20} {name} (Real) {'='*20}")
    
    # 1. Topology Stats
    n_bus = len(net.bus)
    n_line = len(net.line) + len(net.trafo)
    
    # 2. Bus Type Identification
    # In Pandapower:
    # - Generator: In 'gen', 'ext_grid', or 'sgen' table
    # - Load: In 'load' table (and not a generator)
    # - Connection: Neither
    
    gen_buses = set(net.gen.bus.values) | set(net.ext_grid.bus.values) | set(net.sgen.bus.values)
    load_buses = set(net.load.bus.values) - gen_buses
    all_buses = set(net.bus.index.values)
    conn_buses = all_buses - gen_buses - load_buses
    
    n_gen = len(gen_buses)
    n_load = len(load_buses)
    n_conn = len(conn_buses)
    
    print(f"Topology:")
    print(f"  - Nodes (N): {n_bus}")
    print(f"  - Edges (M): {n_line}")
    print(f"Bus Types:")
    print(f"  - Generators: {n_gen} ({n_gen/n_bus*100:.1f}%)")
    print(f"  - Loads:      {n_load} ({n_load/n_bus*100:.1f}%)")
    print(f"  - Connection: {n_conn} ({n_conn/n_bus*100:.1f}%)")
    
    return {'N': n_bus, 'G': n_gen, 'L': n_load, 'C': n_conn}

def analyze_synthetic_graph(graph, name):
    """
    Analyzes the generated NetworkX graph for Topology and Bus Types.
    """
    print(f"\n{'='*20} {name} (Synthetic) {'='*20}")
    
    # 1. Topology
    n_bus = graph.number_of_nodes()
    n_line = graph.number_of_edges()
    
    # 2. Bus Types (from attributes)
    n_gen = 0
    n_load = 0
    n_conn = 0
    
    for n, d in graph.nodes(data=True):
        b_type = d.get('bus_type')
        if b_type == 'Gen':
            n_gen += 1
        elif b_type == 'Load':
            n_load += 1
        else:
            n_conn += 1
            
    print(f"Topology:")
    print(f"  - Nodes (N): {n_bus}")
    print(f"  - Edges (M): {n_line}")
    print(f"Bus Types:")
    print(f"  - Generators: {n_gen} ({n_gen/n_bus*100:.1f}%)")
    print(f"  - Loads:      {n_load} ({n_load/n_bus*100:.1f}%)")
    print(f"  - Connection: {n_conn} ({n_conn/n_bus*100:.1f}%)")

    return {'N': n_bus, 'G': n_gen, 'L': n_load, 'C': n_conn}

def main():
    # 1. Load Real IEEE 30 Grid
    net_real = pn.case30()
    stats_real = analyze_pandapower_network(net_real, "IEEE 30 Grid")

    # 2. Extract Topology Characteristics
    params = extract_topology_params(net_real)
    
    print("\n[Topology Extraction Result]")
    for i, degs in enumerate(params['degrees_by_level']):
        diam = params['diameters_by_level'][i] if i < len(params['diameters_by_level']) else 0
        print(f"  Level {i}: {len(degs)} nodes, Avg Degree: {np.mean(degs):.2f}, Diameter: {diam}")

    # 3. Generate Synthetic Grid using Extracted Characteristics
    print("\n[Generating Synthetic Grid Topology...]")
    gen = PowerGridGenerator(seed=42)
    
    synthetic_graph = gen.generate_grid(
        degrees_by_level=params['degrees_by_level'],
        diameters_by_level=params['diameters_by_level'],
        transformer_degrees=params['transformer_degrees'],
        keep_lcc=True
    )
    
    # 4. Assign Bus Types (Synthetic)
    print("\n[Allocating Bus Types...]")
    allocator = BusTypeAllocator(synthetic_graph)
    # The allocator decides ratios based on N size in its internal logic.
    # We run it to see if it matches the IEEE 30 profile "naturally" via AIS.
    bus_types = allocator.allocate(max_iter=50)
    nx.set_node_attributes(synthetic_graph, bus_types, name="bus_type")
    
    # 5. Compare
    stats_syn = analyze_synthetic_graph(synthetic_graph, "Synthetic Clone")
    
    print("\n" + "="*40)
    print("COMPARISON SUMMARY")
    print("="*40)
    print(f"{'Metric':<15} | {'Real':<10} | {'Synthetic':<10} | {'Diff':<10}")
    print("-" * 55)
    print(f"{'Nodes':<15} | {stats_real['N']:<10} | {stats_syn['N']:<10} | {stats_syn['N'] - stats_real['N']:<10}")
    print(f"{'Generators':<15} | {stats_real['G']:<10} | {stats_syn['G']:<10} | {stats_syn['G'] - stats_real['G']:<10}")
    print(f"{'Loads':<15} | {stats_real['L']:<10} | {stats_syn['L']:<10} | {stats_syn['L'] - stats_real['L']:<10}")
    print(f"{'Connections':<15} | {stats_real['C']:<10} | {stats_syn['C']:<10} | {stats_syn['C'] - stats_real['C']:<10}")

if __name__ == "__main__":
    main()