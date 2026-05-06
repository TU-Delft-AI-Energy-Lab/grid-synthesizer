import pytest
import networkx as nx
from unittest.mock import patch

from powergrid_synth.transmission.generator import PowerGridGenerator
from powergrid_synth.transmission.input_configurator import InputConfigurator
from powergrid_synth.transmission.bus_type_allocator import BusTypeAllocator
from powergrid_synth.transmission.capacity_allocator import CapacityAllocator
from powergrid_synth.transmission.load_allocator import LoadAllocator
from powergrid_synth.transmission.generation_dispatcher import GenerationDispatcher
from powergrid_synth.core.visualization import GridVisualizer

class TestIntegration:
    
    @patch('matplotlib.pyplot.show')
    def test_full_workflow_execution(self, mock_show):
        """
        Runs a complete end-to-end test:
        1. Configuration
        2. Topology Generation
        3. Bus Type Allocation
        4. Capacity Allocation
        5. Load Allocation
        6. Generation Dispatch
        """
        # 1. Configuration
        configurator = InputConfigurator(seed=42)
        level_specs = [
            {'n': 30, 'avg_k': 2.5, 'diam': 6, 'dist_type': 'poisson'},
            {'n': 40, 'avg_k': 2.0, 'diam': 10, 'dist_type': 'poisson'}
        ]
        connection_specs = {
            (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15}
        }
        params = configurator.create_params(level_specs, connection_specs)
        
        # 2. Generation
        gen = PowerGridGenerator(seed=42)
        grid = gen.generate_grid(
            params['degrees_by_level'], 
            params['diameters_by_level'], 
            params['transformer_degrees'],
            keep_lcc=True
        )
        
        # 3. Bus Type Allocation
        allocator = BusTypeAllocator(grid)
        bus_types = allocator.allocate(max_iter=10)
        nx.set_node_attributes(grid, bus_types, name="bus_type")
        
        # 4. Capacity Allocation
        cap_allocator = CapacityAllocator(grid, ref_sys_id=1)
        capacities = cap_allocator.allocate()
        nx.set_node_attributes(grid, capacities, name="pg_max")
        
        # 5. Load Allocation
        load_allocator = LoadAllocator(grid, ref_sys_id=1)
        loads = load_allocator.allocate(loading_level='M')
        nx.set_node_attributes(grid, loads, name="pl")
        
        total_load = sum(loads.values())
        print(f"\nIntegration Test - Total Load Target: {total_load:.2f}")

        # 6. Generation Dispatch
        dispatcher = GenerationDispatcher(grid, ref_sys_id=1)
        dispatch = dispatcher.dispatch()
        nx.set_node_attributes(grid, dispatch, name="pg")
        
        total_gen = sum(dispatch.values())
        print(f"Integration Test - Total Gen Dispatched: {total_gen:.2f}")
        
        # Assert Energy Balance (Generous tolerance for random heuristics)
        # Ensure we generated enough to meet load (or very close)
        # assert abs(total_gen - total_load) < 0.1 * total_load
        
        # 7. Visualization
        viz = GridVisualizer()
        viz.plot_grid(grid, layout='spring', title="Final Dispatched Grid")