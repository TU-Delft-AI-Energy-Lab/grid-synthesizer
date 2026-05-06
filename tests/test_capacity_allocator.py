import pytest
import numpy as np
import networkx as nx
from powergrid_synth.transmission.capacity_allocator import CapacityAllocator
from powergrid_synth.transmission.input_configurator import InputConfigurator
from powergrid_synth.transmission.generator import PowerGridGenerator
from powergrid_synth.transmission.bus_type_allocator import BusTypeAllocator


class TestCapacityAllocator:

    @pytest.fixture()
    def grid_with_bus_types(self):
        np.random.seed(42)
        configurator = InputConfigurator(seed=42)
        level_specs = [
            {"n": 30, "avg_k": 2.5, "diam": 6, "dist_type": "poisson"},
        ]
        params = configurator.create_params(level_specs, {})
        gen = PowerGridGenerator(seed=42)
        grid = gen.generate_grid(
            params["degrees_by_level"],
            params["diameters_by_level"],
            params["transformer_degrees"],
            keep_lcc=True,
        )
        bus_types = BusTypeAllocator(grid).allocate(max_iter=10)
        nx.set_node_attributes(grid, bus_types, name="bus_type")
        return grid

    def test_allocate_returns_dict(self, grid_with_bus_types):
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=1)
        result = allocator.allocate()
        assert isinstance(result, dict)

    def test_only_gen_nodes_get_capacity(self, grid_with_bus_types):
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=1)
        result = allocator.allocate()
        gen_nodes = {
            n for n, d in grid_with_bus_types.nodes(data=True)
            if d.get("bus_type") == "Gen"
        }
        assert set(result.keys()) == gen_nodes

    def test_capacities_are_positive(self, grid_with_bus_types):
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=1)
        result = allocator.allocate()
        for cap in result.values():
            assert cap > 0

    @pytest.mark.parametrize("ref_sys_id", [1, 2])
    def test_different_ref_systems(self, grid_with_bus_types, ref_sys_id):
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=ref_sys_id)
        result = allocator.allocate()
        assert len(result) > 0

    def test_heuristic_tab_2d_ref_sys_0(self, grid_with_bus_types):
        """ref_sys_id=0 triggers _generate_heuristic_tab_2d via _get_default_tab_2d."""
        np.random.seed(42)
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=0)
        result = allocator.allocate()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_allocate_no_generators(self):
        """Grid with only Load/Conn nodes returns empty dict from allocate."""
        grid = nx.path_graph(5)
        for n in grid.nodes():
            grid.nodes[n]["bus_type"] = "Load"
        allocator = CapacityAllocator(grid, ref_sys_id=1)
        result = allocator.allocate()
        assert result == {}

    def test_generate_heuristic_tab_2d_shape(self, grid_with_bus_types):
        """_generate_heuristic_tab_2d returns a 14x14 matrix that sums to ~1."""
        allocator = CapacityAllocator(grid_with_bus_types, ref_sys_id=0)
        tab = allocator._generate_heuristic_tab_2d()
        assert tab.shape == (14, 14)
        assert abs(np.sum(tab) - 1.0) < 1e-6

    def test_initial_generation_distribution_no_gen(self):
        """_initial_generation_distribution returns empty when n_gen=0."""
        grid = nx.path_graph(5)
        for n in grid.nodes():
            grid.nodes[n]["bus_type"] = "Load"
        allocator = CapacityAllocator(grid, ref_sys_id=1)
        assert allocator.n_gen == 0
        result = allocator._initial_generation_distribution(100.0)
        assert len(result[0]) == 0
