import pytest
import numpy as np
import networkx as nx
from powergrid_synth.transmission.transmission import TransmissionLineAllocator
from powergrid_synth.transmission.input_configurator import InputConfigurator
from powergrid_synth.transmission.generator import PowerGridGenerator
from powergrid_synth.transmission.bus_type_allocator import BusTypeAllocator
from powergrid_synth.transmission.capacity_allocator import CapacityAllocator
from powergrid_synth.transmission.load_allocator import LoadAllocator
from powergrid_synth.transmission.generation_dispatcher import GenerationDispatcher


class TestTransmissionLineAllocator:

    @pytest.fixture()
    def dispatched_grid(self):
        """Small grid with bus types, capacities, loads, and dispatch set."""
        np.random.seed(42)
        configurator = InputConfigurator(seed=42)
        level_specs = [
            {"n": 25, "avg_k": 2.5, "diam": 5, "dist_type": "poisson"},
        ]
        params = configurator.create_params(level_specs, {})

        gen = PowerGridGenerator(seed=42)
        grid = gen.generate_grid(
            params["degrees_by_level"],
            params["diameters_by_level"],
            params["transformer_degrees"],
            keep_lcc=True,
        )

        bus_types = BusTypeAllocator(grid).allocate(max_iter=5)
        nx.set_node_attributes(grid, bus_types, name="bus_type")

        caps = CapacityAllocator(grid, ref_sys_id=1).allocate()
        nx.set_node_attributes(grid, caps, name="pg_max")

        loads = LoadAllocator(grid, ref_sys_id=1).allocate(loading_level="M")
        nx.set_node_attributes(grid, loads, name="pl")

        dispatch = GenerationDispatcher(grid, ref_sys_id=1).dispatch()
        nx.set_node_attributes(grid, dispatch, name="pg")

        return grid

    def test_allocate_returns_dict(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        result = allocator.allocate()
        assert isinstance(result, dict)

    def test_all_edges_get_capacity(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        result = allocator.allocate()
        assert len(result) == dispatched_grid.number_of_edges()

    def test_edges_have_impedance_attrs(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        allocator.allocate()
        for u, v, data in dispatched_grid.edges(data=True):
            assert "x" in data, f"Edge ({u},{v}) missing 'x' attribute"
            assert "r" in data, f"Edge ({u},{v}) missing 'r' attribute"
            assert data["x"] > 0

    def test_generate_phi_range(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        phi = allocator._generate_phi(100)
        assert phi.shape == (100,)
        assert np.all(phi >= 0.01)
        assert np.all(phi <= 89.99)

    def test_generate_beta(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        beta = allocator._generate_beta(50)
        assert beta.shape == (50,)
        assert np.all(beta > 0)

    def test_capacities_are_positive(self, dispatched_grid):
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        result = allocator.allocate()
        for cap in result.values():
            assert cap > 0

    def test_assign_betas_empty(self, dispatched_grid):
        """_assign_betas returns empty array for empty inputs."""
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        result = allocator._assign_betas(np.array([]), np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_allocate_empty_graph(self):
        """allocate() on a graph with no edges returns empty dict."""
        g = nx.Graph()
        g.add_node(0, bus_type="PQ", pg_max=0, pl=10, pg=0)
        allocator = TransmissionLineAllocator(g, ref_sys_id=1)
        result = allocator.allocate()
        assert result == {}

    def test_allocate_with_refine_topology(self, dispatched_grid):
        """allocate(refine_topology=True) should still return valid capacities."""
        np.random.seed(42)
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        result = allocator.allocate(refine_topology=True)
        assert isinstance(result, dict)
        assert len(result) > 0
        for cap in result.values():
            assert cap > 0

    def test_generate_beta_sorted(self, dispatched_grid):
        """_generate_beta returns values in sorted order."""
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        beta = allocator._generate_beta(200)
        assert np.all(beta[:-1] <= beta[1:])

    def test_generate_beta_no_zeros(self, dispatched_grid):
        """_generate_beta should have no near-zero values (< 1e-4)."""
        np.random.seed(0)
        allocator = TransmissionLineAllocator(dispatched_grid, ref_sys_id=1)
        beta = allocator._generate_beta(500)
        assert np.all(beta >= 1e-4)

    def test_fallback_ref_sys_id(self):
        """Invalid ref_sys_id should fall back to system 1."""
        g = nx.Graph()
        g.add_nodes_from([0, 1], bus_type="PQ", pg_max=0, pl=10, pg=0)
        g.add_edge(0, 1)
        allocator = TransmissionLineAllocator(g, ref_sys_id=99)
        # Should not raise; falls back to ref_sys_id=1
        assert allocator.stats is not None
        assert 'Tab_2D_FlBeta' in allocator.stats
