import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch
import matplotlib.pyplot as plt

# Import your package modules
from powergrid_synth.transmission.generator import PowerGridGenerator
from powergrid_synth.transmission.input_configurator import InputConfigurator
from powergrid_synth.transmission.bus_type_allocator import BusTypeAllocator
from powergrid_synth.core.visualization import GridVisualizer


class TestBusTypeAllocator:
    """Unit tests for BusTypeAllocator to cover missing branches."""

    def _make_graph(self, n=30, m=2, seed=42):
        return nx.barabasi_albert_graph(n, m, seed=seed)

    # --- _normalize_ratio ---

    def test_normalize_ratio_valid(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        result = alloc._normalize_ratio([3, 3, 4])
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-9

    def test_normalize_ratio_wrong_length(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        with pytest.raises(ValueError, match="exactly 3"):
            alloc._normalize_ratio([1, 2])

    def test_normalize_ratio_zero_sum(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        with pytest.raises(ValueError, match="positive"):
            alloc._normalize_ratio([0, 0, 0])

    # --- _get_ratios ---

    def test_get_ratios_small_network(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        ratios = alloc._get_ratios(100)
        assert ratios == [0.23, 0.55, 0.22]

    def test_get_ratios_medium_network(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        ratios = alloc._get_ratios(5000)
        assert ratios == [0.33, 0.44, 0.23]

    def test_get_ratios_large_network(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        ratios = alloc._get_ratios(10000)
        assert ratios == [0.2, 0.4, 0.4]

    # --- custom bus_type_ratio in constructor ---

    def test_custom_bus_type_ratio(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g, bus_type_ratio=[3, 3, 4])
        assert abs(alloc.ratio_types[0] - 0.3) < 1e-9
        assert abs(alloc.ratio_types[1] - 0.3) < 1e-9
        assert abs(alloc.ratio_types[2] - 0.4) < 1e-9

    # --- _calculate_bus_ratios ---

    def test_calculate_bus_ratios_empty(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        result = alloc._calculate_bus_ratios(np.array([], dtype=int))
        assert result == [0.0, 0.0, 0.0]

    # --- _calculate_entropy_score ---

    def test_calculate_entropy_score_model1(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g, entropy_model=1)
        bus_ratios = [0.3, 0.5, 0.2]
        link_ratios = [0.1, 0.2, 0.05, 0.3, 0.15, 0.2]
        score = alloc._calculate_entropy_score(bus_ratios, link_ratios)
        assert isinstance(score, float)
        assert score > 0

    # --- _get_d_parameter ---

    def test_get_d_parameter_model0_large(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g, entropy_model=0)
        # e^8 ≈ 2981, use n > 2981 so log(n) > 8
        d = alloc._get_d_parameter(5000)
        assert isinstance(d, float)

    def test_get_d_parameter_model1(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g, entropy_model=1)
        d = alloc._get_d_parameter(100)
        assert isinstance(d, float)

    # --- _get_d_parameter model1 large n (line 225) ---

    def test_get_d_parameter_model1_large(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g, entropy_model=1)
        # e^8 ≈ 2981, use n > 2981 so log(n) > 8
        d = alloc._get_d_parameter(5000)
        assert isinstance(d, float)
        assert d < 0  # Large-n branch should give negative d

    # --- _generate_random_assignment fallback (line 252) ---

    def test_generate_random_assignment_all_leaves(self):
        """When all nodes are leaves (degree 1), connection bus assignment falls back."""
        # Path graph: nodes 0 and 3 are endpoints (degree 1), nodes 1 and 2 have degree 2
        g = nx.path_graph(4)  # nodes: 0-1-2-3
        alloc = BusTypeAllocator(g, bus_type_ratio=[0.0, 0.0, 1.0])
        # ratio says 100% connection buses but only 2 non-leaf nodes
        # n_conn = round(4 * 1.0) = 4, but only 2 non-leaf → fallback to all nodes
        assign = alloc._generate_random_assignment()
        assert len(assign) == 4
        assert set(np.unique(assign)).issubset({1, 2, 3})

    # --- plot_entropy_pdf (lines 587-620) ---

    @patch('matplotlib.pyplot.show')
    def test_plot_entropy_pdf(self, mock_show):
        import matplotlib
        matplotlib.use("Agg")
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        alloc.allocate(max_iter=3, population_size=3)
        # w_samples should be populated after allocate
        assert len(alloc.w_samples) > 0
        alloc.plot_entropy_pdf()  # should not raise
        plt.close("all")

    @patch('matplotlib.pyplot.show')
    def test_plot_entropy_pdf_no_samples(self, mock_show, capsys):
        """plot_entropy_pdf with no samples prints warning."""
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        alloc.w_samples = []
        alloc.plot_entropy_pdf()
        captured = capsys.readouterr()
        assert "No entropy samples" in captured.out

    # --- allocate ---

    def test_allocate_returns_all_nodes(self):
        g = self._make_graph()
        alloc = BusTypeAllocator(g)
        bus_types = alloc.allocate(max_iter=5, population_size=3)
        assert set(bus_types.keys()) == set(g.nodes())
        for label in bus_types.values():
            assert label in ("Gen", "Load", "Conn")


class TestIntegration:
    
    @patch('matplotlib.pyplot.show') # Prevent plot window from blocking tests
    def test_full_workflow_execution(self, mock_show):
        """
        Runs a complete end-to-end test:
        1. Configure Input
        2. Generate Topology (3 levels)
        3. Allocat Bus Types
        4. Visualize (smoke test)
        """
        # 1. Configuration
        configurator = InputConfigurator(seed=123)
        level_specs = [
            {'n': 20, 'avg_k': 3.0, 'diam': 5, 'dist_type': 'poisson'},
            {'n': 40, 'avg_k': 2.0, 'diam': 8, 'dist_type': 'poisson'}
        ]
        connection_specs = {
            (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15}
        }
        params = configurator.create_params(level_specs, connection_specs)
        
        # 2. Generation
        gen = PowerGridGenerator(seed=123)
        grid = gen.generate_grid(
            params['degrees_by_level'], 
            params['diameters_by_level'], 
            params['transformer_degrees'],
            keep_lcc=True
        )
        
        assert grid.number_of_nodes() > 0
        assert grid.number_of_edges() > 0
        
        # 3. Bus Type Allocation
        allocator = BusTypeAllocator(grid)
        # Use small max_iter for speed in testing
        bus_types = allocator.allocate(max_iter=50, population_size=5)
        
        nx.set_node_attributes(grid, bus_types, name="bus_type")
        
        # Check that attributes are actually set
        node_sample = list(grid.nodes())[0]
        assert 'bus_type' in grid.nodes[node_sample]
        assert grid.nodes[node_sample]['bus_type'] in ['Gen', 'Load', 'Conn']

        # 4. Visualization (Smoke Test)
        # We call the plotting functions to ensure they don't crash.
        # The @patch decorator prevents the window from actually opening.
        viz = GridVisualizer()
        viz.plot_grid(grid, layout='spring', title="Integration Test Grid")
        viz.plot_bus_types(grid, layout='spring', title="Integration Test Bus Types")
        
        # Also test the interactive helper methods (just ensuring no crash)
        # Note: We can't fully test interactivity in CI, but we can call the setup logic
        try:
            viz.plot_interactive(grid)
            viz.plot_interactive_bus_types(grid)
        except Exception as e:
            pytest.fail(f"Interactive plot setup failed: {e}")