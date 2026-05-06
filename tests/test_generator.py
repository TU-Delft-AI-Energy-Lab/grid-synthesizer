import pytest
import networkx as nx
import numpy as np
from powergrid_synth.transmission.generator import PowerGridGenerator

class TestGenerator:
    def test_single_level_generation(self):
        """Test generating a simple 1-level grid."""
        deg_0 = [3, 3, 2, 2, 1, 1, 1, 1]
        diam_0 = 3
        
        gen = PowerGridGenerator(seed=123)
        grid = gen.generate_grid(
            degrees_by_level=[deg_0],
            diameters_by_level=[diam_0],
            transformer_degrees={}
        )
        
        assert isinstance(grid, nx.Graph)
        assert grid.number_of_nodes() >= len(deg_0)
        # Check attributes
        assert grid.nodes[0]['voltage_level'] == 0

    def test_multi_level_generation(self):
        """Test generating a 2-level grid with connections."""
        # Level 0
        deg_0 = [2]*10
        diam_0 = 4
        # Level 1
        deg_1 = [2]*20
        diam_1 = 6
        
        # Transformers
        t_0_1 = [1]*10
        t_1_0 = [1]*20
        
        gen = PowerGridGenerator(seed=42)
        grid = gen.generate_grid(
            degrees_by_level=[deg_0, deg_1],
            diameters_by_level=[diam_0, diam_1],
            transformer_degrees={
                (0, 1): (t_0_1, t_1_0)
            }
        )
        
        assert grid.number_of_nodes() >= 30
        
        # Verify we have nodes from both levels
        levels = set(nx.get_node_attributes(grid, 'voltage_level').values())
        assert 0 in levels
        assert 1 in levels
        
        # Verify connectivity (at least some edges should exist)
        assert grid.number_of_edges() > 0

    def test_graph_attributes(self):
        """Ensure produced graph works with standard NX algorithms."""
        deg_0 = [3]*10
        diam_0 = 3
        
        gen = PowerGridGenerator(seed=42)
        grid = gen.generate_grid([deg_0], [diam_0], {})
        
        # Should not crash
        density = nx.density(grid)
        assert 0 <= density <= 1