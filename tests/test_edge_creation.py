import pytest
import numpy as np
from powergrid_synth.transmission.edge_creation import EdgeCreator


class TestEdgeCreator:

    @pytest.fixture()
    def simple_setup(self):
        """Minimal setup: 6 nodes, 3 boxes (diameter=2)."""
        d_prime = np.array([2, 2, 2, 2, 2, 2])
        v = np.array([0, 0, 1, 1, 2, 2])  # box assignments
        D = {0, 2, 4}  # diameter path: one node per box
        S = {1, 3, 5}  # subdiameter path
        return d_prime, v, D, S

    def test_returns_list_of_tuples(self, simple_setup):
        creator = EdgeCreator()
        edges = creator.generate_edges(*simple_setup)
        assert isinstance(edges, list)
        for e in edges:
            assert isinstance(e, tuple)
            assert len(e) == 2

    def test_edges_are_sorted_pairs(self, simple_setup):
        creator = EdgeCreator()
        edges = creator.generate_edges(*simple_setup)
        for u, w in edges:
            assert u < w

    def test_diameter_path_edges_present(self, simple_setup):
        """Diameter path nodes (0, 2, 4) should form a chain."""
        creator = EdgeCreator()
        edges = creator.generate_edges(*simple_setup)
        edge_set = set(edges)
        # (0,2) and (2,4) should be in edges
        assert (0, 2) in edge_set or (0, 2) in edge_set
        assert (2, 4) in edge_set

    def test_no_self_loops(self, simple_setup):
        creator = EdgeCreator()
        edges = creator.generate_edges(*simple_setup)
        for u, w in edges:
            assert u != w

    def test_larger_graph(self):
        """Test with more nodes to exercise within-box Chung-Lu."""
        np.random.seed(42)
        n = 20
        d_prime = np.array([3] * n)
        v = np.array([i // 5 for i in range(n)])  # 4 boxes
        D = {0, 5, 10, 15}
        S = {1, 6, 11, 16}
        creator = EdgeCreator()
        edges = creator.generate_edges(d_prime, v, D, S)
        assert len(edges) > 0
