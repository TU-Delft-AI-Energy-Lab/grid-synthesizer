import pytest
import networkx as nx
from powergrid_synth.core.input_extractor import extract_topology_params_from_graph


class TestInputExtractor:

    @pytest.fixture()
    def two_level_graph(self):
        """A simple 2-level graph with known structure."""
        G = nx.Graph()
        # Level 0: 5 nodes, chain
        for i in range(5):
            G.add_node(i, voltage_level=0)
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        # Level 1: 8 nodes, chain
        for i in range(5, 13):
            G.add_node(i, voltage_level=1)
        G.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)])
        # Transformer edges
        G.add_edges_from([(2, 5), (4, 8)])
        return G

    def test_returns_required_keys(self, two_level_graph):
        result = extract_topology_params_from_graph(two_level_graph)
        assert "degrees_by_level" in result
        assert "diameters_by_level" in result
        assert "transformer_degrees" in result

    def test_degrees_by_level_count(self, two_level_graph):
        result = extract_topology_params_from_graph(two_level_graph)
        assert len(result["degrees_by_level"]) == 2  # 2 voltage levels

    def test_diameters_by_level_count(self, two_level_graph):
        result = extract_topology_params_from_graph(two_level_graph)
        assert len(result["diameters_by_level"]) == 2

    def test_level0_diameter(self, two_level_graph):
        """Level 0 is a chain of 5 nodes → diameter = 4."""
        result = extract_topology_params_from_graph(two_level_graph)
        assert result["diameters_by_level"][0] == 4

    def test_transformer_degrees_present(self, two_level_graph):
        result = extract_topology_params_from_graph(two_level_graph)
        td = result["transformer_degrees"]
        assert len(td) > 0
        # (0, 1) should be a key
        assert (0, 1) in td

    def test_single_level_graph(self):
        """Graph with only one voltage level should have no transformer degrees."""
        G = nx.cycle_graph(10)
        nx.set_node_attributes(G, 0, "voltage_level")
        result = extract_topology_params_from_graph(G)
        assert len(result["degrees_by_level"]) == 1
        assert len(result["transformer_degrees"]) == 0
