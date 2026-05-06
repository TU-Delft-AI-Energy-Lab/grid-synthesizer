import pytest
import networkx as nx
from powergrid_synth.core.grid_graph import (
    PowerGridGraph,
    TransmissionGrid,
    DistributionGrid,
)


class TestPowerGridGraph:

    @pytest.fixture()
    def multi_level_graph(self):
        g = PowerGridGraph()
        for i in range(5):
            g.add_node(i, voltage_level=0)
        for i in range(5, 15):
            g.add_node(i, voltage_level=1)
        for i in range(15, 25):
            g.add_node(i, voltage_level=2)
        # intra-level edges
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        g.add_edges_from([(5, 6), (6, 7), (7, 8)])
        # inter-level (transformer) edges
        g.add_edges_from([(4, 5), (8, 15)])
        return g

    def test_level_returns_correct_nodes(self, multi_level_graph):
        sub0 = multi_level_graph.level(0)
        assert set(sub0.nodes()) == {0, 1, 2, 3, 4}

    def test_level_returns_correct_nodes_level1(self, multi_level_graph):
        sub1 = multi_level_graph.level(1)
        assert set(sub1.nodes()) == set(range(5, 15))

    def test_level_contains_only_intra_edges(self, multi_level_graph):
        sub0 = multi_level_graph.level(0)
        for u, v in sub0.edges():
            assert sub0.nodes[u]['voltage_level'] == 0
            assert sub0.nodes[v]['voltage_level'] == 0

    def test_level_nonexistent_returns_empty(self, multi_level_graph):
        sub99 = multi_level_graph.level(99)
        assert sub99.number_of_nodes() == 0

    def test_is_subclass_of_nx_graph(self):
        g = PowerGridGraph()
        assert isinstance(g, nx.Graph)


# ------------------------------------------------------------------
# TransmissionGrid tests
# ------------------------------------------------------------------

class TestTransmissionGrid:

    @pytest.fixture()
    def trans_grid(self):
        g = TransmissionGrid()
        for i in range(5):
            g.add_node(i, voltage_level=0)
        for i in range(5, 10):
            g.add_node(i, voltage_level=1)
        for i in range(10, 13):
            g.add_node(i, voltage_level=2)
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        g.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 9)])
        g.add_edges_from([(10, 11), (11, 12)])
        g.add_edges_from([(4, 5), (9, 10)])
        return g

    def test_is_subclass(self):
        g = TransmissionGrid()
        assert isinstance(g, PowerGridGraph)
        assert isinstance(g, nx.Graph)

    def test_voltage_levels(self, trans_grid):
        assert trans_grid.voltage_levels == [0, 1, 2]

    def test_n_levels(self, trans_grid):
        assert trans_grid.n_levels == 3

    def test_level_inherited(self, trans_grid):
        sub = trans_grid.level(1)
        assert set(sub.nodes()) == set(range(5, 10))

    def test_empty_grid(self):
        g = TransmissionGrid()
        assert g.voltage_levels == []
        assert g.n_levels == 0


# ------------------------------------------------------------------
# DistributionGrid tests
# ------------------------------------------------------------------

class TestDistributionGrid:

    @pytest.fixture()
    def dist_grid(self):
        """A small hand-built radial distribution feeder."""
        g = DistributionGrid()
        #     0 (root)
        #    / \
        #   1   2 (load)
        #  / \
        # 3   4 (load)
        g.add_node(0, h=0, node_type="intermediate", P_mw=0.0, pf=0.95)
        g.add_node(1, h=1, node_type="intermediate", P_mw=0.0, pf=0.95)
        g.add_node(2, h=1, node_type="load", P_mw=0.5, pf=0.9)
        g.add_node(3, h=2, node_type="load", P_mw=0.3, pf=0.92)
        g.add_node(4, h=2, node_type="injection", P_mw=-0.2, pf=0.95)
        g.add_edge(0, 1, length_km=0.1)
        g.add_edge(0, 2, length_km=0.2)
        g.add_edge(1, 3, length_km=0.15)
        g.add_edge(1, 4, length_km=0.05)
        return g

    def test_is_subclass(self):
        g = DistributionGrid()
        assert isinstance(g, PowerGridGraph)
        assert isinstance(g, nx.Graph)

    def test_root(self, dist_grid):
        assert dist_grid.root == 0

    def test_max_hop(self, dist_grid):
        assert dist_grid.max_hop == 2

    def test_is_radial(self, dist_grid):
        assert dist_grid.is_radial is True

    def test_not_radial_with_cycle(self, dist_grid):
        dist_grid.add_edge(3, 4)  # creates cycle
        assert dist_grid.is_radial is False

    def test_nodes_at_hop(self, dist_grid):
        assert set(dist_grid.nodes_at_hop(0)) == {0}
        assert set(dist_grid.nodes_at_hop(1)) == {1, 2}
        assert set(dist_grid.nodes_at_hop(2)) == {3, 4}
        assert dist_grid.nodes_at_hop(5) == []

    def test_nodes_by_type(self, dist_grid):
        assert set(dist_grid.nodes_by_type("load")) == {2, 3}
        assert set(dist_grid.nodes_by_type("injection")) == {4}
        assert set(dist_grid.nodes_by_type("intermediate")) == {0, 1}
        assert dist_grid.nodes_by_type("nonexistent") == []

    def test_total_load_mw(self, dist_grid):
        assert dist_grid.total_load_mw == pytest.approx(0.8)

    def test_total_gen_mw(self, dist_grid):
        assert dist_grid.total_gen_mw == pytest.approx(0.2)

    def test_from_nx(self, dist_grid):
        # Build a plain nx.Graph with the same data
        plain = nx.Graph()
        plain.add_nodes_from(dist_grid.nodes(data=True))
        plain.add_edges_from(dist_grid.edges(data=True))
        plain.graph["v_nom_kv"] = 0.4

        dg = DistributionGrid.from_nx(plain)
        assert isinstance(dg, DistributionGrid)
        assert dg.root == 0
        assert dg.max_hop == 2
        assert dg.total_load_mw == pytest.approx(0.8)
        assert dg.graph["v_nom_kv"] == 0.4

    def test_from_nx_preserves_edges(self, dist_grid):
        plain = nx.Graph()
        plain.add_nodes_from(dist_grid.nodes(data=True))
        plain.add_edges_from(dist_grid.edges(data=True))

        dg = DistributionGrid.from_nx(plain)
        assert dg.number_of_edges() == dist_grid.number_of_edges()
        for u, v, d in dg.edges(data=True):
            assert "length_km" in d

    def test_from_generated_feeder(self):
        from powergrid_synth.distribution import SchweetzerFeederGenerator
        gen = SchweetzerFeederGenerator(seed=42)
        G = gen.generate_feeder(n_nodes=20, total_load_mw=3.0)
        dg = DistributionGrid.from_nx(G)
        assert isinstance(dg, DistributionGrid)
        assert dg.root == 0
        assert dg.is_radial
        assert dg.total_load_mw == pytest.approx(3.0, rel=0.1)

    def test_empty_grid(self):
        g = DistributionGrid()
        assert g.root is None
        assert g.max_hop == 0
        assert g.total_load_mw == 0.0
        assert g.total_gen_mw == 0.0
