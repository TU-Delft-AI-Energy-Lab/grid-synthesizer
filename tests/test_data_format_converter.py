import tempfile

import pytest
import networkx as nx

pp = pytest.importorskip("pandapower", reason="pandapower not installed")
ppl = pytest.importorskip("pypowsybl", reason="pypowsybl not installed")

from powergrid_synth.core.data_format_converter import (
    pandapower_to_nx,
    nx_to_pandapower,
    pandapower_to_pypowsybl,
    pypowsybl_to_nx,
    load_grid,
)


class TestPandapowerToNx:

    def test_case9_conversion(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(net.bus)

    def test_nodes_have_voltage_level(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        for _, data in G.nodes(data=True):
            assert "voltage_level" in data

    def test_nodes_have_bus_type(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        bus_types = {data["bus_type"] for _, data in G.nodes(data=True)}
        assert bus_types.issubset({"Gen", "Load", "Conn"})

    def test_edges_present(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        assert G.number_of_edges() > 0

    def test_base_kv_map_stored(self):
        net = pp.networks.case9()
        G = pandapower_to_nx(net)
        assert "base_kv_map" in G.graph

    def test_case118_conversion(self):
        net = pp.networks.case118()
        G = pandapower_to_nx(net)
        assert G.number_of_nodes() == len(net.bus)
        assert G.number_of_edges() >= len(net.line)


class TestNxToPandapower:

    def test_roundtrip_case9(self):
        """Convert case9 to nx and back to pandapower."""
        net_orig = pp.networks.case9()
        G = pandapower_to_nx(net_orig)
        net_back = nx_to_pandapower(G)
        assert isinstance(net_back, pp.pandapowerNet)
        assert len(net_back.bus) == G.number_of_nodes()

    def test_from_fully_param_grid(self, fully_parameterised_grid):
        net = nx_to_pandapower(fully_parameterised_grid)
        assert isinstance(net, pp.pandapowerNet)
        assert len(net.bus) == fully_parameterised_grid.number_of_nodes()
        # Should have some generators
        assert len(net.gen) + len(net.ext_grid) > 0
        # Should have some loads
        assert len(net.load) > 0
        # Should have lines
        assert len(net.line) + len(net.trafo) > 0

    def test_custom_base_kv_map(self, fully_parameterised_grid):
        net = nx_to_pandapower(
            fully_parameterised_grid,
            base_mva=100.0,
            base_kv_map={0: 380.0, 1: 110.0},
        )
        assert isinstance(net, pp.pandapowerNet)


class TestPandapowerToPypowsybl:

    def test_convert_case9(self):
        net = pp.networks.case9()
        pp.runpp(net)
        ppl_net = pandapower_to_pypowsybl(net)
        assert ppl_net is not None


class TestPypowsyblToNx:

    @pytest.fixture()
    def ieee9(self):
        return ppl.network.create_ieee9()

    @pytest.fixture()
    def ieee14(self):
        return ppl.network.create_ieee14()

    def test_returns_graph(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        assert isinstance(G, nx.Graph)

    def test_node_count_matches_buses(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        assert G.number_of_nodes() == len(ieee9.get_buses())

    def test_edges_present(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        n_lines = len(ieee9.get_lines())
        n_trafos = len(ieee9.get_2_windings_transformers())
        assert G.number_of_edges() == n_lines + n_trafos

    def test_nodes_have_voltage_level(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        for _, data in G.nodes(data=True):
            assert "voltage_level" in data
            assert "vn_kv" in data
            assert isinstance(data["voltage_level"], int)

    def test_nodes_have_bus_type(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        bus_types = {data["bus_type"] for _, data in G.nodes(data=True)}
        assert bus_types.issubset({"Gen", "Load", "Conn"})

    def test_gen_nodes_have_pg(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        gen_nodes = [n for n, d in G.nodes(data=True) if d["bus_type"] == "Gen"]
        assert len(gen_nodes) >= 1
        for n in gen_nodes:
            assert "pg" in G.nodes[n]

    def test_load_nodes_have_pl(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        load_nodes = [n for n, d in G.nodes(data=True) if d["bus_type"] == "Load"]
        assert len(load_nodes) >= 1
        for n in load_nodes:
            assert "pl" in G.nodes[n]

    def test_base_kv_map_stored(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        assert "base_kv_map" in G.graph
        assert len(G.graph["base_kv_map"]) >= 1

    def test_edge_types(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        for u, v, d in G.edges(data=True):
            assert d["type"] in ("line", "transformer")

    def test_line_edges_have_impedance(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        for u, v, d in G.edges(data=True):
            if d["type"] == "line":
                assert "r" in d
                assert "x" in d

    def test_transformer_edges_have_capacity(self, ieee9):
        G = pypowsybl_to_nx(ieee9)
        for u, v, d in G.edges(data=True):
            if d["type"] == "transformer":
                assert "capacity" in d

    def test_ieee14_larger_network(self, ieee14):
        G = pypowsybl_to_nx(ieee14)
        assert G.number_of_nodes() == len(ieee14.get_buses())
        assert G.number_of_edges() > 0
        assert nx.is_connected(G)


class TestLoadGrid:

    def test_load_from_xiidm_file(self):
        """Save an IEEE-9 network as XIIDM, reload via load_grid."""
        net = ppl.network.create_ieee9()
        with tempfile.NamedTemporaryFile(suffix=".xiidm", delete=False) as f:
            net.save(f.name, format="XIIDM")
            path = f.name
        G = load_grid(path)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(net.get_buses())
        import os
        os.unlink(path)

    def test_load_preserves_attributes(self):
        """Loaded grid should have same attributes as direct conversion."""
        net = ppl.network.create_ieee14()
        with tempfile.NamedTemporaryFile(suffix=".xiidm", delete=False) as f:
            net.save(f.name, format="XIIDM")
            path = f.name
        G = load_grid(path)
        assert "base_kv_map" in G.graph
        bus_types = {d["bus_type"] for _, d in G.nodes(data=True)}
        assert bus_types.issubset({"Gen", "Load", "Conn"})
        import os
        os.unlink(path)
