"""Tests for powergrid_synth.distribution modules."""

import math

import networkx as nx
import numpy as np
import pytest

from powergrid_synth.distribution import (
    CableLibraryEntry,
    DistributionInputModel,
    DistributionSynthParams,
    FeederInputSample,
    SchweetzerFeederGenerator,
    compare_feeders,
    compute_emergent_properties,
    fit_params_from_feeders,
    kl_divergence_discrete,
    validate_tree,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def generator():
    return SchweetzerFeederGenerator(seed=42)


@pytest.fixture
def small_feeder(generator):
    return generator.generate_feeder(n_nodes=20, total_load_mw=3.0)


@pytest.fixture
def feeder_with_gen(generator):
    return generator.generate_feeder(n_nodes=30, total_load_mw=5.0, total_gen_mw=0.5)


# ------------------------------------------------------------------
# distribution_params tests
# ------------------------------------------------------------------

class TestDistributionParams:
    def test_default_params(self):
        p = DistributionSynthParams()
        assert p.hop_dist.r == pytest.approx(3.14)
        assert p.hop_dist.p == pytest.approx(0.41)
        assert len(p.cable_library) == 6
        assert len(p.pf_cdf) == 10

    def test_custom_params(self):
        from powergrid_synth.distribution import NegBinomParams
        p = DistributionSynthParams(hop_dist=NegBinomParams(r=5.0, p=0.3))
        assert p.hop_dist.r == 5.0

    def test_cable_library_entries(self):
        p = DistributionSynthParams()
        for cable in p.cable_library:
            assert cable.max_i_ka > 0
            assert cable.r_ohm_per_km > 0
            assert cable.frequency > 0


# ------------------------------------------------------------------
# distribution_synthesis tests
# ------------------------------------------------------------------

class TestSchweetzerFeederGenerator:
    def test_creates_correct_node_count(self, generator):
        G = generator.generate_feeder(n_nodes=25, total_load_mw=4.0)
        assert G.number_of_nodes() == 25

    def test_creates_tree(self, small_feeder):
        G = small_feeder
        assert G.number_of_edges() == G.number_of_nodes() - 1

    def test_connected(self, small_feeder):
        assert nx.is_connected(small_feeder)

    def test_source_at_hop_zero(self, small_feeder):
        assert small_feeder.nodes[0]["h"] == 0

    def test_root_at_hop_one(self, small_feeder):
        assert small_feeder.nodes[1]["h"] == 1

    def test_hop_distances_monotonic_on_edges(self, small_feeder):
        for u, v in small_feeder.edges:
            assert abs(small_feeder.nodes[u]["h"] - small_feeder.nodes[v]["h"]) == 1

    def test_all_nodes_have_power_factor(self, small_feeder):
        for n in small_feeder.nodes:
            assert 0 < small_feeder.nodes[n]["pf"] <= 1.0

    def test_has_cable_attributes(self, small_feeder):
        for e in small_feeder.edges:
            assert "cable_type" in small_feeder.edges[e]
            assert "length_km" in small_feeder.edges[e]
            assert small_feeder.edges[e]["length_km"] > 0

    def test_has_impedance(self, small_feeder):
        for e in small_feeder.edges:
            assert "r_ohm" in small_feeder.edges[e]
            assert "x_ohm" in small_feeder.edges[e]

    def test_load_nodes_have_positive_power(self, small_feeder):
        for n in small_feeder.nodes:
            if small_feeder.nodes[n].get("node_type") == "load":
                assert small_feeder.nodes[n]["P_mw"] > 0

    def test_injection_nodes_have_negative_power(self, feeder_with_gen):
        inj = [
            n for n in feeder_with_gen.nodes
            if feeder_with_gen.nodes[n].get("node_type") == "injection"
        ]
        for n in inj:
            assert feeder_with_gen.nodes[n]["P_mw"] < 0

    def test_intermediate_nodes_zero_power(self, small_feeder):
        for n in small_feeder.nodes:
            if small_feeder.nodes[n].get("node_type") == "intermediate":
                assert small_feeder.nodes[n]["P_mw"] == 0.0

    def test_reproducible_with_seed(self):
        g1 = SchweetzerFeederGenerator(seed=99)
        g2 = SchweetzerFeederGenerator(seed=99)
        G1 = g1.generate_feeder(20, 3.0)
        G2 = g2.generate_feeder(20, 3.0)
        assert G1.number_of_nodes() == G2.number_of_nodes()
        assert G1.number_of_edges() == G2.number_of_edges()
        for n in G1.nodes:
            assert G1.nodes[n]["h"] == G2.nodes[n]["h"]

    def test_nominal_voltage_stored(self, small_feeder):
        assert small_feeder.graph["v_nom_kv"] == 10.0

    def test_custom_voltage(self, generator):
        G = generator.generate_feeder(15, 2.0, v_nom_kv=20.0)
        assert G.graph["v_nom_kv"] == 20.0


# ------------------------------------------------------------------
# distribution_validation tests
# ------------------------------------------------------------------

class TestValidation:
    def test_validate_tree_ok(self, small_feeder):
        issues = validate_tree(small_feeder)
        assert issues == []

    def test_validate_detects_disconnected(self):
        G = nx.Graph()
        G.add_node(0, h=0)
        G.add_node(1, h=1)
        # no edge
        issues = validate_tree(G)
        assert any("not connected" in i for i in issues)

    def test_validate_detects_cycle(self):
        G = nx.Graph()
        for i in range(4):
            G.add_node(i, h=i)
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 1)  # cycle
        issues = validate_tree(G)
        assert any("Not a tree" in i for i in issues)

    def test_kl_divergence_identical(self):
        p = np.array([10, 20, 30])
        assert kl_divergence_discrete(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_kl_divergence_positive(self):
        p = np.array([1, 2, 3, 4])
        q = np.array([4, 3, 2, 1])
        assert kl_divergence_discrete(p, q) > 0

    def test_emergent_properties_keys(self, small_feeder):
        props = compute_emergent_properties(small_feeder)
        expected_keys = {
            "n_nodes", "n_edges", "max_hop", "mean_degree",
            "total_load_mw", "total_gen_mw",
            "frac_intermediate", "frac_injection",
            "mean_length_km", "max_length_km", "total_length_km",
        }
        assert set(props.keys()) == expected_keys

    def test_compare_feeders_returns_kl(self, generator):
        G1 = generator.generate_feeder(20, 3.0)
        gen2 = SchweetzerFeederGenerator(seed=123)
        G2 = gen2.generate_feeder(20, 3.0)
        result = compare_feeders(G1, G2)
        assert "hop_distance_kl" in result
        assert "degree_kl" in result


# ------------------------------------------------------------------
# distribution_input_model tests
# ------------------------------------------------------------------

class TestInputModel:
    def test_fit_and_sample(self, generator):
        feeders = [
            generator.generate_feeder(20 + i * 5, 3.0 + i, 0.1 * (i + 1))
            for i in range(5)
        ]
        model = DistributionInputModel(seed=42)
        model.fit(feeders)
        samples = model.sample(5)
        assert len(samples) == 5
        for s in samples:
            assert isinstance(s, FeederInputSample)
            assert s.n_nodes >= 3
            assert s.total_load_mw >= 0

    def test_fit_from_arrays(self):
        model = DistributionInputModel(seed=42)
        model.fit_from_arrays(
            n_nodes=[20, 30, 40, 25, 55, 35],
            total_load=[3.0, 5.0, 7.0, 4.5, 8.0, 6.2],
            total_gen=[0.0, 0.5, 1.0, 0.2, 2.0, 0.3],
        )
        samples = model.sample(3)
        assert len(samples) == 3

    def test_pdf(self):
        model = DistributionInputModel(seed=42)
        model.fit_from_arrays(
            [20, 30, 40, 25, 55, 35],
            [3.0, 5.0, 7.0, 4.5, 8.0, 6.2],
            [0, 0.5, 1.0, 0.2, 2.0, 0.3],
        )
        d = model.pdf(25.0, 4.0, 0.25)
        assert d > 0

    def test_sample_without_fit_raises(self):
        model = DistributionInputModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.sample()


# ------------------------------------------------------------------
# distribution_analysis tests
# ------------------------------------------------------------------

class TestAnalysis:
    def test_fit_params_roundtrip(self, generator):
        feeders = [generator.generate_feeder(30, 5.0, 0.5) for _ in range(5)]
        params = fit_params_from_feeders(feeders)
        assert isinstance(params, DistributionSynthParams)
        assert params.hop_dist.r > 0
        assert params.hop_dist.p > 0
        assert params.degree_dist.pi > 0

    def test_fit_with_few_feeders(self, small_feeder):
        params = fit_params_from_feeders([small_feeder])
        assert isinstance(params, DistributionSynthParams)


# ------------------------------------------------------------------
# distribution_converter tests
# ------------------------------------------------------------------

class TestDistributionConverter:
    """Tests for pandapower_to_feeders and feeder_summary."""

    @pytest.fixture()
    def cigre_lv(self):
        import pandapower.networks as pn
        return pn.create_cigre_network_lv()

    def test_pandapower_to_feeders_returns_list(self, cigre_lv):
        from powergrid_synth.distribution import pandapower_to_feeders
        feeders = pandapower_to_feeders(cigre_lv)
        assert isinstance(feeders, list)
        assert len(feeders) >= 1

    def test_feeder_has_required_node_attributes(self, cigre_lv):
        from powergrid_synth.distribution import pandapower_to_feeders
        feeders = pandapower_to_feeders(cigre_lv)
        for f in feeders:
            for n in f.nodes:
                assert "h" in f.nodes[n], f"Node {n} missing 'h'"
                assert "P_mw" in f.nodes[n], f"Node {n} missing 'P_mw'"
                assert "node_type" in f.nodes[n], f"Node {n} missing 'node_type'"
                assert "pf" in f.nodes[n], f"Node {n} missing 'pf'"

    def test_feeder_has_required_edge_attributes(self, cigre_lv):
        from powergrid_synth.distribution import pandapower_to_feeders
        feeders = pandapower_to_feeders(cigre_lv)
        for f in feeders:
            for u, v, d in f.edges(data=True):
                assert "length_km" in d
                assert d["length_km"] >= 0

    def test_feeder_hop_distances_valid(self, cigre_lv):
        from powergrid_synth.distribution import pandapower_to_feeders
        feeders = pandapower_to_feeders(cigre_lv)
        for f in feeders:
            hops = [f.nodes[n]["h"] for n in f.nodes]
            assert min(hops) == 0  # root exists
            # hop diff on each edge should be 1
            for u, v in f.edges:
                assert abs(f.nodes[u]["h"] - f.nodes[v]["h"]) == 1

    def test_feeder_node_types_valid(self, cigre_lv):
        from powergrid_synth.distribution import pandapower_to_feeders
        feeders = pandapower_to_feeders(cigre_lv)
        valid_types = {"load", "injection", "intermediate"}
        for f in feeders:
            for n in f.nodes:
                assert f.nodes[n]["node_type"] in valid_types

    def test_feeder_is_connected(self, cigre_lv):
        from powergrid_synth.distribution import pandapower_to_feeders
        feeders = pandapower_to_feeders(cigre_lv)
        for f in feeders:
            assert nx.is_connected(f)

    def test_feeder_summary_keys(self, cigre_lv):
        from powergrid_synth.distribution import pandapower_to_feeders, feeder_summary
        feeders = pandapower_to_feeders(cigre_lv)
        summaries = feeder_summary(feeders)
        assert len(summaries) == len(feeders)
        expected_keys = {"n_nodes", "n_edges", "max_hop", "is_tree",
                         "total_load_mw", "total_gen_mw", "node_types"}
        for s in summaries:
            assert set(s.keys()) == expected_keys

    def test_feeder_summary_values_consistent(self, cigre_lv):
        from powergrid_synth.distribution import pandapower_to_feeders, feeder_summary
        feeders = pandapower_to_feeders(cigre_lv)
        summaries = feeder_summary(feeders)
        for f, s in zip(feeders, summaries):
            assert s["n_nodes"] == f.number_of_nodes()
            assert s["n_edges"] == f.number_of_edges()

    def test_feeders_compatible_with_fit_params(self, cigre_lv):
        from powergrid_synth.distribution import pandapower_to_feeders
        feeders = pandapower_to_feeders(cigre_lv)
        params = fit_params_from_feeders(feeders)
        assert isinstance(params, DistributionSynthParams)

    def test_small_components_filtered(self):
        """Components with < 3 nodes should be excluded."""
        from powergrid_synth.distribution import pandapower_to_feeders
        import pandapower as pp
        # Create a minimal net with one 2-node component
        net = pp.create_empty_network()
        pp.create_buses(net, 2, vn_kv=0.4)
        pp.create_line_from_parameters(
            net, from_bus=0, to_bus=1,
            length_km=0.1, r_ohm_per_km=0.1, x_ohm_per_km=0.1,
            c_nf_per_km=0, max_i_ka=0.3,
        )
        pp.create_ext_grid(net, bus=0)
        feeders = pandapower_to_feeders(net)
        assert len(feeders) == 0  # 2 nodes < 3 threshold


class TestPypowsyblToFeeders:
    """Tests for pypowsybl_to_feeders."""

    ppl = pytest.importorskip("pypowsybl", reason="pypowsybl not installed")

    @pytest.fixture()
    def ieee9_ppl(self):
        import pypowsybl as ppl
        return ppl.network.create_ieee9()

    @pytest.fixture()
    def ieee14_ppl(self):
        import pypowsybl as ppl
        return ppl.network.create_ieee14()

    def test_returns_list(self, ieee9_ppl):
        from powergrid_synth.distribution import pypowsybl_to_feeders
        feeders = pypowsybl_to_feeders(ieee9_ppl)
        assert isinstance(feeders, list)
        assert len(feeders) >= 1

    def test_feeder_has_required_node_attributes(self, ieee9_ppl):
        from powergrid_synth.distribution import pypowsybl_to_feeders
        feeders = pypowsybl_to_feeders(ieee9_ppl)
        for f in feeders:
            for n in f.nodes:
                assert "h" in f.nodes[n], f"Node {n} missing 'h'"
                assert "P_mw" in f.nodes[n], f"Node {n} missing 'P_mw'"
                assert "node_type" in f.nodes[n], f"Node {n} missing 'node_type'"
                assert "pf" in f.nodes[n], f"Node {n} missing 'pf'"

    def test_feeder_has_required_edge_attributes(self, ieee9_ppl):
        from powergrid_synth.distribution import pypowsybl_to_feeders
        feeders = pypowsybl_to_feeders(ieee9_ppl)
        for f in feeders:
            for u, v, d in f.edges(data=True):
                assert "length_km" in d
                assert d["length_km"] >= 0

    def test_feeder_hop_distances_valid(self, ieee9_ppl):
        from powergrid_synth.distribution import pypowsybl_to_feeders
        feeders = pypowsybl_to_feeders(ieee9_ppl)
        for f in feeders:
            hops = [f.nodes[n]["h"] for n in f.nodes]
            assert min(hops) == 0  # root exists

    def test_feeder_node_types_valid(self, ieee9_ppl):
        from powergrid_synth.distribution import pypowsybl_to_feeders
        feeders = pypowsybl_to_feeders(ieee9_ppl)
        valid_types = {"load", "injection", "intermediate"}
        for f in feeders:
            for n in f.nodes:
                assert f.nodes[n]["node_type"] in valid_types

    def test_feeder_is_connected(self, ieee9_ppl):
        from powergrid_synth.distribution import pypowsybl_to_feeders
        feeders = pypowsybl_to_feeders(ieee9_ppl)
        for f in feeders:
            assert nx.is_connected(f)

    def test_ieee14_feeders(self, ieee14_ppl):
        from powergrid_synth.distribution import pypowsybl_to_feeders
        feeders = pypowsybl_to_feeders(ieee14_ppl)
        assert isinstance(feeders, list)
        assert len(feeders) >= 1
        total_nodes = sum(f.number_of_nodes() for f in feeders)
        assert total_nodes >= 3

    def test_feeders_compatible_with_fit_params(self, ieee9_ppl):
        from powergrid_synth.distribution import pypowsybl_to_feeders
        feeders = pypowsybl_to_feeders(ieee9_ppl)
        params = fit_params_from_feeders(feeders)
        assert isinstance(params, DistributionSynthParams)
