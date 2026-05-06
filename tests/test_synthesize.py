import pytest
import os
import tempfile
import networkx as nx

pytest.importorskip("pandapower", reason="pandapower not installed")

from powergrid_synth.transmission.synthesize import synthesize


class TestSynthesizeModeReference:

    @pytest.fixture()
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_reference_case118(self, tmpdir):
        grid = synthesize(
            mode="reference",
            reference_case="case118",
            seed=42,
            output_dir=tmpdir,
            output_name="test_ref",
            export_formats=["json"],
        )
        assert isinstance(grid, nx.Graph)
        assert grid.number_of_nodes() > 0
        assert grid.number_of_edges() > 0
        assert os.path.isfile(os.path.join(tmpdir, "test_ref.json"))

    def test_reference_custom_net(self, tmpdir):
        import pandapower.networks as pn
        net = pn.case9()
        grid = synthesize(
            mode="reference",
            reference_net=net,
            seed=42,
            output_dir=tmpdir,
            output_name="test_custom",
            export_formats=["json"],
        )
        assert isinstance(grid, nx.Graph)
        assert grid.number_of_nodes() > 0

    def test_reference_multiple_formats(self, tmpdir):
        pytest.importorskip("pypowsybl", reason="pypowsybl required for matpower export")
        grid = synthesize(
            mode="reference",
            reference_case="case_ieee30",
            seed=99,
            output_dir=tmpdir,
            output_name="multi_fmt",
            export_formats=["json", "matpower"],
        )
        assert isinstance(grid, nx.Graph)

    def test_reference_nodes_have_bus_type(self, tmpdir):
        grid = synthesize(
            mode="reference",
            reference_case="case_ieee30",
            seed=42,
            output_dir=tmpdir,
            export_formats=["json"],
        )
        for _, data in grid.nodes(data=True):
            assert "bus_type" in data

    def test_reference_nodes_have_electrical_attrs(self, tmpdir):
        grid = synthesize(
            mode="reference",
            reference_case="case_ieee30",
            seed=42,
            output_dir=tmpdir,
            export_formats=["json"],
        )
        gen_count = sum(
            1 for _, d in grid.nodes(data=True) if d.get("bus_type") == "Gen"
        )
        assert gen_count > 0
        # At least some gen nodes should have pg_max
        pg_max_nodes = [
            d.get("pg_max", 0)
            for _, d in grid.nodes(data=True)
            if d.get("bus_type") == "Gen"
        ]
        assert any(v > 0 for v in pg_max_nodes)


class TestSynthesizeModeSynthetic:

    @pytest.fixture()
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_synthetic_basic(self, tmpdir):
        grid = synthesize(
            mode="synthetic",
            level_specs=[
                {"n": 20, "avg_k": 2.5, "diam": 5, "dist_type": "poisson"},
                {"n": 30, "avg_k": 2.0, "diam": 8, "dist_type": "poisson"},
            ],
            connection_specs={
                (0, 1): {"type": "k-stars", "c": 0.174, "gamma": 4.15},
            },
            seed=42,
            output_dir=tmpdir,
            output_name="test_syn",
            export_formats=["json"],
        )
        assert isinstance(grid, nx.Graph)
        assert grid.number_of_nodes() > 0
        assert grid.number_of_edges() > 0

    def test_synthetic_edges_have_impedance(self, tmpdir):
        grid = synthesize(
            mode="synthetic",
            level_specs=[
                {"n": 20, "avg_k": 2.5, "diam": 5, "dist_type": "poisson"},
            ],
            connection_specs={},
            seed=42,
            output_dir=tmpdir,
            export_formats=["json"],
        )
        for u, v, data in grid.edges(data=True):
            assert "x" in data
            assert "r" in data


class TestSynthesizeValidation:

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            synthesize(mode="invalid")

    def test_reference_without_case_raises(self):
        with pytest.raises(ValueError, match="requires reference_net"):
            synthesize(mode="reference")

    def test_synthetic_without_specs_raises(self):
        with pytest.raises(ValueError, match="requires both"):
            synthesize(mode="synthetic")

    def test_unknown_case_raises(self):
        with pytest.raises(ValueError, match="Unknown reference case"):
            synthesize(
                mode="reference",
                reference_case="case_does_not_exist_99999",
            )
