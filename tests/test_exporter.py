import pytest
import os
import tempfile

pytest.importorskip("pandapower", reason="pandapower not installed")
pytest.importorskip("pypowsybl", reason="pypowsybl not installed")

from powergrid_synth.core.exporter import GridExporter

_has_xlsxwriter = pytest.importorskip("xlsxwriter", reason="xlsxwriter not installed") if False else None

try:
    import xlsxwriter  # noqa: F401
    _has_xlsxwriter = True
except ImportError:
    _has_xlsxwriter = False


class TestGridExporter:

    @pytest.fixture()
    def exporter(self, fully_parameterised_grid):
        return GridExporter(fully_parameterised_grid)

    @pytest.fixture()
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_to_json(self, exporter, tmpdir):
        path = os.path.join(tmpdir, "grid.json")
        exporter.to_json(path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    @pytest.mark.skipif(not _has_xlsxwriter, reason="xlsxwriter not installed")
    def test_to_excel(self, exporter, tmpdir):
        path = os.path.join(tmpdir, "grid.xlsx")
        exporter.to_excel(path)
        assert os.path.isfile(path)

    def test_to_sqlite(self, exporter, tmpdir):
        path = os.path.join(tmpdir, "grid.sqlite")
        exporter.to_sqlite(path)
        assert os.path.isfile(path)

    def test_to_pickle(self, exporter, tmpdir):
        path = os.path.join(tmpdir, "grid.p")
        exporter.to_pickle(path)
        assert os.path.isfile(path)

    def test_to_pypowsybl_xiidm(self, exporter, tmpdir):
        path = os.path.join(tmpdir, "grid.xiidm")
        exporter.to_pypowsybl(path, format="XIIDM")
        assert os.path.isfile(path)

    def test_to_matpower(self, exporter, tmpdir):
        path = os.path.join(tmpdir, "grid")
        exporter.to_matpower(path)
        # matpower creates file(s) at path
        assert any(f.startswith("grid") for f in os.listdir(tmpdir))

    def test_to_cgmes(self, exporter, tmpdir):
        path = os.path.join(tmpdir, "grid_cgmes")
        exporter.to_cgmes(path)
        # CGMES creates a directory or zip with XML files
        assert len(os.listdir(tmpdir)) > 0

    def test_to_psse(self, exporter, tmpdir):
        path = os.path.join(tmpdir, "grid")
        exporter.to_psse(path)
        assert len(os.listdir(tmpdir)) > 0

    def test_custom_base_kv_map(self, fully_parameterised_grid, tmpdir):
        exp = GridExporter(
            fully_parameterised_grid,
            base_kv_map={0: 380.0, 1: 110.0},
        )
        path = os.path.join(tmpdir, "grid.json")
        exp.to_json(path)
        assert os.path.isfile(path)
