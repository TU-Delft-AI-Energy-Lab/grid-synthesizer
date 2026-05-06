import pytest
import networkx as nx
import matplotlib
matplotlib.use("Agg")
from powergrid_synth.core.hierarchical_analysis import HierarchicalAnalyzer


class TestHierarchicalAnalyzer:

    @pytest.fixture()
    def multi_level_graph(self):
        G = nx.Graph()
        # Level 0: 10 nodes
        for i in range(10):
            G.add_node(i, voltage_level=0)
        for i in range(9):
            G.add_edge(i, i + 1)
        # Level 1: 15 nodes
        for i in range(10, 25):
            G.add_node(i, voltage_level=1)
        for i in range(10, 24):
            G.add_edge(i, i + 1)
        # Transformer
        G.add_edge(5, 15)
        return G

    def test_run_global_analysis(self, multi_level_graph, capsys):
        analyzer = HierarchicalAnalyzer(multi_level_graph)
        analyzer.run_global_analysis()
        captured = capsys.readouterr()
        assert "GLOBAL GRID ANALYSIS" in captured.out

    def test_run_level_analysis(self, multi_level_graph, capsys):
        analyzer = HierarchicalAnalyzer(multi_level_graph)
        analyzer.run_level_analysis()
        captured = capsys.readouterr()
        assert "VOLTAGE LEVEL 0" in captured.out
        assert "VOLTAGE LEVEL 1" in captured.out

    def test_plot_all_levels(self, multi_level_graph):
        analyzer = HierarchicalAnalyzer(multi_level_graph)
        analyzer.plot_all_levels(log_scale=False)  # should not raise

    def test_no_voltage_levels(self, capsys):
        G = nx.cycle_graph(10)
        analyzer = HierarchicalAnalyzer(G)
        analyzer.run_level_analysis()
        captured = capsys.readouterr()
        assert "No 'voltage_level'" in captured.out
