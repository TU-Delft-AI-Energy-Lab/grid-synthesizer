import pytest
import networkx as nx
import matplotlib
matplotlib.use("Agg")
from powergrid_synth.core.analysis import GridAnalyzer


class TestGridAnalyzer:

    @pytest.fixture()
    def connected_graph(self):
        G = nx.barabasi_albert_graph(50, 2, seed=42)
        return G

    @pytest.fixture()
    def disconnected_graph(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        G.add_edges_from([(10, 11), (11, 12)])
        return G

    def test_basic_stats_keys(self, connected_graph):
        analyzer = GridAnalyzer(connected_graph)
        stats = analyzer.get_basic_stats()
        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "density" in stats

    def test_basic_stats_values(self, connected_graph):
        analyzer = GridAnalyzer(connected_graph)
        stats = analyzer.get_basic_stats()
        assert stats["num_nodes"] == 50
        assert stats["num_edges"] > 0
        assert 0 <= stats["density"] <= 1

    def test_path_metrics_connected(self, connected_graph):
        analyzer = GridAnalyzer(connected_graph)
        metrics = analyzer.get_path_metrics()
        assert metrics["is_connected"] is True
        assert metrics["lcc_size"] == 50
        assert metrics["diameter"] > 0
        assert metrics["avg_path_length"] > 0

    def test_path_metrics_disconnected(self, disconnected_graph):
        analyzer = GridAnalyzer(disconnected_graph)
        metrics = analyzer.get_path_metrics()
        assert metrics["is_connected"] is False
        assert metrics["lcc_size"] == 3

    def test_clustering_metrics(self, connected_graph):
        analyzer = GridAnalyzer(connected_graph)
        clust = analyzer.get_clustering_metrics()
        assert "avg_clustering_coef" in clust
        assert "transitivity" in clust
        assert 0 <= clust["avg_clustering_coef"] <= 1
        assert 0 <= clust["transitivity"] <= 1

    def test_analyze_does_not_crash(self, connected_graph, capsys):
        analyzer = GridAnalyzer(connected_graph)
        analyzer.analyze()
        captured = capsys.readouterr()
        assert "Power Grid Topological Analysis" in captured.out

    def test_plot_degree_distribution_log(self, connected_graph):
        analyzer = GridAnalyzer(connected_graph)
        analyzer.plot_degree_distribution(log_scale=True)

    def test_plot_degree_distribution_linear(self, connected_graph):
        analyzer = GridAnalyzer(connected_graph)
        analyzer.plot_degree_distribution(log_scale=False)
