import pytest
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from powergrid_synth.core.comparison import GraphComparator


class TestGraphComparator:

    @pytest.fixture()
    def two_graphs(self):
        g1 = nx.barabasi_albert_graph(40, 2, seed=10)
        g2 = nx.barabasi_albert_graph(45, 2, seed=20)
        return g1, g2

    @pytest.fixture()
    def level_graphs(self):
        """Two graphs with voltage_level attributes for per-level comparison."""
        def _make(n, seed):
            G = nx.barabasi_albert_graph(n, 2, seed=seed)
            for node in G.nodes():
                G.nodes[node]["voltage_level"] = 0 if node < n // 2 else 1
            return G
        return _make(40, 10), _make(45, 20)

    @pytest.fixture(autouse=True)
    def cleanup_figures(self):
        """Close all matplotlib figures after each test."""
        yield
        plt.close("all")

    def test_init(self, two_graphs):
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        assert comp.synth_graph is g1
        assert comp.ref_graph is g2

    def test_get_comparison_data_returns_dataframe(self, two_graphs):
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        df = comp._get_comparison_data(g1, g2)
        assert "Metric" in df.columns
        assert len(df) == 8  # 8 metrics

    def test_print_metric_comparison(self, two_graphs, capsys):
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        comp.print_metric_comparison()
        captured = capsys.readouterr()
        assert "GRAPH COMPARISON REPORT" in captured.out

    def test_plot_degree_comparison(self, two_graphs):
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        comp.plot_degree_comparison()  # should not raise

    def test_print_level_metrics(self, level_graphs, capsys):
        g1, g2 = level_graphs
        comp = GraphComparator(g1, g2)
        comp.print_level_metrics()
        captured = capsys.readouterr()
        assert "Level" in captured.out or "LEVEL" in captured.out or len(captured.out) > 0

    def test_plot_degree_comparison_empty_graphs(self):
        """Empty graphs trigger early return (line 103)."""
        empty1 = nx.Graph()
        empty2 = nx.Graph()
        comp = GraphComparator(empty1, empty2)
        result = comp.plot_degree_comparison()
        assert result is None

    def test_plot_degree_comparison_linear_scale(self, two_graphs):
        """Exercise the linear-scale bar chart branch (lines 133-171)."""
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        comp.plot_degree_comparison(log_scale=False)

    def test_plot_degree_comparison_show_lines(self, two_graphs):
        """Exercise show_lines=True branch (lines 125-126)."""
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        comp.plot_degree_comparison(show_lines=True)

    def test_plot_degree_comparison_with_custom_graphs(self, two_graphs):
        """Pass explicit synth_graph/ref_graph parameters."""
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        custom1 = nx.barabasi_albert_graph(30, 2, seed=42)
        custom2 = nx.barabasi_albert_graph(35, 2, seed=43)
        comp.plot_degree_comparison(synth_graph=custom1, ref_graph=custom2)

    def test_plot_degree_comparison_with_ax(self, two_graphs):
        """Pass a pre-created matplotlib Axes object."""
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        fig, ax = plt.subplots()
        comp.plot_degree_comparison(ax=ax)
        assert len(ax.get_lines()) > 0 or len(ax.collections) > 0

    def test_print_level_metrics_no_common_levels(self, capsys):
        """No common voltage_level attributes prints message (lines 192-193)."""
        g1 = nx.barabasi_albert_graph(20, 2, seed=10)
        g2 = nx.barabasi_albert_graph(20, 2, seed=20)
        for n in g1.nodes():
            g1.nodes[n]["voltage_level"] = 0
        for n in g2.nodes():
            g2.nodes[n]["voltage_level"] = 1
        comp = GraphComparator(g1, g2)
        comp.print_level_metrics()
        captured = capsys.readouterr()
        assert "No common" in captured.out

    def test_plot_level_topology_comparison(self, level_graphs):
        """Exercise plot_level_topology_comparison (lines 211-313)."""
        g1, g2 = level_graphs
        comp = GraphComparator(g1, g2)
        comp.plot_level_topology_comparison()

    def test_plot_level_topology_comparison_no_common_levels(self, capsys):
        """No common levels prints message and returns (lines 215-217)."""
        g1 = nx.barabasi_albert_graph(20, 2, seed=10)
        g2 = nx.barabasi_albert_graph(20, 2, seed=20)
        for n in g1.nodes():
            g1.nodes[n]["voltage_level"] = 0
        for n in g2.nodes():
            g2.nodes[n]["voltage_level"] = 1
        comp = GraphComparator(g1, g2)
        comp.plot_level_topology_comparison()
        captured = capsys.readouterr()
        assert "No common levels" in captured.out

    def test_plot_all_levels_comparison(self, level_graphs):
        """Exercise plot_all_levels_comparison (lines 319-354)."""
        g1, g2 = level_graphs
        comp = GraphComparator(g1, g2)
        comp.plot_all_levels_comparison()

    def test_plot_all_levels_comparison_log_false(self, level_graphs):
        """Exercise plot_all_levels_comparison with log_scale=False."""
        g1, g2 = level_graphs
        comp = GraphComparator(g1, g2)
        comp.plot_all_levels_comparison(log_scale=False)

    def test_plot_all_levels_comparison_no_common_levels(self, capsys):
        """No common levels prints message and returns (lines 323-325)."""
        g1 = nx.barabasi_albert_graph(20, 2, seed=10)
        g2 = nx.barabasi_albert_graph(20, 2, seed=20)
        for n in g1.nodes():
            g1.nodes[n]["voltage_level"] = 0
        for n in g2.nodes():
            g2.nodes[n]["voltage_level"] = 1
        comp = GraphComparator(g1, g2)
        comp.plot_all_levels_comparison()
        captured = capsys.readouterr()
        assert "No common levels" in captured.out

    def test_run_full_comparison(self, level_graphs):
        """Exercise run_full_comparison (lines 358-369)."""
        g1, g2 = level_graphs
        comp = GraphComparator(g1, g2)
        comp.run_full_comparison()

    def test_print_metric_comparison_custom_graphs(self, two_graphs, capsys):
        """Pass custom graphs to print_metric_comparison."""
        g1, g2 = two_graphs
        comp = GraphComparator(g1, g2)
        custom1 = nx.barabasi_albert_graph(25, 2, seed=50)
        custom2 = nx.barabasi_albert_graph(30, 2, seed=51)
        comp.print_metric_comparison(synth_graph=custom1, ref_graph=custom2,
                                     title="CUSTOM COMPARISON")
        captured = capsys.readouterr()
        assert "CUSTOM COMPARISON" in captured.out

    def test_plot_all_levels_single_level(self, capsys):
        """Single common level triggers axes=[axes] branch (line 334)."""
        g1 = nx.barabasi_albert_graph(30, 2, seed=10)
        g2 = nx.barabasi_albert_graph(35, 2, seed=20)
        for n in g1.nodes():
            g1.nodes[n]["voltage_level"] = 0
        for n in g2.nodes():
            g2.nodes[n]["voltage_level"] = 0
        comp = GraphComparator(g1, g2)
        comp.plot_all_levels_comparison()
        captured = capsys.readouterr()
        assert "1 Levels" in captured.out
        fig = plt.gcf()
        assert fig is not None
        assert len(fig.get_axes()) == 1
