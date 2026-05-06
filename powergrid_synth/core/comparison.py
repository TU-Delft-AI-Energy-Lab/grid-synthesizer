r"""
This module conveniently compares the synthetically generated power grid topology with the real power grid topology. 

TODO: perhaps also include another module that compares the other parts of the generated data like bus types and so on. 
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
from .analysis import GridAnalyzer

class GraphComparator:
    """
    Compares a synthetic power grid against a reference (real-world) graph.
    Provides tabular metric comparisons and visual distribution overlaps,
    both globally and per voltage level.
    """
    def __init__(self, synth_graph: nx.Graph, ref_graph: nx.Graph, 
                 synth_label: str = "Synthetic", ref_label: str = "Reference (Real)"):
        self.synth_graph = synth_graph
        self.ref_graph = ref_graph
        self.synth_label = synth_label
        self.ref_label = ref_label

    def _get_comparison_data(self, graph1: nx.Graph, graph2: nx.Graph) -> pd.DataFrame:
        """Helper to generate the dataframe for two specific graphs."""
        analyzer1 = GridAnalyzer(graph1)
        analyzer2 = GridAnalyzer(graph2)
        
        # Gather metrics
        s_basic = analyzer1.get_basic_stats()
        s_path = analyzer1.get_path_metrics()
        s_clust = analyzer1.get_clustering_metrics()
        
        r_basic = analyzer2.get_basic_stats()
        r_path = analyzer2.get_path_metrics()
        r_clust = analyzer2.get_clustering_metrics()

        data = {
            "Metric": [
                "Nodes", "Edges", "Density", 
                "Connected?", "Diameter (LCC)", "Avg Path Len (LCC)", 
                "Avg Clustering", "Transitivity"
            ],
            self.synth_label: [
                s_basic['num_nodes'], s_basic['num_edges'], f"{s_basic['density']:.6f}",
                "Yes" if s_path['is_connected'] else "No", 
                s_path['diameter'], f"{s_path['avg_path_length']:.4f}",
                f"{s_clust['avg_clustering_coef']:.4f}", f"{s_clust['transitivity']:.4f}"
            ],
            self.ref_label: [
                r_basic['num_nodes'], r_basic['num_edges'], f"{r_basic['density']:.6f}",
                "Yes" if r_path['is_connected'] else "No", 
                r_path['diameter'], f"{r_path['avg_path_length']:.4f}",
                f"{r_clust['avg_clustering_coef']:.4f}", f"{r_clust['transitivity']:.4f}"
            ]
        }
        return pd.DataFrame(data)

    def print_metric_comparison(self, synth_graph: Optional[nx.Graph] = None, 
                              ref_graph: Optional[nx.Graph] = None, 
                              title: str = "GRAPH COMPARISON REPORT"):
        """Prints a side-by-side table of topological metrics."""
        s_g = synth_graph if synth_graph is not None else self.synth_graph
        r_g = ref_graph if ref_graph is not None else self.ref_graph
        
        df = self._get_comparison_data(s_g, r_g)
        
        print("\n" + "="*60)
        print(title)
        print("="*60)
        print(df.to_string(index=False))
        print("="*60 + "\n")

    def plot_degree_comparison(self, synth_graph: Optional[nx.Graph] = None, 
                             ref_graph: Optional[nx.Graph] = None, 
                             ax: Optional[plt.Axes] = None,
                             log_scale: bool = True,
                             fig_size: Tuple = (8, 5),
                             show_lines: bool = False,
                             title: str = "Degree Distribution Comparison"):
        """
        Plots overlaid degree distributions.
        
        Args:
            synth_graph: Custom synthetic graph (or None for self.synth_graph).
            ref_graph: Custom reference graph (or None for self.ref_graph).
            ax: Matplotlib axis to plot on. If None, creates new figure.
            log_scale: Whether to use log-log scale (default True).
            title: Title for the plot.
        """
        s_g = synth_graph if synth_graph is not None else self.synth_graph
        r_g = ref_graph if ref_graph is not None else self.ref_graph
        
        deg_synth = [d for n, d in s_g.degree()]
        deg_ref = [d for n, d in r_g.degree()]
        
        if not deg_synth or not deg_ref:
            # Handle empty graphs
            return
        
        created_figure = False
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
            created_figure = True
        
        if log_scale:
            # Log-Log Plot
            # Helper to get log-log coordinates
            def get_log_coords(degrees):
                counts = np.bincount(degrees)
                vals = np.nonzero(counts)[0]
                return vals, counts[vals]
            
            x_s, y_s = get_log_coords(deg_synth)
            x_r, y_r = get_log_coords(deg_ref)
            
            ax.loglog(x_s, y_s, 'bo', markersize=5, alpha=0.7, label=self.synth_label)
            ax.loglog(x_r, y_r, 'r^', markersize=5, alpha=0.7, label=self.ref_label)
            
            if show_lines:
                ax.loglog(x_s, y_s, '-', linewidth=1, alpha=0.2)
                ax.loglog(x_r, y_r, '-', linewidth=1, alpha=0.2)
            
            ax.set_xlabel("Degree (log)")
            ax.set_ylabel("Count (log)")
        else:
            # Side-by-Side Bar Chart for Linear Scale
            # Use discrete integer counting for precise alignment
            from collections import Counter
            import matplotlib.ticker as ticker
            
            counts_s = Counter(deg_synth)
            counts_r = Counter(deg_ref)
            
            min_deg = min(min(deg_synth), min(deg_ref))
            max_deg = max(max(deg_synth), max(deg_ref))
            
            # Create array of all integers in range
            all_k = np.arange(min_deg, max_deg + 1)
            
            # Calculate probabilities
            n_s = len(deg_synth)
            n_r = len(deg_ref)
            probs_s = [counts_s.get(k, 0) / n_s for k in all_k]
            probs_r = [counts_r.get(k, 0) / n_r for k in all_k]
            
            # Width of bars
            width = 0.35
            
            # Plot Ref on left, Synth on right
            ax.bar(all_k - width/2, probs_r, width, label=self.ref_label, color='orange', alpha=0.8)
            ax.bar(all_k + width/2, probs_s, width, label=self.synth_label, color='blue', alpha=0.8)
            
            ax.set_xlabel("Degree")
            ax.set_ylabel("Probability")
            
            # Ensure integer ticks centered between the bars
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            # Add some padding to x-axis
            # Ensure a minimum visible range so bars don't look excessively wide if degree variance is low
            deg_span = max_deg - min_deg
            if deg_span < 5:
                center = (min_deg + max_deg) / 2
                ax.set_xlim(center - 3, center + 3)
            else:
                ax.set_xlim(min_deg - 1, max_deg + 1)

        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
        if created_figure:
            plt.tight_layout()
            plt.show()

    def print_level_metrics(self):
        """
        Iterates through voltage levels found in both graphs and prints metrics.
        Does NOT plot.
        """
        levels_s = set(nx.get_node_attributes(self.synth_graph, 'voltage_level').values())
        levels_r = set(nx.get_node_attributes(self.ref_graph, 'voltage_level').values())
        
        common_levels = sorted(list(levels_s.intersection(levels_r)))
        
        if not common_levels:
            print("No common 'voltage_level' attributes found between graphs.")
            return

        for level in common_levels:
            # Extract Subgraphs
            nodes_s = [n for n, d in self.synth_graph.nodes(data=True) if d.get('voltage_level') == level]
            nodes_r = [n for n, d in self.ref_graph.nodes(data=True) if d.get('voltage_level') == level]
            
            sub_s = self.synth_graph.subgraph(nodes_s)
            sub_r = self.ref_graph.subgraph(nodes_r)
            
            # Print Comparison
            self.print_metric_comparison(sub_s, sub_r, title=f"LEVEL {level} COMPARISON")

    def plot_level_topology_comparison(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plots side-by-side bar comparisons for topology metrics per voltage level:
        Nodes, Edges, Diameter, Avg Path Length, Avg Clustering.
        """
        levels_s = set(nx.get_node_attributes(self.synth_graph, 'voltage_level').values())
        levels_r = set(nx.get_node_attributes(self.ref_graph, 'voltage_level').values())
        common_levels = sorted(list(levels_s.intersection(levels_r)))
        
        if not common_levels:
            print("No common levels found for topology comparison.")
            return

        # Prepare Data Containers
        metrics_map = {
            'Nodes': {'r': [], 's': []},
            'Edges': {'r': [], 's': []},
            'Diameter': {'r': [], 's': []},
            'Avg Distance': {'r': [], 's': []},
            'Clustering Coeff': {'r': [], 's': []}
        }
        
        # Mapping for better y-axis labels
        y_labels = {
            'Nodes': 'Count',
            'Edges': 'Count',
            'Diameter': 'Hop Count',
            'Avg Distance': 'Avg Hops',
            'Clustering Coeff': 'Coefficient'
        }
        
        for level in common_levels:
            # Extract Subgraphs
            n_s = [n for n, d in self.synth_graph.nodes(data=True) if d.get('voltage_level') == level]
            n_r = [n for n, d in self.ref_graph.nodes(data=True) if d.get('voltage_level') == level]
            
            sub_s = self.synth_graph.subgraph(n_s)
            sub_r = self.ref_graph.subgraph(n_r)
            
            # Analyze (Using GridAnalyzer for consistency)
            an_s = GridAnalyzer(sub_s)
            an_r = GridAnalyzer(sub_r)
            
            s_basic = an_s.get_basic_stats()
            s_path = an_s.get_path_metrics()
            s_clust = an_s.get_clustering_metrics()
            
            r_basic = an_r.get_basic_stats()
            r_path = an_r.get_path_metrics()
            r_clust = an_r.get_clustering_metrics()
            
            # Store
            metrics_map['Nodes']['s'].append(s_basic['num_nodes'])
            metrics_map['Nodes']['r'].append(r_basic['num_nodes'])
            
            metrics_map['Edges']['s'].append(s_basic['num_edges'])
            metrics_map['Edges']['r'].append(r_basic['num_edges'])
            
            metrics_map['Diameter']['s'].append(s_path['diameter'])
            metrics_map['Diameter']['r'].append(r_path['diameter'])
            
            metrics_map['Avg Distance']['s'].append(s_path['avg_path_length'])
            metrics_map['Avg Distance']['r'].append(r_path['avg_path_length'])
            
            metrics_map['Clustering Coeff']['s'].append(s_clust['avg_clustering_coef'])
            metrics_map['Clustering Coeff']['r'].append(r_clust['avg_clustering_coef'])

        # Plotting
        metric_names = list(metrics_map.keys())
        n_metrics = len(metric_names)
        
        # Grid layout (e.g., 2 rows x 3 cols)
        cols = 3
        rows = math.ceil(n_metrics / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        x = np.arange(len(common_levels))
        width = 0.35
        
        for i, metric in enumerate(metric_names):
            ax = axes[i]
            data = metrics_map[metric]
            
            ax.bar(x - width/2, data['r'], width, label=self.ref_label, color='orange', alpha=0.8)
            ax.bar(x + width/2, data['s'], width, label=self.synth_label, color='blue', alpha=0.8)
            
            ax.set_title(metric)
            ax.set_xticks(x)
            # Rotate x-labels slightly for readability
            ax.set_xticklabels([f"Level {l}" for l in common_levels], rotation=45, ha='right')
            # Add descriptive y-label
            ax.set_ylabel(y_labels.get(metric, "Value"))
            
            # Set x limits to prevent single bars from becoming too wide
            ax.set_xlim(-1, len(common_levels))
            
            if i == 0:
                ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.3, axis='y')
        
        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    def plot_all_levels_comparison(self, log_scale: bool = True):
        """
        Plots degree comparison for all common voltage levels in a single figure.
        """
        levels_s = set(nx.get_node_attributes(self.synth_graph, 'voltage_level').values())
        levels_r = set(nx.get_node_attributes(self.ref_graph, 'voltage_level').values())
        common_levels = sorted(list(levels_s.intersection(levels_r)))
        
        if not common_levels:
            print("No common levels to plot.")
            return

        n_levels = len(common_levels)
        cols = 3 if n_levels > 3 else n_levels
        rows = math.ceil(n_levels / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        
        if n_levels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        print(f"Plotting Combined Comparison Figure for {n_levels} Levels (Log Scale: {log_scale})...")
        
        for i, level in enumerate(common_levels):
            nodes_s = [n for n, d in self.synth_graph.nodes(data=True) if d.get('voltage_level') == level]
            nodes_r = [n for n, d in self.ref_graph.nodes(data=True) if d.get('voltage_level') == level]
            
            sub_s = self.synth_graph.subgraph(nodes_s)
            sub_r = self.ref_graph.subgraph(nodes_r)
            
            self.plot_degree_comparison(sub_s, sub_r, ax=axes[i], log_scale=log_scale, title=f"Level {level}")
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    def compare_degree_distributions(self, show_pvalue: bool = False) -> pd.DataFrame:
        """
        Computes Kolmogorov-Smirnov (KS) and Relative Hausdorff (RH) statistics
        between the degree distributions of the synthetic and reference graphs,
        per voltage level.  Prints and returns the results table.

        Args:
            show_pvalue: If True, include the KS p-value column. Default False.

        Returns:
            pd.DataFrame with columns Level, KS Statistic, RH Distance
            (and KS p-value if show_pvalue is True).
        """
        from scipy.stats import ks_2samp
        from scipy.spatial.distance import directed_hausdorff

        def _degree_seq(graph, nodes=None):
            if nodes is not None:
                sub = graph.subgraph(nodes)
            else:
                sub = graph
            return [d for _, d in sub.degree()]

        def _relative_hausdorff(seq_a, seq_b):
            """1-D relative Hausdorff distance between two degree sequences."""
            if not seq_a or not seq_b:
                return float('nan')
            a = np.sort(seq_a).reshape(-1, 1).astype(float)
            b = np.sort(seq_b).reshape(-1, 1).astype(float)
            h = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
            denom = max(a.max(), b.max())
            return h / denom if denom > 0 else 0.0

        levels_s = set(nx.get_node_attributes(self.synth_graph, 'voltage_level').values())
        levels_r = set(nx.get_node_attributes(self.ref_graph, 'voltage_level').values())
        common_levels = sorted(list(levels_s.intersection(levels_r)))

        rows = []

        # Per voltage level
        for level in common_levels:
            nodes_s = [n for n, d in self.synth_graph.nodes(data=True) if d.get('voltage_level') == level]
            nodes_r = [n for n, d in self.ref_graph.nodes(data=True) if d.get('voltage_level') == level]

            deg_s = _degree_seq(self.synth_graph, nodes_s)
            deg_r = _degree_seq(self.ref_graph, nodes_r)

            if deg_s and deg_r:
                ks_stat, ks_p = ks_2samp(deg_s, deg_r)
                rh = _relative_hausdorff(deg_s, deg_r)
            else:
                ks_stat, ks_p, rh = float('nan'), float('nan'), float('nan')

            row = {
                'Level': f'Level {level}',
                'KS Statistic': ks_stat,
                'RH Distance': rh,
            }
            if show_pvalue:
                row['KS p-value'] = ks_p
            rows.append(row)

        df = pd.DataFrame(rows)

        # Print table
        print("\n" + "=" * 65)
        print("DEGREE DISTRIBUTION COMPARISON  (KS & Relative Hausdorff)")
        print("=" * 65)
        fmt = df.copy()
        fmt['KS Statistic'] = fmt['KS Statistic'].map('{:.4f}'.format)
        fmt['RH Distance'] = fmt['RH Distance'].map('{:.4f}'.format)
        if show_pvalue:
            fmt['KS p-value'] = fmt['KS p-value'].map('{:.4e}'.format)
        print(fmt.to_string(index=False))
        print("=" * 65 + "\n")

        # return df

    def run_full_comparison(self, log_scale: bool = True):
        """Runs global comparison followed by per-level comparison."""
        print(">>> Running Global Comparison")
        self.print_metric_comparison(title="GLOBAL GRAPH COMPARISON")
        self.plot_degree_comparison(title="Global Degree Comparison", log_scale=log_scale)
        
        print(">>> Running Per-Level Metric Comparison")
        self.print_level_metrics()
        
        print(">>> Plotting Per-Level Topology Metrics")
        self.plot_level_topology_comparison()

        print(">>> Plotting Per-Level Degree Distributions")
        self.plot_all_levels_comparison(log_scale=log_scale)

        print(">>> Computing Degree Distribution Statistics (KS & RH)")
        self.compare_degree_distributions()