r"""
Grid Analysis Module
====================

This module provides tools for analyzing the topological properties of power grid graphs.
It includes the :class:`GridAnalyzer` for calculating metrics like degree distribution,
path lengths, and clustering coefficients.
"""


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

class GridAnalyzer:
    """
    A class to perform topological analysis on power grid graphs.
    Can be used for both synthetic and real-world grids.
    
    Args:
        graph (nx.Graph): The networkx graph object representing the power grid.
    """
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Returns fundamental counts (nodes, edges, density).
        
        Returns: 
            Dict[str, Any]: A dictionary containing 'num_nodes', 'num_edges', and 'density'.
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph)
        }

    def get_path_metrics(self) -> Dict[str, Any]:
        """
        Calculates path-based metrics: Diameter and Average Shortest Path Length.
        """
        if nx.is_connected(self.graph):
            G_sub = self.graph
            is_connected = True
        else:
            # Get largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            G_sub = self.graph.subgraph(largest_cc)
            is_connected = False
        
        try:
            diameter = nx.diameter(G_sub)
            avg_path_len = nx.average_shortest_path_length(G_sub)
        except Exception as e:
            # Fallback for empty graphs or other edge cases
            diameter = float('nan')
            avg_path_len = float('nan')

        return {
            "is_connected": is_connected,
            "lcc_size": len(G_sub),
            "diameter": diameter,
            "avg_path_length": avg_path_len
        }

    def get_clustering_metrics(self) -> Dict[str, float]:
        """Calculates global and average local clustering coefficients."""
        return {
            "avg_clustering_coef": nx.average_clustering(self.graph),
            "transitivity": nx.transitivity(self.graph) # Global clustering
        }

    def analyze(self):
        """Prints a comprehensive text summary of the grid."""
        stats = self.get_basic_stats()
        path_stats = self.get_path_metrics()
        clust_stats = self.get_clustering_metrics()

        print("\n=== Power Grid Topological Analysis ===")
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Edges: {stats['num_edges']}")
        print(f"Density: {stats['density']:.6f}")
        
        conn_str = "Yes" if path_stats['is_connected'] else f"No (Metrics based on LCC of size {path_stats['lcc_size']})"
        print(f"Connected: {conn_str}")
        print(f"Diameter: {path_stats['diameter']}")
        print(f"Avg Shortest Path Length: {path_stats['avg_path_length']:.4f}")
        print(f"Avg Local Clustering Coeff: {clust_stats['avg_clustering_coef']:.4f}")
        print("=======================================\n")

    def plot_degree_distribution(self, ax: Optional[plt.Axes] = None, 
                               log_scale: bool = True, 
                               figsize: tuple = (6, 4),
                               title: str = "Degree Distribution"):
        """
        Plots the degree distribution.
        
        Args:
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            log_scale: If True, plots log-log. If False, plots linear histogram.
            figsize: Size of figure if creating a new one.
            title: Title of the plot.
        """
        degrees = [d for n, d in self.graph.degree()]
        
        if not degrees:
            print("Graph has no nodes/degrees to plot.")
            return

        created_figure = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_figure = True
        
        if log_scale:
            # Log-Log Plot
            degree_counts = np.bincount(degrees)
            # Filter out zero counts for log plot
            vals = np.nonzero(degree_counts)[0]
            counts = degree_counts[vals]
            
            if len(vals) > 0:
                ax.loglog(vals, counts, 'bo', markersize=5)
            ax.set_xlabel("Degree (log)")
            ax.set_ylabel("Count (log)")
        else:
            # Linear Histogram
            ax.hist(degrees, bins=range(min(degrees), max(degrees) + 2), 
                 color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel("Degree")
            ax.set_ylabel("Count")
            
        ax.set_title(title)
        ax.grid(True, which="both", ls="--", alpha=0.5)

        if created_figure:
            plt.tight_layout()
            plt.show()