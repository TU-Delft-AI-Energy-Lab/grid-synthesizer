r"""
This module performs an analysis of the generated synthetic grid topology for each subgraph, as well as for the entire global graph. 
"""
import networkx as nx
import matplotlib.pyplot as plt
import math
from .analysis import GridAnalyzer

class HierarchicalAnalyzer:
    """
    A wrapper class that manages analysis for multi-level power grids.
    It can perform a global analysis and then iterate through each voltage 
    level to perform subgraph analysis.
    """
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def run_global_analysis(self):
        """Runs the GridAnalyzer on the entire graph."""
        print("\n" + "="*40)
        print("GLOBAL GRID ANALYSIS")
        print("="*40)
        
        analyzer = GridAnalyzer(self.graph)
        analyzer.analyze()
        
        # Plot Global (Default Linear)
        print("Plotting Global Degree Distribution...")
        analyzer.plot_degree_distribution(log_scale=True)

    def run_level_analysis(self):
        """
        Detects all voltage levels in the graph, extracts the subgraph for each,
        and runs the GridAnalyzer (stats only) on those subgraphs.
        """
        levels = sorted(list(set(nx.get_node_attributes(self.graph, 'voltage_level').values())))

        if not levels:
            print("No 'voltage_level' attributes found in graph nodes. Skipping level analysis.")
            return

        for level in levels:
            print("\n" + "="*40)
            print(f"ANALYSIS FOR VOLTAGE LEVEL {level}")
            print("="*40)
            
            level_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('voltage_level') == level]
            subgraph = self.graph.subgraph(level_nodes)
            
            level_analyzer = GridAnalyzer(subgraph)
            level_analyzer.analyze()

    def plot_all_levels(self, log_scale: bool = True):
        """
        Plots degree distributions for all voltage levels in a single figure (subplots).
        
        Args:
            log_scale: If True, uses log-log scale. If False, uses linear scale.
        """
        levels = sorted(list(set(nx.get_node_attributes(self.graph, 'voltage_level').values())))
        if not levels:
            print("No levels to plot.")
            return

        n_levels = len(levels)
        cols = 3 if n_levels > 3 else n_levels
        rows = math.ceil(n_levels / cols)
        
        # Create a big figure
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        
        # Flatten axes array for easy iteration, handle case of single subplot
        if n_levels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        print(f"Plotting Combined Figure for {n_levels} Levels (Log Scale: {log_scale})...")
        
        for i, level in enumerate(levels):
            level_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('voltage_level') == level]
            subgraph = self.graph.subgraph(level_nodes)
            
            analyzer = GridAnalyzer(subgraph)
            # Pass the specific axis to the analyzer
            analyzer.plot_degree_distribution(ax=axes[i], log_scale=log_scale, title=f"Level {level}")
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    def run_full_report(self, log_scale: bool = True):
        """
        Convenience method to run analysis and plotting.
        Args:
            log_scale: Whether to plot the multi-level figure in log scale.
        """
        self.run_global_analysis()
        self.run_level_analysis()
        self.plot_all_levels(log_scale=log_scale)