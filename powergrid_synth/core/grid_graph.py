r"""
This module creates a PowerGridGraph class, inherited from nx.Graph, with some customized methods like accessing subgraph by specifying voltage_level.  
"""
import numpy as np
import networkx as nx

class PowerGridGraph(nx.Graph):
    """
    Custom NetworkX Graph for Power Grids.
    Extends nx.Graph to support subgraph extraction by voltage level.
    """
    def level(self, voltage_level: int) -> nx.Graph:
        """
        Returns a subgraph view containing only nodes at the specified voltage level.
        
        Args:
            voltage_level (int): The voltage level index (e.g., 0, 1).
            
        Returns:
            nx.Graph: A subgraph view of the grid.
        """
        nodes = [n for n, d in self.nodes(data=True) if d.get('voltage_level') == voltage_level]
        return self.subgraph(nodes)


class TransmissionGrid(PowerGridGraph):
    """Graph representation of a transmission-level power grid.

    Inherits from :class:`PowerGridGraph` and adds convenience methods
    specific to meshed, multi-voltage-level transmission networks.
    """

    @property
    def voltage_levels(self) -> list[int]:
        """Sorted list of distinct voltage levels present in the grid."""
        levels = {d.get("voltage_level") for _, d in self.nodes(data=True) if d.get("voltage_level") is not None}
        return sorted(levels)

    @property
    def n_levels(self) -> int:
        """Number of distinct voltage levels."""
        return len(self.voltage_levels)


class DistributionGrid(PowerGridGraph):
    """Graph representation of a radial distribution feeder.

    Inherits from :class:`PowerGridGraph` and adds convenience properties
    for radial-tree distribution grids annotated with hop distances,
    node types, and cable parameters (Schweitzer et al. 2017 format).
    """

    @property
    def root(self):
        """Return the root node (hop distance 0)."""
        for n, d in self.nodes(data=True):
            if d.get("h") == 0:
                return n
        return None

    @property
    def max_hop(self) -> int:
        """Maximum hop distance in the feeder."""
        hops = [d.get("h", 0) for _, d in self.nodes(data=True)]
        return max(hops) if hops else 0

    @property
    def is_radial(self) -> bool:
        """True if the graph is a connected tree."""
        return nx.is_connected(self) and nx.is_tree(self)

    def nodes_at_hop(self, h: int) -> list:
        """Return list of nodes at the given hop distance."""
        return [n for n, d in self.nodes(data=True) if d.get("h") == h]

    def nodes_by_type(self, node_type: str) -> list:
        """Return list of nodes with the given ``node_type`` attribute.

        Parameters
        ----------
        node_type : str
            One of ``'load'``, ``'injection'``, or ``'intermediate'``.
        """
        return [n for n, d in self.nodes(data=True) if d.get("node_type") == node_type]

    @property
    def total_load_mw(self) -> float:
        """Total real-power load (MW) across all load nodes."""
        return sum(d.get("P_mw", 0.0) for _, d in self.nodes(data=True) if d.get("P_mw", 0.0) > 0)

    @property
    def total_gen_mw(self) -> float:
        """Total real-power generation (MW) across all injection nodes."""
        return sum(-d.get("P_mw", 0.0) for _, d in self.nodes(data=True) if d.get("P_mw", 0.0) < 0)

    @classmethod
    def from_nx(cls, G: nx.Graph) -> "DistributionGrid":
        """Create a DistributionGrid from any NetworkX graph.

        All node, edge, and graph attributes are copied.
        """
        dg = cls()
        dg.add_nodes_from(G.nodes(data=True))
        dg.add_edges_from(G.edges(data=True))
        dg.graph.update(G.graph)
        return dg