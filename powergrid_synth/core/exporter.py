"""
This module implements the GridExporter class, responsible for exporting
the generated synthetic grid to standard file formats.

It converts the internal NetworkX graph into pandapower and pypowsybl
network objects, then delegates to their built-in export functions.

Supported pandapower formats:
    - JSON, Excel, SQLite, Pickle

Supported pypowsybl formats:
    - CGMES, XIIDM, MATPOWER, PSS/E, UCTE, AMPL, BIIDM, JIIDM
"""
import os
import networkx as nx
from typing import Optional


class GridExporter:
    """
    Exports the generated synthetic grid to standard file formats via
    pandapower and pypowsybl built-in functions.

    Args:
        graph: The synthetic power grid as a NetworkX Graph.
        base_mva: System base apparent power in MVA.
        base_kv_map: Dictionary mapping voltage level indices to kV values
            (e.g., ``{0: 380.0, 1: 110.0, 2: 20.0}``).  If ``None``,
            a default mapping is used.

    Example::

        exporter = GridExporter(grid, base_mva=100.0,
                                base_kv_map={0: 380.0, 1: 110.0})
        exporter.to_json("output/grid.json")
        exporter.to_excel("output/grid.xlsx")
        exporter.to_pypowsybl("output/grid", format="CGMES")
    """

    def __init__(self, graph: nx.Graph, base_mva: float = 100.0,
                 base_kv_map: Optional[dict] = None):
        self.graph = graph
        self.base_mva = base_mva
        self.base_kv_map = base_kv_map

        # Lazily built network objects (created on first use)
        self._pp_net = None
        self._ppl_net = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pandapower_net(self):
        """Return a pandapower network, building it once from the graph."""
        if self._pp_net is None:
            from .data_format_converter import nx_to_pandapower

            self._pp_net = nx_to_pandapower(
                self.graph, base_mva=self.base_mva, base_kv_map=self.base_kv_map
            )
        return self._pp_net

    def _get_pypowsybl_net(self):
        """Return a pypowsybl network, building it once from pandapower."""
        if self._ppl_net is None:
            from .data_format_converter import pandapower_to_pypowsybl

            self._ppl_net = pandapower_to_pypowsybl(self._get_pandapower_net())
        return self._ppl_net

    # ------------------------------------------------------------------
    # Pandapower exports
    # ------------------------------------------------------------------

    def to_json(self, filepath: str) -> None:
        """Export the grid to a pandapower JSON file.

        Args:
            filepath: Destination file path (e.g. ``"output/grid.json"``).
        """
        import pandapower as pp
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        pp.to_json(self._get_pandapower_net(), filepath)
        print(f"-> pandapower JSON export: {filepath}")

    def to_excel(self, filepath: str, include_results: bool = True) -> None:
        """Export the grid to a pandapower Excel file.

        Args:
            filepath: Destination file path (e.g. ``"output/grid.xlsx"``).
            include_results: Whether to include power flow results if available.
        """
        import pandapower as pp
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        pp.to_excel(self._get_pandapower_net(), filepath,
                     include_results=include_results)
        print(f"-> pandapower Excel export: {filepath}")

    def to_sqlite(self, filepath: str, include_results: bool = False) -> None:
        """Export the grid to a pandapower SQLite database.

        Args:
            filepath: Destination file path (e.g. ``"output/grid.sqlite"``).
            include_results: Whether to include power flow results.
        """
        import pandapower as pp
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        pp.to_sqlite(self._get_pandapower_net(), filepath,
                      include_results=include_results)
        print(f"-> pandapower SQLite export: {filepath}")

    def to_pickle(self, filepath: str) -> None:
        """Export the grid to a pandapower pickle file.

        Args:
            filepath: Destination file path (e.g. ``"output/grid.p"``).
        """
        import pandapower as pp
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        pp.to_pickle(self._get_pandapower_net(), filepath)
        print(f"-> pandapower pickle export: {filepath}")

    # ------------------------------------------------------------------
    # Pypowsybl exports
    # ------------------------------------------------------------------

    def to_pypowsybl(self, filepath: str, format: str = "XIIDM",
                      parameters: Optional[dict] = None) -> None:
        """Export the grid using pypowsybl's built-in save.

        Supported formats: CGMES, XIIDM, MATPOWER, PSS/E, UCTE, AMPL,
        BIIDM, JIIDM.

        Args:
            filepath: Destination path (file or directory depending on format).
            format: One of the pypowsybl export format strings.
            parameters: Optional format-specific parameters.
        """
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._get_pypowsybl_net().save(filepath, format=format,
                                        parameters=parameters)
        print(f"-> pypowsybl {format} export: {filepath}")

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    def to_matpower(self, filepath: str) -> None:
        """Export to MATPOWER format via pypowsybl.

        Args:
            filepath: Destination path (e.g. ``"output/grid"``).
        """
        self.to_pypowsybl(filepath, format="MATPOWER")

    def to_cgmes(self, filepath: str, parameters: Optional[dict] = None) -> None:
        """Export to CGMES (Common Grid Model Exchange Standard) via pypowsybl.

        Args:
            filepath: Destination directory path.
            parameters: Optional CGMES-specific parameters.
        """
        self.to_pypowsybl(filepath, format="CGMES", parameters=parameters)

    def to_psse(self, filepath: str) -> None:
        """Export to PSS/E format via pypowsybl.

        Args:
            filepath: Destination path.
        """
        self.to_pypowsybl(filepath, format="PSS/E")