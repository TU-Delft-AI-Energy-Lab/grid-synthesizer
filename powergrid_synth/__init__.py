"""
PowerGridSynth Package
======================

The main package for PowerGridSynth. 

This package contains submodules for generating, analyzing, and exporting 
synthetic power grids, organized into three subpackages:

- ``core``: shared utilities (analysis, comparison, visualization, etc.)
- ``transmission``: CLC-based transmission grid synthesis pipeline
- ``distribution``: Schweitzer-based distribution feeder synthesis pipeline
"""
# --- Core (shared) ----------------------------------------------------------
from .core.grid_graph import PowerGridGraph, TransmissionGrid, DistributionGrid
from .core.analysis import GridAnalyzer
from .core.comparison import GraphComparator
from .core.hierarchical_analysis import HierarchicalAnalyzer
from .core.input_extractor import extract_topology_params_from_graph
from .core.reference_data import get_reference_stats
from .core.visualization import GridVisualizer

# --- Transmission pipeline ---------------------------------------------------
from .transmission.generator import PowerGridGenerator
from .transmission.input_configurator import InputConfigurator
from .transmission.bus_type_allocator import BusTypeAllocator
from .transmission.capacity_allocator import CapacityAllocator
from .transmission.load_allocator import LoadAllocator
from .transmission.generation_dispatcher import GenerationDispatcher
from .transmission.transmission import TransmissionLineAllocator


# ---------------------------------------------------------------------------
# Lazy imports for modules that require optional dependencies
# (pandapower, pypowsybl, lightsim2grid, numba, pypowsybl-jupyter).
# The real import is deferred until the name is actually used, so that
# users who only need the core synthesis pipeline are never blocked by
# missing optional packages.
# ---------------------------------------------------------------------------

def __getattr__(name):
    _LAZY = {
        "synthesize": (".transmission.synthesize", "synthesize"),
        "synthesize_distribution": (".distribution.synthesize", "synthesize_distribution"),
        "GridExporter": (".core.exporter", "GridExporter"),
        "pandapower_to_nx": (".core.data_format_converter", "pandapower_to_nx"),
        "nx_to_pandapower": (".core.data_format_converter", "nx_to_pandapower"),
        "pandapower_to_pypowsybl": (
            ".core.data_format_converter",
            "pandapower_to_pypowsybl",
        ),
        "pypowsybl_to_nx": (
            ".core.data_format_converter",
            "pypowsybl_to_nx",
        ),
        "load_grid": (
            ".core.data_format_converter",
            "load_grid",
        ),
    }

    if name in _LAZY:
        module_path, attr = _LAZY[name]
        _EXTRA_DEPS = {
            "synthesize": "pandapower",
            "synthesize_distribution": "pandapower",
            "GridExporter": "pandapower",
            "pandapower_to_nx": "pandapower",
            "nx_to_pandapower": "pandapower",
            "pandapower_to_pypowsybl": "pypowsybl",
            "pypowsybl_to_nx": "pypowsybl",
            "load_grid": "pypowsybl",
        }
        try:
            import importlib
            mod = importlib.import_module(module_path, __name__)
            obj = getattr(mod, attr)
            # Cache in the module namespace so __getattr__ is only called once
            globals()[name] = obj
            return obj
        except ImportError as exc:
            pkg = _EXTRA_DEPS.get(name, "the required optional package")
            raise ImportError(
                f"{name!r} requires {pkg}. "
                f"Install it with: pip install powergrid_synth[export]"
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core (always available)
    "PowerGridGenerator",
    "InputConfigurator",
    "BusTypeAllocator",
    "CapacityAllocator",
    "LoadAllocator",
    "GenerationDispatcher",
    "TransmissionLineAllocator",
    "GridVisualizer",
    "get_reference_stats",
    "GraphComparator",
    "PowerGridGraph",
    "TransmissionGrid",
    "DistributionGrid",
    "HierarchicalAnalyzer",
    "extract_topology_params_from_graph",
    # Optional (lazy-loaded, require pandapower / pypowsybl)
    "synthesize",
    "synthesize_distribution",
    "GridExporter",
    "pandapower_to_nx",
    "nx_to_pandapower",
    "pandapower_to_pypowsybl",
    "pypowsybl_to_nx",
    "load_grid",
]

__version__ = "0.1.1"