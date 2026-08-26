"""
Complete power-grid synthesis workflow.

This module exposes a single public function :func:`synthesize` that runs
the full CLC pipeline — topology generation, bus-type assignment,
generation/load allocation, dispatch, transmission-line parameterization,
and export — without any intermediate visualisation.

Two operation modes are supported:

* **Mode I – reference-based** (``mode="reference"``):
  Load an existing power grid from pandapower, pypowsybl, or a file in
  any pypowsybl-supported format (CGMES, XIIDM, MATPOWER, PSS/E, etc.),
  extract its topological parameters, and generate a structurally similar
  synthetic clone.

* **Mode II – synthetic** (``mode="synthetic"``):
  Build a grid entirely from user-specified voltage-level specifications
  (node counts, average degrees, diameters, degree distributions) and
  inter-level connection parameters.

Example — Mode I (reference-based, pandapower)::

    from powergrid_synth.synthesize import synthesize

    synthesize(
        mode="reference",
        reference_case="case118",
        seed=42,
        output_dir="output",
        export_formats=["json", "cgmes"],
    )

Example — Mode I (reference-based, pypowsybl built-in)::

    synthesize(
        mode="reference",
        reference_case="ieee118",
        seed=42,
        output_dir="output",
        export_formats=["json", "cgmes"],
    )

Example — Mode I (reference-based, file path)::

    synthesize(
        mode="reference",
        reference_file="path/to/grid.xiidm",
        seed=42,
        output_dir="output",
        export_formats=["json", "cgmes"],
    )

Example — Mode II (fully synthetic)::

    from powergrid_synth.synthesize import synthesize

    synthesize(
        mode="synthetic",
        level_specs=[
            {"n": 50,  "avg_k": 3.5, "diam": 10, "dist_type": "dgln"},
            {"n": 150, "avg_k": 2.5, "diam": 15, "dist_type": "dpl"},
            {"n": 300, "avg_k": 2.0, "diam": 20, "dist_type": "poisson"},
        ],
        connection_specs={
            (0, 1): {"type": "k-stars", "c": 0.174, "gamma": 4.15},
            (1, 2): {"type": "k-stars", "c": 0.15,  "gamma": 4.15},
        },
        seed=42,
        output_dir="output",
        export_formats=["json", "matpower"],
    )
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx

from .bus_type_allocator import BusTypeAllocator
from .capacity_allocator import CapacityAllocator
from .generation_dispatcher import GenerationDispatcher
from .generator import PowerGridGenerator
from .input_configurator import InputConfigurator
from ..core.input_extractor import extract_topology_params_from_graph
from .load_allocator import LoadAllocator
from .transmission import TransmissionLineAllocator
from ..privacy import engine as _perturb_engine
from ..privacy import settings as _perturb_settings
from ..privacy.registry import Domain, Mode, get_descriptor
from ..privacy.report import build_report
from ..privacy.rng import make_rng, resolve_seed


def _get_pandapower_cases() -> Dict[str, object]:
    try:
        import pandapower.networks as pn
    except ImportError as exc:
        raise ImportError(
            "synthesize() in reference mode requires pandapower. "
            "Install it with: pip install powergrid_synth[export]"
        ) from exc

    return {name: getattr(pn, name) for name in dir(pn) if name.startswith("case")}


def _get_pypowsybl_builtins() -> Dict[str, object]:
    """Return a mapping of short names to pypowsybl built-in network factories."""
    try:
        import pypowsybl as ppl
    except ImportError:
        return {}
    builtins = {}
    for name in dir(ppl.network):
        if name.startswith("create_ieee"):
            # e.g. create_ieee14 -> "ieee14"
            short = name.replace("create_", "")
            builtins[short] = getattr(ppl.network, name)
    return builtins


def _is_pypowsybl_network(obj) -> bool:
    """Check if *obj* is a pypowsybl Network without requiring the import."""
    return type(obj).__module__.startswith("pypowsybl")

# Maps user-facing format names to (GridExporter method, default extension).
_EXPORT_DISPATCH = {
    "json":     ("to_json",      ".json"),
    "excel":    ("to_excel",     ".xlsx"),
    "sqlite":   ("to_sqlite",    ".sqlite"),
    "pickle":   ("to_pickle",    ".p"),
    "xiidm":    ("to_pypowsybl", ".xiidm"),
    "cgmes":    ("to_cgmes",     "_cgmes"),
    "matpower": ("to_matpower",  ""),
    "psse":     ("to_psse",      ""),
}


def synthesize(
    *,
    mode: str,
    # --- Mode I (reference) ---
    reference_case: Optional[str] = None,
    reference_net=None,
    reference_file: Optional[str] = None,
    # --- Mode II (synthetic) ---
    level_specs: Optional[List[Dict]] = None,
    connection_specs: Optional[Dict[Tuple[int, int], Dict]] = None,
    # --- Common parameters ---
    seed: Optional[int] = None,
    keep_lcc: bool = True,
    entropy_model: int = 0,
    bus_type_ratio: Optional[List[float]] = None,
    ref_sys_id: int = 1,
    loading_level: str = "H",
    refine_topology: bool = False,
    base_kv_map: Optional[Dict[int, float]] = None,
    perturbation_config=None,
    output_dir: str = "output",
    output_name: str = "synthetic_grid",
    export_formats: Sequence[str] = ("json",),
) -> nx.Graph:
    """Run the full CLC synthesis pipeline and export the result.

    Parameters
    ----------
    mode : ``"reference"`` or ``"synthetic"``
        Selects the operation mode.

        * ``"reference"`` — extract topology parameters from an existing
          pandapower network (Mode I).
        * ``"synthetic"`` — generate topology from user-provided level /
          connection specs (Mode II).

    reference_case : str, optional
        Name of a built-in network.  Supports both pandapower case names
        (e.g. ``"case118"``) and pypowsybl IEEE names (e.g.
        ``"ieee14"``, ``"ieee118"``).  Ignored if *reference_net* or
        *reference_file* is given.
    reference_net : pandapowerNet or pypowsybl.network.Network, optional
        A pre-loaded network object (pandapower or pypowsybl).  Takes
        precedence over *reference_case* and *reference_file*.
    reference_file : str, optional
        Path to a grid file in any pypowsybl-supported format (CGMES,
        XIIDM, MATPOWER, IEEE-CDF, PSS/E, UCTE, etc.).  Takes
        precedence over *reference_case*.

    level_specs : list of dict, optional
        Voltage-level specifications for Mode II.  Each dict must have
        keys ``"n"`` (int), ``"avg_k"`` (float), ``"diam"`` (int), and
        ``"dist_type"`` (``"dgln"`` | ``"dpl"`` | ``"poisson"``).
        Optionally ``"max_k"`` (int).
    connection_specs : dict, optional
        Transformer connection specs for Mode II.  Maps
        ``(level_i, level_j)`` tuples to dicts with ``"type"``
        (``"k-stars"`` | ``"simple"``) and parameters ``"c"``, ``"gamma"``.

    seed : int, optional
        Random seed for reproducibility.
    keep_lcc : bool
        If ``True`` (default), keep only the largest connected component
        after topology generation.
    entropy_model : int
        Bus-type entropy model — 0 (standard) or 1 (weighted).
    bus_type_ratio : list of float, optional
        Target ``[Gen%, Load%, Conn%]`` ratios.  If ``None``, default
        ratios based on network size are used.
    ref_sys_id : int
        Reference system for statistical tables (0–3).
    loading_level : str
        Load level — ``"D"`` (deterministic), ``"L"`` (light),
        ``"M"`` (medium), ``"H"`` (heavy, default).
    refine_topology : bool
        If ``True``, the transmission allocator may add/remove edges to
        improve DCPF convergence.
    base_kv_map : dict, optional
        Custom ``{level_index: kV}`` mapping.  If ``None`` and
        ``mode="reference"``, the mapping is extracted from the reference
        grid.
    perturbation_config : str, path, dict or PerturbationSettings, optional
        Controls how much of the reference grid reaches the synthetic one.
        Each registered input is handled in one of three modes — ``raw``
        (passed through untouched), ``perturb`` (displaced by calibrated
        noise) or ``off`` (never read; a public fallback is used instead).
        ``None`` keeps every parameter at its registry default, which is
        ``raw``, so the pipeline behaves exactly as it did before this
        argument existed.  See :mod:`powergrid_synth.privacy.settings` for
        the configuration format, and ``perturbation.example.toml`` for a
        worked file.  Only applies in ``mode="reference"``; a synthetic run
        reads no reference values and so has nothing to protect.
    output_dir : str
        Directory for exported files (created if needed).
    output_name : str
        Base filename (without extension) for exported files.
    export_formats : sequence of str
        One or more format names to export.  Supported:
        ``"json"``, ``"excel"``, ``"sqlite"``, ``"pickle"``,
        ``"xiidm"``, ``"cgmes"``, ``"matpower"``, ``"psse"``.

    Returns
    -------
    nx.Graph
        The fully parameterised synthetic grid (a
        :class:`~powergrid_synth.grid_graph.PowerGridGraph`).

    Raises
    ------
    ValueError
        If *mode* is not ``"reference"`` or ``"synthetic"``, or if
        required arguments for the chosen mode are missing.
    """
    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    mode = mode.lower().strip()
    if mode not in ("reference", "synthetic"):
        raise ValueError(
            f"mode must be 'reference' or 'synthetic', got {mode!r}"
        )

    if mode == "reference":
        if reference_net is None and reference_file is None and reference_case is None:
            raise ValueError(
                "Mode 'reference' requires reference_net, "
                "reference_file, or reference_case."
            )
    else:
        if level_specs is None or connection_specs is None:
            raise ValueError(
                "Mode 'synthetic' requires both level_specs and "
                "connection_specs."
            )

    # ------------------------------------------------------------------
    # 1. Obtain topology parameters
    # ------------------------------------------------------------------
    # Perturbation state — initialised here so it is in scope after the mode
    # branch.  The node-count offset is applied after topology generation and
    # LCC filtering (step [2.5]) and needs the settings regardless of mode.
    _perturb = _perturb_settings.coerce(perturbation_config)
    _resolved_seed = resolve_seed(_perturb.seed if _perturb.seed is not None else seed)
    _perturb_details: Dict[str, Dict] = {}
    _perturb_fallbacks: Dict[str, bool] = {}

    def _stream_rng(stream_name: str):
        """RNG for one noise stream, keyed by the resolved seed."""
        return make_rng(_resolved_seed, stream_name)

    if mode == "reference":
        graph_ref = _load_reference(
            reference_net=reference_net,
            reference_file=reference_file,
            reference_case=reference_case,
        )
        params = extract_topology_params_from_graph(graph_ref)

        # Use the reference grid's base_kv_map unless the user overrides.
        if base_kv_map is None:
            base_kv_map = graph_ref.graph.get("base_kv_map")

        if _perturb.mode_of('base_kv_map') is Mode.OFF:
            # Never publish the reference grid's real nominal voltages; the
            # exporter falls back to its generic level map.
            base_kv_map = None
            graph_ref.graph.pop('base_kv_map', None)

        if _perturb.should_perturb('diameters_by_level'):
            _span_descriptor = get_descriptor('diameters_by_level')
            _span_strength = _perturb.strength_of('diameters_by_level')
            _spans_before = list(params.get('diameters_by_level', []))
            _spans_after = []
            for _level, _span in enumerate(_spans_before):
                if _span <= 0:
                    # A span of zero means the level holds a single bus or
                    # none.  Displacing it would invent an extent the level
                    # does not have, so structural zeros are preserved.
                    _spans_after.append(_span)
                    continue
                _spans_after.append(
                    _perturb_engine.perturb_integer_value(
                        _span_descriptor,
                        int(_span),
                        strength=_span_strength,
                        rng=_stream_rng(f'topology.span[level={_level}]'),
                    )
                )
            params['diameters_by_level'] = _spans_after
            _perturb_details['diameters_by_level'] = {
                'target_before': _spans_before,
                'target_after': _spans_after,
            }

        if _perturb.should_perturb('degrees_by_level'):
            _deg_descriptor = get_descriptor('degrees_by_level')
            _strength = _perturb.strength_of('degrees_by_level')
            _new_degrees, _n_fallbacks = [], 0
            for _level, _sequence in enumerate(params.get('degrees_by_level', [])):
                _seq, _attempts, _fell_back = _perturb_engine.perturb_degree_sequence(
                    _deg_descriptor,
                    _sequence,
                    strength=_strength,
                    rng=_stream_rng(f'topology.degree_dist[level={_level}]'),
                )
                _new_degrees.append(_seq)
                _n_fallbacks += int(_fell_back)
            params['degrees_by_level'] = _new_degrees
            if _n_fallbacks:
                _perturb_fallbacks['degrees_by_level'] = True
                print(
                    f"  [1.5] degrees_by_level: {_n_fallbacks} level(s) fell back "
                    "to the unperturbed sequence (no graphical resample found)."
                )

        print(
            f"[1] Loaded reference grid: "
            f"{graph_ref.number_of_nodes()} nodes, "
            f"{graph_ref.number_of_edges()} edges."
        )

    else:  # mode == "synthetic"
        configurator = InputConfigurator(seed=seed)
        params = configurator.create_params(level_specs, connection_specs)
        print(f"[1] Generated synthetic input parameters for {len(level_specs)} voltage levels.")

    # ------------------------------------------------------------------
    # 2. Generate topology
    # ------------------------------------------------------------------
    gen = PowerGridGenerator(seed=seed)
    graph = gen.generate_grid(
        degrees_by_level=params["degrees_by_level"],
        diameters_by_level=params["diameters_by_level"],
        transformer_degrees=params["transformer_degrees"],
        keep_lcc=keep_lcc,
    )
    print(
        f"[2] Topology generated: "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges."
    )

    # ------------------------------------------------------------------
    # 2.5  Node-count offset, applied to the generated graph after LCC
    #      filtering so the non-linear filter cannot distort the noise.
    # ------------------------------------------------------------------
    if _perturb.should_perturb("node_count"):
        graph, _level_deltas = _perturb_engine.perturb_graph_node_count(
            graph,
            descriptor=get_descriptor("node_count"),
            strength=_perturb.strength_of("node_count"),
            rng_for_level=lambda level: _stream_rng(
                f"topology.node_count[level={level}]"
            ),
        )
        _total_delta = sum(_level_deltas.values())
        _perturb_details["node_count"] = {
            "per_level_delta": {str(k): v for k, v in _level_deltas.items()},
            "total_delta": _total_delta,
        }
        if _total_delta != 0:
            print(
                f"  [2.5] node_count: total bus delta = {_total_delta:+d} "
                f"(per level: {_level_deltas})"
            )

    # The span is a generator target, not an output value, so confirm the
    # displacement survived into the graph an adversary would observe.
    if _perturb.should_perturb("diameters_by_level"):
        _realised_spans = _measure_spans_by_level(graph)
        _span_detail = _perturb_details.setdefault("diameters_by_level", {})
        _span_detail["realised"] = _realised_spans
        _targets = _span_detail.get("target_after", [])
        _moved = any(
            target != realised
            for target, realised in zip(_span_detail.get("target_before", []), _realised_spans)
        )
        if _targets and not _moved:
            _perturb_fallbacks["diameters_by_level"] = True
            print(
                "  [2.5] diameters_by_level: the displaced span target was not "
                "realised in the generated graph; treat this parameter as "
                "unperturbed for this run."
            )

    # ------------------------------------------------------------------
    # 3. Assign bus types
    # ------------------------------------------------------------------
    allocator = BusTypeAllocator(
        graph,
        entropy_model=entropy_model,
        bus_type_ratio=bus_type_ratio,
    )
    bus_types = allocator.allocate()
    nx.set_node_attributes(graph, bus_types, name="bus_type")
    _print_bus_types(bus_types)

    # ------------------------------------------------------------------
    # 4. Allocate generation capacities
    # ------------------------------------------------------------------
    cap_alloc = CapacityAllocator(graph, ref_sys_id=ref_sys_id)
    capacities = cap_alloc.allocate()
    nx.set_node_attributes(graph, capacities, name="pg_max")
    total_cap = sum(capacities.values())
    print(f"[4] Generation capacities: total = {total_cap:.1f} MW")

    # ------------------------------------------------------------------
    # 5. Allocate loads
    # ------------------------------------------------------------------
    load_alloc = LoadAllocator(graph, ref_sys_id=ref_sys_id)
    loads = load_alloc.allocate(loading_level=loading_level)
    nx.set_node_attributes(graph, loads, name="pl")
    reactive_loads = load_alloc.allocate_reactive(loads)
    nx.set_node_attributes(graph, reactive_loads, name="ql")
    total_load = sum(loads.values())
    total_ql = sum(reactive_loads.values())
    print(
        f"[5] Loads allocated: total P = {total_load:.1f} MW, "
        f"Q = {total_ql:.1f} Mvar "
        f"({total_load / total_cap:.0%} loading)"
    )

    # ------------------------------------------------------------------
    # 6. Dispatch generation
    # ------------------------------------------------------------------
    dispatcher = GenerationDispatcher(graph, ref_sys_id=ref_sys_id)
    dispatch = dispatcher.dispatch()
    nx.set_node_attributes(graph, dispatch, name="pg")
    total_gen = sum(dispatch.values())
    print(
        f"[6] Generation dispatched: {total_gen:.1f} MW "
        f"(reserve {total_cap - total_gen:.1f} MW)"
    )

    # ------------------------------------------------------------------
    # 7. Allocate transmission lines (impedance + capacity)
    # ------------------------------------------------------------------
    trans_alloc = TransmissionLineAllocator(graph, ref_sys_id=ref_sys_id)
    line_caps = trans_alloc.allocate(refine_topology=refine_topology)
    n_lines = len(line_caps)
    avg_cap = sum(line_caps.values()) / n_lines if n_lines else 0
    print(
        f"[7] Transmission lines: {n_lines} lines, "
        f"avg capacity = {avg_cap:.1f} MVA"
    )

    # ------------------------------------------------------------------
    # 8. Export
    # ------------------------------------------------------------------
    from ..core.exporter import GridExporter

    os.makedirs(output_dir, exist_ok=True)
    exporter = GridExporter(graph, base_kv_map=base_kv_map)

    for fmt in export_formats:
        fmt_lower = fmt.lower().strip()
        if fmt_lower not in _EXPORT_DISPATCH:
            print(f"  [!] Unknown format {fmt!r}, skipping. "
                  f"Available: {sorted(_EXPORT_DISPATCH)}")
            continue

        method_name, default_ext = _EXPORT_DISPATCH[fmt_lower]
        filepath = os.path.join(output_dir, output_name + default_ext)
        method = getattr(exporter, method_name)

        if fmt_lower == "xiidm":
            method(filepath, format="XIIDM")
        else:
            method(filepath)

        print(f"  -> Exported {fmt_lower}: {filepath}")

    # ------------------------------------------------------------------
    # 9. Perturbation report — what was protected, and what was not.
    # ------------------------------------------------------------------
    _report = build_report(
        _perturb,
        Domain.TRANSMISSION,
        _resolved_seed,
        details=_perturb_details,
        fallbacks=_perturb_fallbacks,
    )
    for _line in _report.summary_lines():
        print(_line)
    if output_dir:
        _report.write_json(
            os.path.join(output_dir, output_name + "_perturbation_report.json")
        )
    graph.graph["perturbation_report"] = _report.to_dict()

    print("[8] Done.")
    return graph


def _measure_spans_by_level(graph) -> List[int]:
    """Return the realised network span of each voltage level.

    The span is the diameter of the level's largest connected component, which
    is what the topology generator is asked to target and what an observer of
    the published grid can measure.

    Args:
        graph: Generated topology whose nodes carry ``voltage_level``.

    Returns:
        Span per level, indexed by level number; ``0`` for an empty level.
    """
    nodes_by_level: Dict[int, List] = {}
    for node, data in graph.nodes(data=True):
        nodes_by_level.setdefault(int(data.get("voltage_level", 0)), []).append(node)

    spans: List[int] = []
    for level in range(max(nodes_by_level) + 1 if nodes_by_level else 0):
        nodes = nodes_by_level.get(level, [])
        if not nodes:
            spans.append(0)
            continue
        subgraph = graph.subgraph(nodes)
        if subgraph.number_of_nodes() < 2:
            spans.append(0)
            continue
        component = max(nx.connected_components(subgraph), key=len)
        spans.append(int(nx.diameter(subgraph.subgraph(component))))
    return spans


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_reference(
    *,
    reference_net=None,
    reference_file: Optional[str] = None,
    reference_case: Optional[str] = None,
) -> nx.Graph:
    """Resolve a reference grid to a NetworkX graph.

    Priority: *reference_net* > *reference_file* > *reference_case*.
    """
    # --- 1. Pre-loaded network object (pandapower or pypowsybl) ----------
    if reference_net is not None:
        if _is_pypowsybl_network(reference_net):
            from ..core.data_format_converter import pypowsybl_to_nx
            return pypowsybl_to_nx(reference_net)
        else:
            # Assume pandapower
            from ..core.data_format_converter import pandapower_to_nx
            return pandapower_to_nx(reference_net)

    # --- 2. File path (any pypowsybl-supported format) -------------------
    if reference_file is not None:
        from ..core.data_format_converter import load_grid
        return load_grid(reference_file)

    # --- 3. Built-in case name -------------------------------------------
    assert reference_case is not None

    # Try pypowsybl built-ins first (e.g. "ieee14", "ieee118")
    ppl_builtins = _get_pypowsybl_builtins()
    if reference_case in ppl_builtins:
        from ..core.data_format_converter import pypowsybl_to_nx
        net = ppl_builtins[reference_case]()
        return pypowsybl_to_nx(net)

    # Try pandapower built-ins (e.g. "case118", "case14")
    pandapower_cases = _get_pandapower_cases()
    factory = pandapower_cases.get(reference_case)
    if factory is not None:
        from ..core.data_format_converter import pandapower_to_nx
        return pandapower_to_nx(factory())

    # Try as a file path as last resort
    if os.path.exists(reference_case):
        from ..core.data_format_converter import load_grid
        return load_grid(reference_case)

    raise ValueError(
        f"Unknown reference case {reference_case!r}. "
        f"Available pandapower cases: {sorted(pandapower_cases)}. "
        f"Available pypowsybl built-ins: {sorted(ppl_builtins)}. "
        f"Or provide a file path via reference_file."
    )


def _print_bus_types(bus_types: Dict[int, str]) -> None:
    from collections import Counter
    counts = Counter(bus_types.values())
    total = sum(counts.values())
    parts = ", ".join(
        f"{t}: {counts[t]} ({counts[t] / total:.0%})"
        for t in ("Gen", "Load", "Conn")
        if t in counts
    )
    print(f"[3] Bus types assigned: {parts}")
