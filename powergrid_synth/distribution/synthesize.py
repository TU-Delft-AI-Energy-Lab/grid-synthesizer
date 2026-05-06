"""
One-line distribution feeder synthesis workflow.

This module exposes a single public function
:func:`synthesize_distribution` that runs the full Schweitzer pipeline —
reference loading, parameter fitting, feeder generation, validation, and
export — without any intermediate steps.

Two operation modes are supported:

* **Mode I – reference-based** (``mode="reference"``):
  Load an existing distribution network from pandapower, pypowsybl, or a
  file in any pypowsybl-supported format, extract its statistical
  parameters, and generate structurally similar synthetic feeders.

* **Mode II – default / parametric** (``mode="default"``):
  Generate feeders using the built-in default parameters from Table III
  of Schweitzer et al. (2017), or from a user-supplied
  :class:`DistributionSynthParams` object.

Example — Mode I (reference-based, pandapower)::

    from powergrid_synth.distribution.synthesize import synthesize_distribution

    feeders = synthesize_distribution(
        mode="reference",
        reference_case="cigre_lv",
        n_feeders=5,
        n_nodes=20,
        total_load_mw=0.5,
        seed=42,
    )

Example — Mode I (reference-based, pypowsybl network)::

    import pypowsybl as pp
    net = pp.network.load("/path/to/grid.cgmes")

    feeders = synthesize_distribution(
        mode="reference",
        reference_net=net,
        n_feeders=3,
        n_nodes=25,
        total_load_mw=1.0,
        seed=42,
    )

Example — Mode II (default parameters)::

    feeders = synthesize_distribution(
        mode="default",
        n_feeders=10,
        n_nodes=30,
        total_load_mw=0.8,
        seed=42,
    )
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

import networkx as nx

from .distribution_analysis import fit_params_from_feeders
from .distribution_converter import (
    feeder_summary,
    pandapower_to_feeders,
    pypowsybl_to_feeders,
)
from .distribution_params import DistributionSynthParams
from .distribution_synthesis import SchweetzerFeederGenerator
from .distribution_validation import (
    compare_feeders,
    compute_emergent_properties,
    validate_tree,
)


# ---------------------------------------------------------------------------
# pandapower built-in distribution cases
# ---------------------------------------------------------------------------

_PP_DISTRIBUTION_CASES = {
    "cigre_lv": "create_cigre_network_lv",
    "cigre_mv": "create_cigre_network_mv",
}


def _get_pandapower_dist_case(name: str):
    """Load a pandapower built-in distribution network by short name."""
    try:
        import pandapower.networks as pn
    except ImportError as exc:
        raise ImportError(
            "synthesize_distribution() in reference mode requires "
            "pandapower.  Install it with: pip install pandapower"
        ) from exc

    factory_name = _PP_DISTRIBUTION_CASES.get(name)
    if factory_name is not None:
        return getattr(pn, factory_name)()

    # Also try any pandapower factory whose name matches
    if hasattr(pn, name):
        return getattr(pn, name)()

    return None


def _is_pypowsybl_network(obj) -> bool:
    """Check if *obj* is a pypowsybl Network without a hard import."""
    return type(obj).__module__.startswith("pypowsybl")


def synthesize_distribution(
    *,
    mode: str,
    # --- Mode I (reference) ---
    reference_case: Optional[str] = None,
    reference_net=None,
    reference_file: Optional[str] = None,
    # --- Mode II (default / parametric) ---
    params: Optional[DistributionSynthParams] = None,
    # --- Feeder generation settings ---
    n_feeders: int = 1,
    n_nodes: int = 20,
    total_load_mw: float = 0.5,
    total_gen_mw: float = 0.0,
    v_nom_kv: float = 10.0,
    assign_cable_types: bool = True,
    assign_cable_lengths: bool = True,
    # --- Common parameters ---
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    output_name: str = "synthetic_feeder",
    export_formats: Sequence[str] = ("json",),
) -> List[nx.Graph]:
    """Run the full Schweitzer distribution synthesis pipeline.

    Parameters
    ----------
    mode : ``"reference"`` or ``"default"``
        Selects the operation mode.

        * ``"reference"`` — fit synthesis parameters from an existing
          distribution network (pandapower, pypowsybl, or file), then
          generate synthetic feeders with the fitted parameters.
        * ``"default"`` — use default Table III parameters (or a
          user-supplied :class:`DistributionSynthParams`).

    reference_case : str, optional
        Short name of a built-in distribution network.  Currently
        supported: ``"cigre_lv"``, ``"cigre_mv"``, or any pandapower
        factory function name.  Ignored when *reference_net* or
        *reference_file* is given.
    reference_net : pandapowerNet or pypowsybl.network.Network, optional
        A pre-loaded network object.  Takes precedence over
        *reference_case* and *reference_file*.
    reference_file : str, optional
        Path to a grid file in any pypowsybl-supported format.

    params : DistributionSynthParams, optional
        Custom parameters for ``mode="default"``.  If ``None``, the
        built-in defaults from Schweitzer Table III are used.

    n_feeders : int
        Number of synthetic feeders to generate (default 1).
    n_nodes : int
        Number of nodes per feeder (default 20).
    total_load_mw : float
        Total load in MW per feeder (default 0.5).
    total_gen_mw : float
        Total generation in MW per feeder (default 0.0).
    v_nom_kv : float
        Nominal voltage in kV (default 10.0).
    assign_cable_types : bool
        Run Step 4 — cable type assignment (default True).
    assign_cable_lengths : bool
        Run Step 5 — cable length / impedance assignment (default True).

    seed : int, optional
        Random seed for reproducibility.
    output_dir : str, optional
        If given, export feeders to this directory.
    output_name : str
        Base filename (without extension) for exported files.
    export_formats : sequence of str
        Export formats when *output_dir* is set.  Supported:
        ``"json"``, ``"excel"``, ``"sqlite"``, ``"pickle"``.

    Returns
    -------
    list[nx.Graph]
        Generated synthetic feeders as annotated NetworkX graphs.

    Raises
    ------
    ValueError
        If *mode* is invalid or required arguments are missing.
    """
    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    mode = mode.lower().strip()
    if mode not in ("reference", "default"):
        raise ValueError(
            f"mode must be 'reference' or 'default', got {mode!r}"
        )

    if mode == "reference":
        if (
            reference_net is None
            and reference_file is None
            and reference_case is None
        ):
            raise ValueError(
                "Mode 'reference' requires reference_net, "
                "reference_file, or reference_case."
            )

    # ------------------------------------------------------------------
    # 1. Obtain synthesis parameters
    # ------------------------------------------------------------------
    ref_feeders = None
    if mode == "reference":
        ref_feeders = _load_reference_feeders(
            reference_net=reference_net,
            reference_file=reference_file,
            reference_case=reference_case,
        )
        fitted_params = fit_params_from_feeders(ref_feeders)
        summary = feeder_summary(ref_feeders)
        n_ref = len(ref_feeders)
        avg_nodes = sum(s["n_nodes"] for s in summary) / n_ref if n_ref else 0
        print(
            f"[1] Loaded {n_ref} reference feeder(s): "
            f"avg {avg_nodes:.0f} nodes."
        )
    else:
        fitted_params = params if params is not None else DistributionSynthParams()
        print("[1] Using default / user-supplied parameters.")

    # ------------------------------------------------------------------
    # 2. Generate synthetic feeders
    # ------------------------------------------------------------------
    gen = SchweetzerFeederGenerator(params=fitted_params, seed=seed)
    synthetic_feeders: List[nx.Graph] = []

    for i in range(n_feeders):
        feeder = gen.generate_feeder(
            n_nodes=n_nodes,
            total_load_mw=total_load_mw,
            total_gen_mw=total_gen_mw,
            v_nom_kv=v_nom_kv,
            assign_cable_types=assign_cable_types,
            assign_cable_lengths=assign_cable_lengths,
        )
        synthetic_feeders.append(feeder)

    syn_summary = feeder_summary(synthetic_feeders)
    avg_syn_nodes = (
        sum(s["n_nodes"] for s in syn_summary) / n_feeders
        if n_feeders
        else 0
    )
    print(
        f"[2] Generated {n_feeders} synthetic feeder(s): "
        f"avg {avg_syn_nodes:.0f} nodes, "
        f"{n_nodes} requested."
    )

    # ------------------------------------------------------------------
    # 3. Validate & compare (if reference available)
    # ------------------------------------------------------------------
    for i, feeder in enumerate(synthetic_feeders):
        issues = validate_tree(feeder)
        if issues:
            print(f"  [!] Feeder {i}: {issues}")

    if ref_feeders and len(ref_feeders) > 0:
        kl = compare_feeders(synthetic_feeders[0], ref_feeders[0])
        parts = ", ".join(f"{k}={v:.3f}" for k, v in kl.items())
        print(f"[3] KL divergence (feeder 0 vs ref 0): {parts}")
    else:
        props = compute_emergent_properties(synthetic_feeders[0])
        print(
            f"[3] Feeder 0 properties: "
            f"{props['n_nodes']} nodes, "
            f"{props['n_edges']} edges, "
            f"max_hop={props['max_hop']}, "
            f"load={props['total_load_mw']:.3f} MW"
        )

    # ------------------------------------------------------------------
    # 4. Export (optional)
    # ------------------------------------------------------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _export_feeders(
            synthetic_feeders,
            output_dir=output_dir,
            output_name=output_name,
            export_formats=export_formats,
        )
        print(f"[4] Exported to {output_dir}/")

    print("[Done]")
    return synthetic_feeders


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_reference_feeders(
    *,
    reference_net=None,
    reference_file: Optional[str] = None,
    reference_case: Optional[str] = None,
) -> List[nx.Graph]:
    """Resolve a reference source to a list of feeder graphs.

    Priority: *reference_net* > *reference_file* > *reference_case*.
    """
    # --- 1. Pre-loaded network object ------------------------------------
    if reference_net is not None:
        if _is_pypowsybl_network(reference_net):
            return pypowsybl_to_feeders(reference_net)
        else:
            return pandapower_to_feeders(reference_net)

    # --- 2. File path (pypowsybl-supported format) -----------------------
    if reference_file is not None:
        try:
            import pypowsybl as pp
        except ImportError as exc:
            raise ImportError(
                "Loading from file requires pypowsybl.  "
                "Install it with: pip install pypowsybl"
            ) from exc
        net = pp.network.load(reference_file)
        return pypowsybl_to_feeders(net)

    # --- 3. Built-in case name -------------------------------------------
    assert reference_case is not None

    # Try pandapower built-in distribution cases
    pp_net = _get_pandapower_dist_case(reference_case)
    if pp_net is not None:
        return pandapower_to_feeders(pp_net)

    # Try as file path (last resort)
    if os.path.exists(reference_case):
        try:
            import pypowsybl as pp
        except ImportError as exc:
            raise ImportError(
                "Loading from file requires pypowsybl.  "
                "Install it with: pip install pypowsybl"
            ) from exc
        net = pp.network.load(reference_case)
        return pypowsybl_to_feeders(net)

    raise ValueError(
        f"Unknown reference case {reference_case!r}. "
        f"Available built-ins: {sorted(_PP_DISTRIBUTION_CASES)}. "
        f"Or provide a network object via reference_net or a file "
        f"path via reference_file."
    )


def _export_feeders(
    feeders: List[nx.Graph],
    *,
    output_dir: str,
    output_name: str,
    export_formats: Sequence[str],
) -> None:
    """Export feeders using the GridExporter."""
    from ..core.exporter import GridExporter

    for i, feeder in enumerate(feeders):
        suffix = f"_{i}" if len(feeders) > 1 else ""
        name = f"{output_name}{suffix}"

        exporter = GridExporter(feeder)
        for fmt in export_formats:
            fmt_lower = fmt.lower().strip()
            filepath = os.path.join(output_dir, name)
            if fmt_lower == "json":
                exporter.to_json(filepath + ".json")
            elif fmt_lower == "excel":
                exporter.to_excel(filepath + ".xlsx")
            elif fmt_lower == "sqlite":
                exporter.to_sqlite(filepath + ".sqlite")
            elif fmt_lower == "pickle":
                exporter.to_pickle(filepath + ".p")
            else:
                print(
                    f"  [!] Unknown format {fmt!r} for distribution, "
                    f"skipping. Available: json, excel, sqlite, pickle"
                )
