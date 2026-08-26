"""
Simulate a pooling adversary against the perturbed topology parameters.

The adversary holds a batch of synthetic grids generated from one reference
under one perturbation configuration, pools them, and estimates the reference
value.  Pooling averages away the synthesiser's own randomness, so whatever
survives pooling is what the reference actually leaks.  Perturbation is
supposed to set a floor that pooling cannot cross, because the displacement is
identical in every grid of the batch.

Three parameters are measured, each with the estimator an adversary would
naturally use:

``node_count``
    Buses per voltage level.  Estimator: the pooled mean and its 95\\% interval,
    compared against the true count.

``degrees_by_level``
    Connection-count pattern.  Estimator: the Kolmogorov--Smirnov distance
    between the pooled degree distribution and the reference's.

``diameters_by_level``
    Network span.  The span is a *target* handed to the topology generator
    rather than a value written to the output, so the estimator is the span
    actually realised in the generated grid, against the generator's own
    run-to-run spread.

The numbers in the non-reverse-engineerability report are produced by this
script.

Usage::

    python scripts/measure_topology_recovery.py
    python scripts/measure_topology_recovery.py --n-grids 50 --strengths 0 1.0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_STRENGTHS = (0.0, 0.5, 1.0, 2.0)

#: The parameters this study perturbs together, as one published release would.
PERTURBED_PARAMETERS = ("node_count", "degrees_by_level", "diameters_by_level")


def build_config(strength: float, batch_seed: int):
    """Perturbation configuration for one batch, or ``None`` for the baseline.

    The seed is fixed for the whole batch, which is what a single published
    release looks like: every grid carries the same displacement, so pooling
    within the batch cannot average it away.

    Args:
        strength: Perturbation strength; zero means no perturbation at all.
        batch_seed: Seed shared by every grid in the batch.

    Returns:
        A configuration mapping, or ``None`` when *strength* is zero.
    """
    if strength <= 0.0:
        return None
    return {
        "strength": strength,
        "seed": batch_seed,
        "parameters": {name: {"mode": "perturb"} for name in PERTURBED_PARAMETERS},
    }


def generate_pool(reference_case: str, strength: float, n_grids: int, output_dir: Path):
    """Generate one batch of synthetic grids.

    Args:
        reference_case: Name of the built-in reference network.
        strength: Perturbation strength for the batch.
        n_grids: Pool size.
        output_dir: Scratch directory for the exporter.

    Returns:
        List of generated graphs.
    """
    from powergrid_synth.transmission.synthesize import synthesize

    config = build_config(strength, batch_seed=4242)
    graphs = []
    for grid_index in range(n_grids):
        # The synthesiser is chatty; the progress this script reports is its own.
        with contextlib.redirect_stdout(io.StringIO()):
            graphs.append(
                synthesize(
                    mode="reference",
                    reference_case=reference_case,
                    seed=grid_index,
                    perturbation_config=config,
                    output_dir=str(output_dir),
                    output_name=f"pool_{strength}_{grid_index}",
                    export_formats=(),
                )
            )
        print(f"    grid {grid_index + 1}/{n_grids}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return graphs


def measure(graphs, true_params):
    """Run every adversary estimator against one pooled batch.

    Args:
        graphs: The batch of synthetic grids.
        true_params: Reference parameters, from the topology extractor.

    Returns:
        Mapping of parameter name to that parameter's recovery result.
    """
    from powergrid_synth.privacy.recovery import (
        recover_degree_dist_full,
        recover_node_count,
    )
    from powergrid_synth.transmission.synthesize import _measure_spans_by_level

    true_degrees = true_params["degrees_by_level"]
    true_spans = true_params["diameters_by_level"]
    n_levels = len(true_degrees)

    node_counts = recover_node_count(graphs, n_levels)
    for level in range(n_levels):
        node_counts[level]["true"] = len(true_degrees[level])
        node_counts[level].pop("counts", None)

    spans_per_grid = [_measure_spans_by_level(graph) for graph in graphs]
    spans = {}
    for level in range(n_levels):
        values = [row[level] for row in spans_per_grid if level < len(row)]
        spans[level] = {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "true": int(true_spans[level]),
        }

    return {
        "node_count": node_counts,
        "degrees_by_level": {"ks_distance": recover_degree_dist_full(graphs, true_degrees)},
        "diameters_by_level": spans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-case", default="case118")
    parser.add_argument("--n-grids", type=int, default=30)
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=list(DEFAULT_STRENGTHS)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "output" / "topology_study_results.json",
    )
    parser.add_argument("--scratch", type=Path, default=Path("output") / "topology_study")
    arguments = parser.parse_args()

    arguments.scratch.mkdir(parents=True, exist_ok=True)

    import pandapower.networks as pandapower_networks

    from powergrid_synth.core.data_format_converter import pandapower_to_nx
    from powergrid_synth.core.input_extractor import extract_topology_params_from_graph

    reference_graph = pandapower_to_nx(
        getattr(pandapower_networks, arguments.reference_case)()
    )
    true_params = extract_topology_params_from_graph(reference_graph)

    results = {
        "reference_case": arguments.reference_case,
        "n_grids": arguments.n_grids,
        "perturbed_parameters": list(PERTURBED_PARAMETERS),
        "true_node_counts": [len(level) for level in true_params["degrees_by_level"]],
        "true_spans": list(true_params["diameters_by_level"]),
        "by_strength": {},
    }
    for strength in arguments.strengths:
        print(f"  strength {strength}:")
        graphs = generate_pool(
            arguments.reference_case, strength, arguments.n_grids, arguments.scratch
        )
        results["by_strength"][str(strength)] = measure(graphs, true_params)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(
        f"\n=== pooling adversary, {arguments.reference_case}, "
        f"{arguments.n_grids} grids per configuration ==="
    )
    print(f"true buses per level: {results['true_node_counts']}")
    print(f"true spans per level: {results['true_spans']}")
    for strength, block in results["by_strength"].items():
        print(f"\n  strength {strength}")
        for level, stats in block["node_count"].items():
            print(
                f"    buses L{level}: recovered {stats['mean']:.1f} "
                f"[{stats['ci_lo']:.1f}, {stats['ci_hi']:.1f}]  true {stats['true']}"
            )
        print(f"    degree KS distance: {block['degrees_by_level']['ks_distance']:.4f}")
        for level, stats in block["diameters_by_level"].items():
            print(
                f"    span L{level}: {stats['mean']:.1f} +/- {stats['stdev']:.1f}  "
                f"true {stats['true']}"
            )
    print(f"\nWritten to {arguments.output}")


if __name__ == "__main__":
    main()
