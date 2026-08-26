"""
Draw the adversary-recovery figure for the assessment report.

Reads the JSON written by ``measure_topology_recovery.py`` and plots, for each
perturbed parameter, how the adversary's estimate moves as the perturbation
strength rises.  The reference value is drawn as a dashed line: the further the
estimate sits from it, and the more consistently it does so, the more the
parameter is protected.

Usage::

    python scripts/make_recovery_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPOSITORY_ROOT / "output" / "topology_study_results.json"
FIGURE_PATH = REPOSITORY_ROOT / "output" / "topology_recovery.png"

TRUE_COLOUR = "#B44C43"
ESTIMATE_COLOUR = "#2A5C8A"
LEVEL_COLOURS = ("#1F3B57", "#2A5C8A", "#7BA3C7", "#B8CCE0")


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    strengths = sorted(float(key) for key in results["by_strength"])
    blocks = [results["by_strength"][str(strength)] for strength in strengths]

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    # --- Panel 1: buses per level ------------------------------------------
    # Levels differ by an order of magnitude in size, so the estimate is drawn
    # relative to the truth: 1.0 means the adversary recovered it exactly.
    for level, true_count in enumerate(results["true_node_counts"]):
        ratios = [
            block["node_count"][str(level)]["mean"] / true_count for block in blocks
        ]
        errors = [
            (block["node_count"][str(level)]["mean"]
             - block["node_count"][str(level)]["ci_lo"]) / true_count
            for block in blocks
        ]
        axes[0].errorbar(
            strengths, ratios, yerr=errors, marker="o", capsize=3,
            color=LEVEL_COLOURS[level % len(LEVEL_COLOURS)],
            label=f"level {level} (true {true_count})",
        )
    axes[0].axhline(1.0, ls="--", color=TRUE_COLOUR)
    axes[0].set_title("Buses per voltage level", fontsize=10.5)
    axes[0].set_ylabel("recovered / true")
    axes[0].legend(fontsize=7.5, frameon=False)

    # --- Panel 2: degree distribution, KS distance --------------------------
    ks_values = [block["degrees_by_level"]["ks_distance"] for block in blocks]
    axes[1].plot(strengths, ks_values, marker="o", color=ESTIMATE_COLOUR)
    axes[1].axhline(ks_values[0], ls=":", color="#888888")
    axes[1].annotate("unperturbed", (strengths[-1], ks_values[0]),
                     textcoords="offset points", xytext=(-4, 6),
                     ha="right", color="#888888", fontsize=9)
    axes[1].set_title("Connection-count pattern", fontsize=10.5)
    axes[1].set_ylabel("KS distance from reference")

    # --- Panel 3: realised span, the level with the largest span ------------
    widest = str(max(
        range(len(results["true_spans"])),
        key=lambda level: results["true_spans"][level],
    ))
    span_means = [block["diameters_by_level"][widest]["mean"] for block in blocks]
    span_errors = [block["diameters_by_level"][widest]["stdev"] for block in blocks]
    true_span = results["true_spans"][int(widest)]
    axes[2].errorbar(strengths, span_means, yerr=span_errors, marker="o", capsize=4,
                     color=ESTIMATE_COLOUR)
    axes[2].axhline(true_span, ls="--", color=TRUE_COLOUR)
    axes[2].annotate(f"true = {true_span}", (strengths[-1], true_span),
                     textcoords="offset points", xytext=(-4, 6),
                     ha="right", color=TRUE_COLOUR, fontsize=9)
    axes[2].set_title(f"Network span at level {widest}", fontsize=10.5)
    axes[2].set_ylabel("realised span [hops]")

    for axis in axes:
        axis.set_xlabel("perturbation strength")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=200)
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
