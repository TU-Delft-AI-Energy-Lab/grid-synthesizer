"""
Draw a small synthetic bus-branch model for the assessment report.

Illustrates what the synthesiser produces: buses grouped by voltage level,
lines within a level, transformers between levels, and the role each bus is
assigned --- generator, load or connection.

Kept deliberately small (a few dozen buses) so individual elements stay
legible; a realistic grid drawn this way would be an unreadable hairball.

Usage::

    python scripts/make_structure_figure.py
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = REPOSITORY_ROOT / "output" / "synthetic_structure.png"

# Vertical distance between voltage levels, in layout units.
LEVEL_HEIGHT = 2.6

# Half-height of one level's band. Kept well under LEVEL_HEIGHT so the bands
# stay visually separate and transformers read as crossing between them.
BAND_HALF_HEIGHT = 0.42

# Colour per voltage level, muted so the role markers dominate.
LEVEL_COLOURS = ["#2F5F8F", "#5B8FBF", "#9DBBD6"]

LEVEL_SPECS = [
    {"n": 6, "avg_k": 2.6, "diam": 3, "dist_type": "poisson"},
    {"n": 9, "avg_k": 2.5, "diam": 4, "dist_type": "poisson"},
    {"n": 12, "avg_k": 2.4, "diam": 4, "dist_type": "poisson"},
]
CONNECTION_SPECS = {
    (0, 1): {"type": "k-stars", "c": 0.30, "gamma": 4.15},
    (1, 2): {"type": "k-stars", "c": 0.30, "gamma": 4.15},
}

# Seed for the within-level layout, fixed so the figure is reproducible.
LAYOUT_SEED = 4


def build_small_grid(seed: int = 12):
    """Synthesise a small grid in fully synthetic mode.

    Fully synthetic mode is used rather than reference mode so the figure
    depends on no reference network and anyone can regenerate it.

    Args:
        seed: Seed for reproducibility.

    Returns:
        The generated grid graph.
    """
    from powergrid_synth.transmission.synthesize import synthesize

    with contextlib.redirect_stdout(io.StringIO()):
        return synthesize(
            mode="synthetic",
            level_specs=LEVEL_SPECS,
            connection_specs=CONNECTION_SPECS,
            seed=seed,
            output_dir=str(REPOSITORY_ROOT / "output" / "figure"),
            export_formats=[],
        )


def _nodes_by_level(graph) -> dict:
    grouped: dict = {}
    for node, data in graph.nodes(data=True):
        grouped.setdefault(int(data.get("voltage_level", 0)), []).append(node)
    return grouped


def _layout_one_level(subgraph) -> dict:
    """Lay out one voltage level, packing its components side by side.

    A level's subgraph is often disconnected --- some buses reach the rest of
    the grid only through a transformer. A force-directed layout pushes
    disconnected components arbitrarily far apart, which wastes the canvas and
    squashes everything else, so components are laid out separately and then
    packed left to right.

    Args:
        subgraph: The buses and lines of one voltage level.

    Returns:
        Mapping of node to a local (x, y) position, centred on the origin.
    """
    local: dict = {}
    cursor = 0.0
    for component in sorted(nx.connected_components(subgraph), key=len, reverse=True):
        piece = subgraph.subgraph(component)
        if piece.number_of_edges():
            coordinates = nx.spring_layout(
                piece, seed=LAYOUT_SEED, iterations=400, k=1.4
            )
        else:
            coordinates = {node: np.array([0.0, 0.0]) for node in piece}

        values = np.array(list(coordinates.values()))
        values -= values.mean(axis=0)
        width = max(np.ptp(values[:, 0]), 0.6)
        for node, position in coordinates.items():
            local[node] = np.array(
                [position[0] - values[:, 0].mean() + cursor, position[1]]
            )
        # Leave a clear gap so packed components do not read as connected.
        cursor += width + 1.0

    values = np.array(list(local.values()))
    values_mean = values.mean(axis=0)
    return {node: position - values_mean for node, position in local.items()}


def banded_positions(graph) -> dict:
    """Lay each voltage level out as its own mesh, in a horizontal band.

    A force-directed layout within a level preserves the meshed structure that
    a single row would hide. Bands are then flattened to a common height so
    the voltage hierarchy stays readable, and centred on a common axis so
    transformers run roughly vertically.

    Args:
        graph: The grid graph.

    Returns:
        Mapping of node to (x, y) position.
    """
    grouped = _nodes_by_level(graph)
    positions: dict = {}

    for level in sorted(grouped):
        local = _layout_one_level(graph.subgraph(grouped[level]))
        coordinates = np.array(list(local.values()))
        spread_x = max(np.ptp(coordinates[:, 0]), 0.6)
        spread_y = max(np.ptp(coordinates[:, 1]), 0.6)
        width = 1.5 * np.sqrt(len(coordinates))

        for node, (x, y) in local.items():
            positions[node] = np.array(
                [
                    x / spread_x * width,
                    y / spread_y * (2 * BAND_HALF_HEIGHT) - level * LEVEL_HEIGHT,
                ]
            )

    return {node: tuple(value) for node, value in positions.items()}


def draw(graph, positions) -> None:
    """Render the grid to the report figure path."""
    figure, axis = plt.subplots(figsize=(9.2, 6.2))

    transformer_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("type") == "transformer"
        or graph.nodes[u].get("voltage_level") != graph.nodes[v].get("voltage_level")
    ]
    transformer_set = {tuple(sorted(e)) for e in transformer_edges}
    line_edges = [e for e in graph.edges() if tuple(sorted(e)) not in transformer_set]

    nx.draw_networkx_edges(
        graph, positions, edgelist=line_edges,
        edge_color="#4A4A4A", width=1.3, ax=axis,
    )
    nx.draw_networkx_edges(
        graph, positions, edgelist=transformer_edges,
        edge_color="#B44C43", width=1.5, style="dashed", alpha=0.9, ax=axis,
    )

    grouped = _nodes_by_level(graph)
    for level, nodes in sorted(grouped.items()):
        colour = LEVEL_COLOURS[level % len(LEVEL_COLOURS)]
        nx.draw_networkx_nodes(
            graph, positions, nodelist=nodes, node_color=colour,
            node_size=175, edgecolors="white", linewidths=1.0, ax=axis,
        )
        axis.text(
            min(positions[n][0] for n in nodes) - 1.1,
            -level * LEVEL_HEIGHT,
            f"level {level}",
            fontsize=9.5, color=colour, va="center", ha="right", fontweight="bold",
        )

    # A bus carries exactly one role, so the marker sits just above the bus
    # rather than ringing it. The offset is a fraction of the drawn width, so
    # it stays attached whatever the layout's extent turns out to be.
    all_x = [position[0] for position in positions.values()]
    unit = max(max(all_x) - min(all_x), 1.0) * 0.016
    role_styles = {
        "Gen": ("^", "#2E7D46", 40, (0.0, 1.5 * unit)),
        "Load": ("v", "#B44C43", 40, (0.0, 1.5 * unit)),
    }
    for role, (marker, colour, size, (dx, dy)) in role_styles.items():
        points = [
            (positions[node][0] + dx, positions[node][1] + dy)
            for node, data in graph.nodes(data=True)
            if data.get("bus_type") == role
        ]
        if points:
            xs, ys = zip(*points)
            axis.scatter(xs, ys, marker=marker, c=colour, s=size, zorder=4,
                         linewidths=0.4, edgecolors="white")

    handles = [
        plt.Line2D([], [], color="#4A4A4A", lw=1.3, label="line (within level)"),
        plt.Line2D([], [], color="#B44C43", lw=1.5, ls="--",
                   label="transformer (between levels)"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=LEVEL_COLOURS[1],
                   markersize=9, label="bus"),
        plt.Line2D([], [], marker="^", color="w", markerfacecolor="#2E7D46",
                   markersize=9, label="generator"),
        plt.Line2D([], [], marker="v", color="w", markerfacecolor="#B44C43",
                   markersize=9, label="load"),
    ]
    axis.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
                frameon=False, bbox_to_anchor=(0.5, -0.11))

    axis.set_axis_off()
    axis.margins(0.08)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    graph = build_small_grid()
    draw(graph, banded_positions(graph))

    n_transformers = sum(
        1
        for u, v, d in graph.edges(data=True)
        if d.get("type") == "transformer"
        or graph.nodes[u].get("voltage_level") != graph.nodes[v].get("voltage_level")
    )
    print(f"wrote {FIGURE_PATH}")
    print(
        f"  {graph.number_of_nodes()} buses, "
        f"{graph.number_of_edges() - n_transformers} lines, "
        f"{n_transformers} transformers"
    )


if __name__ == "__main__":
    main()
