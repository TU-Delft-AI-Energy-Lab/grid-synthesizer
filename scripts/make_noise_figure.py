"""
Draw the noise mechanism for the assessment report.

Two panels:

* the Laplace density at several strengths, showing what "stronger
  perturbation" means concretely;
* the round trip for a probability --- logit to the real line, displace, then
  sigmoid back --- showing why the transform is needed at all.

The right panel is the argument for the transform: adding noise directly to a
probability pushes mass outside [0, 1] and clipping piles it at the
boundaries, which is both invalid and detectable. Working on the transformed
scale cannot produce an invalid value.

Usage::

    python scripts/make_noise_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = REPOSITORY_ROOT / "output" / "noise_mechanism.png"

# Strengths shown in the left panel, with k = 1 so that b = s.
STRENGTHS = (0.25, 0.5, 1.0)
STRENGTH_COLOURS = ("#9DBBD6", "#5B8FBF", "#2F5F8F")

# The probability carried through the round trip in the right panel.
EXAMPLE_PROBABILITY = 0.13

ACCENT = "#B44C43"
DRAW_COUNT = 30000
RANDOM_SEED = 3


def laplace_density(x: np.ndarray, scale: float) -> np.ndarray:
    """Zero-centred Laplace density at *scale*."""
    return np.exp(-np.abs(x) / scale) / (2.0 * scale)


def draw_left_panel(axis) -> None:
    """Laplace density at several strengths."""
    grid = np.linspace(-4.0, 4.0, 1200)
    for strength, colour in zip(STRENGTHS, STRENGTH_COLOURS):
        axis.plot(
            grid, laplace_density(grid, strength), color=colour, lw=2.0,
            label=f"$s = {strength}$",
        )
    axis.axvline(0.0, color="#888888", lw=0.8, ls=":")
    axis.set_xlabel(r"displacement $\eta$ (transformed space)")
    axis.set_ylabel("density")
    axis.set_title(
        r"$\eta \sim \mathrm{Laplace}(0,\,b)$,  $b = s\,k$", fontsize=10
    )
    axis.legend(frameon=False, fontsize=9)
    axis.grid(alpha=0.25)
    axis.set_ylim(bottom=0)


def draw_right_panel(axis) -> None:
    """Where a probability lands after a round trip through logit space."""
    generator = np.random.default_rng(RANDOM_SEED)
    logit_value = np.log(EXAMPLE_PROBABILITY / (1.0 - EXAMPLE_PROBABILITY))

    for strength, colour in zip(STRENGTHS, STRENGTH_COLOURS):
        displaced = logit_value + generator.laplace(0.0, strength, DRAW_COUNT)
        recovered = 1.0 / (1.0 + np.exp(-displaced))
        axis.hist(
            recovered, bins=140, range=(0.0, 1.0), density=True,
            histtype="step", color=colour, lw=1.8, label=f"$s = {strength}$",
        )

    axis.axvline(
        EXAMPLE_PROBABILITY, color=ACCENT, lw=1.6, ls="--",
        label=f"reference $p = {EXAMPLE_PROBABILITY}$",
    )
    axis.set_xlabel(r"perturbed probability $p'$")
    axis.set_ylabel("density")
    axis.set_title(
        r"$p' = \sigma\!\left(\mathrm{logit}(p) + \eta\right)$", fontsize=10
    )
    axis.set_xlim(0.0, 1.0)
    axis.legend(frameon=False, fontsize=9)
    axis.grid(alpha=0.25)


def main() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    draw_left_panel(axes[0])
    draw_right_panel(axes[1])
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=170, bbox_inches="tight")
    plt.close(figure)

    # Report the spread so the caption can quote it rather than assert it.
    generator = np.random.default_rng(RANDOM_SEED)
    logit_value = np.log(EXAMPLE_PROBABILITY / (1.0 - EXAMPLE_PROBABILITY))
    print(f"wrote {FIGURE_PATH}")
    for strength in STRENGTHS:
        recovered = 1.0 / (
            1.0
            + np.exp(-(logit_value + generator.laplace(0.0, strength, DRAW_COUNT)))
        )
        low, high = np.percentile(recovered, [5, 95])
        print(
            f"  s={strength}: 90% of draws land in "
            f"[{low:.3f}, {high:.3f}]  (reference {EXAMPLE_PROBABILITY})"
        )


if __name__ == "__main__":
    main()
