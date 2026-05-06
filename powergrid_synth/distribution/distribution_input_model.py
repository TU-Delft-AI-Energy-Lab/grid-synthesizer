"""
Input model for the Schweitzer feeder generator.

Uses a 3-D kernel density estimate (KDE) over (N_nodes, P_load, P_gen) to
sample realistic combinations of feeder size and loading, as described in
Section III-A of Schweitzer et al. (2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde


@dataclass
class FeederInputSample:
    """A single draw from the 3-D input model."""
    n_nodes: int
    total_load_mw: float
    total_gen_mw: float


class DistributionInputModel:
    """3-D KDE input model for feeder size and loading.

    The model is fitted on reference feeders and then sampled to produce
    realistic ``(n_nodes, total_load, total_gen)`` triples.

    Parameters
    ----------
    bw_method : str or float, optional
        Bandwidth method passed to :class:`scipy.stats.gaussian_kde`.
    seed : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        bw_method: Optional[str | float] = None,
        seed: Optional[int] = None,
    ):
        self.bw_method = bw_method
        self.rng = np.random.default_rng(seed)
        self._kde: Optional[gaussian_kde] = None
        self._data: Optional[np.ndarray] = None

    def fit(self, feeders: Sequence[nx.Graph]) -> "DistributionInputModel":
        """Fit the KDE from a collection of reference feeder graphs.

        Each graph must have node attributes ``P_mw`` (positive for load,
        negative for generation).

        Parameters
        ----------
        feeders : sequence of nx.Graph
            Reference feeder graphs.

        Returns
        -------
        self
        """
        rows: List[Tuple[float, float, float]] = []
        for G in feeders:
            n = G.number_of_nodes()
            total_load = sum(
                G.nodes[nd].get("P_mw", 0.0)
                for nd in G.nodes
                if G.nodes[nd].get("P_mw", 0.0) > 0
            )
            total_gen = sum(
                abs(G.nodes[nd].get("P_mw", 0.0))
                for nd in G.nodes
                if G.nodes[nd].get("P_mw", 0.0) < 0
            )
            rows.append((float(n), total_load, total_gen))

        data = np.array(rows).T  # shape (3, n_feeders)
        self._data = data
        self._kde = self._fit_kde(data)
        return self

    def fit_from_arrays(
        self,
        n_nodes: Sequence[int],
        total_load: Sequence[float],
        total_gen: Sequence[float],
    ) -> "DistributionInputModel":
        """Fit directly from arrays (no graph objects needed).

        Parameters
        ----------
        n_nodes : array-like of int
        total_load : array-like of float
        total_gen : array-like of float

        Returns
        -------
        self
        """
        data = np.array([
            np.asarray(n_nodes, dtype=float),
            np.asarray(total_load, dtype=float),
            np.asarray(total_gen, dtype=float),
        ])
        self._data = data
        self._kde = self._fit_kde(data)
        return self

    def _fit_kde(self, data: np.ndarray) -> gaussian_kde:
        """Fit KDE with regularization for near-singular covariance."""
        try:
            return gaussian_kde(data, bw_method=self.bw_method)
        except np.linalg.LinAlgError:
            # Add small jitter to break degeneracy
            scale = np.std(data, axis=1, keepdims=True)
            scale = np.where(scale < 1e-10, 1.0, scale)
            jitter = self.rng.normal(0, 1e-6, size=data.shape) * scale
            return gaussian_kde(data + jitter, bw_method=self.bw_method)

    def sample(self, n_samples: int = 1) -> List[FeederInputSample]:
        """Draw samples from the fitted input model.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        list of FeederInputSample
        """
        if self._kde is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        samples = self._kde.resample(n_samples, seed=self.rng)
        results = []
        for i in range(n_samples):
            n = max(3, int(round(samples[0, i])))  # minimum 3 nodes
            load = max(0.0, float(samples[1, i]))
            gen = max(0.0, float(samples[2, i]))
            results.append(FeederInputSample(n, load, gen))
        return results

    def pdf(self, n_nodes: float, total_load: float, total_gen: float) -> float:
        """Evaluate the KDE density at a given point.

        Parameters
        ----------
        n_nodes, total_load, total_gen : float
            Coordinates in the 3-D input space.

        Returns
        -------
        float
            Estimated probability density.
        """
        if self._kde is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        result = self._kde(np.array([[n_nodes], [total_load], [total_gen]]))
        return float(result[0])
