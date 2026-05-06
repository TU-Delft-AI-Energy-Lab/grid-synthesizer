r"""
Given the size `n` and the average degree `ave_k`, This module returns the degree distribution with optimized distribution parameters for 

* `dgln`: discrete generalized log-normal --- the number $n_d$ of degree $d$ nodes
    
    .. math:: n_d \propto \exp\Big(-\bigg(\frac{\log d}{\alpha}\bigg)^\beta\Big)
    
    for some parameters $\alpha, \beta$. 

* `dpl`: discrete power law --- a scale-free network where the degree distribution follows a power law

    .. math:: n_d \propto d^{-\gamma}
    
    for some parameter $\gamma$. 

This optimizer is based on `Kolda et al. (2014) <https://arxiv.org/abs/1302.6636>`_, where users specify target values for average degree and/or maximum degree. See :doc:`Topology Generation</theory/topology_generation>` on how to set them.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Optional

class DegreeDistributionOptimizer:
    r"""
    Find parameters for ideal degree distributions matching target statistics.

    Supports two distribution families:

    * ``dgln`` (discrete generalized log-normal): :math:`n_d \propto \exp\!\big(-(\log d / \alpha)^\beta\big)`
    * ``dpl`` (discrete power law): :math:`n_d \propto d^{-\gamma}`

    The optimizer minimizes a penalty that combines a max-degree probability
    bound with a squared error on the target average degree, following
    `Kolda et al. (2014) <https://arxiv.org/abs/1302.6636>`_ (FEASTPACK).

    Ported from MATLAB ``degdist_param_search.m`` (Sandia National Labs).

    Parameters
    ----------
    verbose : bool, optional
        If True, print optimization progress. Default is False.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _dgln_pdf(self, n: int, alpha: float, beta: float) -> np.ndarray:
        r"""
        Compute the discrete generalized log-normal (DGLN) PDF.

        .. math:: P(x) \propto \exp\!\Big(-\Big(\frac{\log x}{\alpha}\Big)^\beta\Big), \quad x=1,\dots,n

        Parameters
        ----------
        n : int
            Maximum degree (support is 1..n).
        alpha : float
            Scale parameter (> 0).
        beta : float
            Shape parameter (> 0).

        Returns
        -------
        numpy.ndarray
            Normalized probability vector of length *n*.
        """
        # Avoid division by zero or invalid log domain issues during optimization
        if alpha <= 0 or beta <= 0:
            return np.zeros(n)
            
        x = np.arange(1, n + 1)
        # Calculate unnormalized probs
        # We use np.abs to handle potential negative params proposed by optimizer before bounds check
        try:
            p = np.exp(-((np.log(x)) / alpha) ** beta)
        except RuntimeWarning:
             return np.zeros(n)

        # Normalize
        total = np.sum(p)
        if total > 0:
            p = p / total
        else:
            # Fallback for bad parameters
            p = np.zeros(n)
            p[0] = 1.0 
            
        return p

    def _dpl_pdf(self, n: int, gamma: float) -> np.ndarray:
        r"""
        Compute the discrete power-law (DPL) PDF.

        .. math:: P(x) \propto x^{-\gamma}, \quad x=1,\dots,n

        Parameters
        ----------
        n : int
            Maximum degree (support is 1..n).
        gamma : float
            Power-law exponent (> 0).

        Returns
        -------
        numpy.ndarray
            Normalized probability vector of length *n*.
        """
        x = np.arange(1, n + 1)
        p = x ** (-gamma)
        p = p / np.sum(p)
        return p

    def _objective_func(self, params, dist_type: str, max_deg: int, 
                        target_avg: float, prob_bound: float) -> float:
        r"""
        Objective function for the distribution parameter search.

        The score combines two terms:

        1. **Max-degree bound penalty**: exponential penalty if
           :math:`P(d_{\max}) > \text{prob\_bound}`.
        2. **Average-degree error**: :math:`(\bar{d}_{\text{current}} - \bar{d}_{\text{target}})^2`.

        Parameters
        ----------
        params : array-like
            Distribution parameters: ``(alpha, beta)`` for DGLN or ``(gamma,)`` for DPL.
        dist_type : str
            ``'dgln'`` or ``'dpl'``.
        max_deg : int
            Maximum degree in the support.
        target_avg : float
            Target average degree.
        prob_bound : float
            Upper bound on the probability at ``max_deg``.

        Returns
        -------
        float
            Penalty score (lower is better).
        """
        if dist_type == 'dgln':
            alpha, beta = params
            # Hard constraint penalties for solver
            if alpha <= 0 or beta <= 0: return 1e6
            p = self._dgln_pdf(max_deg, alpha, beta)
        elif dist_type == 'dpl':
            gamma = params[0]
            if gamma <= 0: return 1e6
            p = self._dpl_pdf(max_deg, gamma)
        else:
            raise ValueError(f"Unknown type {dist_type}")

        # 1. Check Max Degree Bound Penalty
        # We want P(max_deg) < prob_bound
        p_max = p[-1]
        if p_max > prob_bound:
            # Penalty grows exponentially if bound is violated
            y1 = (np.exp(1 + p_max - prob_bound))**2 - 1
        else:
            y1 = 0.0

        # 2. Check Average Degree Error
        # Expected average degree = sum(x * p(x))
        x = np.arange(1, max_deg + 1)
        current_avg = np.sum(x * p)
        y2 = (current_avg - target_avg)**2

        score = y1 + y2
        
        if self.verbose:
            print(f"Params: {params}, Score: {score:.6f}, Avg: {current_avg:.4f}")
            
        return score

    def optimize(self, 
                 target_avg: float, 
                 max_deg: int, 
                 dist_type: str = 'dgln', 
                 prob_bound: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Find distribution parameters matching the target average degree.

        Uses Nelder-Mead optimization to minimize :meth:`_objective_func`.
        See `Kolda et al. (2014) <https://arxiv.org/abs/1302.6636>`_ for details.

        Parameters
        ----------
        target_avg : float
            Target average degree :math:`\bar{d}`.
        max_deg : int
            Maximum degree :math:`d_{\max}` (defines the PDF support ``1..max_deg``).
        dist_type : str, optional
            ``'dgln'`` for generalized log-normal or ``'dpl'`` for power law.
            Default is ``'dgln'``.
        prob_bound : float, optional
            Upper bound on :math:`P(d_{\max})`. Default is ``1e-10``.

        Returns
        -------
        best_params : numpy.ndarray
            Optimized parameters: ``(alpha, beta)`` for DGLN or ``(gamma,)`` for DPL.
        final_pdf : numpy.ndarray
            Normalized PDF evaluated at degrees ``1..max_deg``.
        """
        if dist_type == 'dgln':
            initial_guess = [2.0, 2.0]
        elif dist_type == 'dpl':
            initial_guess = [2.0]
        else:
            raise ValueError("Type must be 'dgln' or 'dpl'")

        # Run optimization (Nelder-Mead is typically robust for this)
        result = minimize(
            self._objective_func,
            initial_guess,
            args=(dist_type, max_deg, target_avg, prob_bound),
            method='Nelder-Mead',
            options={'xatol': 1e-4, 'fatol': 1e-4}
        )

        best_params = result.x
        
        # Generate final PDF
        if dist_type == 'dgln':
            final_pdf = self._dgln_pdf(max_deg, best_params[0], best_params[1])
        else:
            final_pdf = self._dpl_pdf(max_deg, best_params[0])
            
        if self.verbose:
            print(f"Optimization Success: {result.success}")
            print(f"Found Params: {best_params}")
            
        return best_params, final_pdf