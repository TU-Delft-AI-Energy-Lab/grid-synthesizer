r"""
This calss assists the operation mode II in grid topology generation, where one takes as inputs 

1. `n`: the number of nodes; `ave_k`: the average node degree; and `diam`: the desired diameter for same-voltage level and the node degree distribution types

- `dgln`: discrete generalized log-normal distribution 
- `dpl`: discrete power-law distribution 

2. the transformer line specs between different-voltage levels: `type`: `k-stars`, parameters `c` and `gamma` for the transformer degree distribution.

Example
-------
.. code-block:: python
   :linenos:
   
    # Define the voltage levels (Node count, Avg Degree, Diameter, Distribution Type)
    level_specs = [
        {'n': 50,  'avg_k': 3.5, 'diam': 10, 'dist_type': 'dgln'},    # (Log-Normal)
        {'n': 150, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'dpl'},     # (Power Law)
        {'n': 300, 'avg_k': 2.0, 'diam': 20, 'dist_type': 'poisson'}  # (Poisson)
    ]
    # Define connections between levels (k-stars model)
    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15}
    }
    # Initialize Configurator
    configurator = InputConfigurator(seed=100)
    # Generating Input Parameters
    params = configurator.create_params(level_specs, connection_specs)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from .deg_dist_optimizer import DegreeDistributionOptimizer

class InputConfigurator:
    r"""
    Generate detailed input sequences for :class:`PowerGridGenerator` from
    high-level parameters.

    This is "operation mode II", where the user specifies only the number of
    nodes, average degree, diameter, and distribution type for each voltage
    level, rather than providing explicit degree sequences.  The configurator
    uses :class:`DegreeDistributionOptimizer` to fit distribution parameters
    and then samples degree sequences.

    See Section 6 of `Aksoy et al. (2018)
    <https://doi.org/10.1093/comnet/cny016>`_ for the synthetic input
    generation guidelines.

    Parameters
    ----------
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.optimizer = DegreeDistributionOptimizer(verbose=False)

    def _generate_poisson_degrees(self, n_nodes: int, avg_degree: float) -> List[int]:
        r"""
        Generate a degree sequence from a Poisson distribution.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        avg_degree : float
            Mean of the Poisson distribution :math:`\lambda = \bar{d}`.

        Returns
        -------
        list of int
            Sampled degree sequence of length *n_nodes*.
        """
        return self.rng.poisson(avg_degree, n_nodes).tolist()

    def _generate_optimized_degrees(self, n_nodes: int, avg_degree: float, 
                                    max_degree: int, dist_type: str) -> List[int]:
        r"""
        Generate a degree sequence by optimizing DGLN or DPL parameters.

        First calls :meth:`DegreeDistributionOptimizer.optimize` to find the
        distribution parameters matching ``avg_degree`` and ``max_degree``,
        then samples *n_nodes* degrees from the resulting PDF.

        Parameters
        ----------
        n_nodes : int
            Number of nodes in the same-voltage subgraph.
        avg_degree : float
            Target average degree :math:`\bar{d}`.
        max_degree : int
            Maximum degree :math:`d_{\max}` (PDF support is ``1..max_degree``).
        dist_type : str
            ``'dgln'`` for generalized log-normal or ``'dpl'`` for power law.

        Returns
        -------
        list of int
            Sampled degree sequence of length *n_nodes*.
        """
        # 1. Optimize parameters to find the ideal PDF
        _, pdf = self.optimizer.optimize(
            target_avg=avg_degree,
            max_deg=max_degree,
            dist_type=dist_type
        )
        
        # 2. Sample from the PDF
        degree_values = np.arange(1, max_degree + 1)
        samples = self.rng.choice(degree_values, size=n_nodes, p=pdf)
        return samples.tolist()

    def _generate_transformer_simple(self, n_nodes: int, prob: float) -> List[int]:
        r"""
        Generate binary transformer degrees (simple probabilistic model).

        Each node independently gets transformer degree 0 or 1 with
        probability ``1 - prob`` and ``prob``, respectively.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        prob : float
            Probability that a node participates in a transformer edge.

        Returns
        -------
        list of int
            Transformer degree list (0 or 1 per node).
        """
        return self.rng.choice([0, 1], size=n_nodes, p=[1.0 - prob, prob]).tolist()

    def _generate_transformer_stars(self, n_i: int, n_j: int, 
                                    c: float = 0.174, gamma: float = 4.15) -> Tuple[List[int], List[int]]:
        r"""
        Generate transformer degrees using the disjoint k-stars model.

        Follows Section 6.2 of `Aksoy et al. (2018)
        <https://doi.org/10.1093/comnet/cny016>`_:

        1. Number of star centres: :math:`h(n_i, n_j) = c \cdot \min(n_i, n_j)`
           with :math:`c \approx 0.174`.
        2. Star sizes sampled from a discrete power law
           :math:`P(k) \propto k^{-\gamma}` with :math:`\gamma \approx 4.15`.
        3. Each star is randomly assigned a centre in level *i* or *j*.
        4. Degree lists are padded with zeros and shuffled.

        Parameters
        ----------
        n_i, n_j : int
            Number of nodes in the two voltage levels.
        c : float, optional
            Coefficient for the number of active star centres.
            Default is 0.174 (paper optimum).
        gamma : float, optional
            Power-law exponent for star sizes.
            Default is 4.15 (paper optimum, Section 6.2).

        Returns
        -------
        tuple of (list of int, list of int)
            ``(t_i, t_j)`` — transformer degree sequences for levels *i* and
            *j*, each of length *n_i* and *n_j* respectively.
        """
        # 1. Determine number of 'star centers' (active hubs)
        # h(n_i, n_j) = c * min(n_i, n_j)
        n_stars = int(max(1, np.round(c * min(n_i, n_j))))
        
        # 2. Generate Star Sizes from Power Law
        # Since gamma is high (4.15), most stars are small (size 1 or 2).
        # We sample from range [1, 100] (arbitrary cap, probability of >100 is negligible)
        max_star_size = 100
        x = np.arange(1, max_star_size + 1)
        probs = x ** (-gamma)
        probs /= probs.sum()
        
        star_sizes = self.rng.choice(x, size=n_stars, p=probs)
        
        # 3. Assign Stars to create Degree Sequences
        # To satisfy bipartite constraint (sum(d_i) == sum(d_j)), we build pairs.
        # For a star of size k:
        #   - Center in Level I: Level I gets one node deg=k, Level J gets k nodes deg=1
        #   - Center in Level J: Level J gets one node deg=k, Level I gets k nodes deg=1
        
        # We start with empty lists of non-zero degrees
        active_degrees_i = []
        active_degrees_j = []
        
        for size in star_sizes:
            # Randomly assign center to I or J (50/50 split assumption)
            if self.rng.random() < 0.5:
                # Center in I
                active_degrees_i.append(size)
                active_degrees_j.extend([1] * size)
            else:
                # Center in J
                active_degrees_j.append(size)
                active_degrees_i.extend([1] * size)
                
        # 4. Pad with zeros to match full node counts
        # We need to ensure we don't have more active nodes than total nodes
        if len(active_degrees_i) > n_i:
            # Truncate if we generated too many (rare given c=0.174)
            active_degrees_i = active_degrees_i[:n_i]
        else:
            active_degrees_i.extend([0] * (n_i - len(active_degrees_i)))
            
        if len(active_degrees_j) > n_j:
            active_degrees_j = active_degrees_j[:n_j]
        else:
            active_degrees_j.extend([0] * (n_j - len(active_degrees_j)))
            
        # 5. Shuffle to distribute connections randomly
        self.rng.shuffle(active_degrees_i)
        self.rng.shuffle(active_degrees_j)
        
        return active_degrees_i, active_degrees_j

    def create_params(self, 
                      levels: List[Dict[str, Any]], 
                      inter_connections: Dict[Tuple[int, int], Dict[str, Any]]) -> Dict[str, Any]:
        r"""
        Generate the full parameter set for :meth:`PowerGridGenerator.generate_grid`.

        Parameters
        ----------
        levels : list of dict
            One dict per voltage level with keys:

            * ``'n'`` (int): number of nodes.
            * ``'avg_k'`` (float): target average degree.
            * ``'diam'`` (int): target diameter.
            * ``'dist_type'`` (str): ``'dgln'``, ``'dpl'``, or ``'poisson'``.
            * ``'max_k'`` (int, optional): maximum degree (default:
              ``min(n - 1, 50)``).  For ``'dpl'`` the paper suggests
              :math:`d_{\max} \approx 1.517\,n^{1/4}`; for ``'dgln'``
              :math:`\bar{d} \approx 2.425 \pm 0.185` is consistent across
              subgraphs (Section 6.1 of Aksoy et al., 2018).

        inter_connections : dict
            Mapping ``(i, j) -> config`` for transformer connections.
            Config is either:

            * ``{'type': 'k-stars', 'c': 0.174, 'gamma': 4.15}``
            * ``{'type': 'simple', 'p_i_j': float, 'p_j_i': float}``

        Returns
        -------
        dict
            Keys ``'degrees_by_level'``, ``'diameters_by_level'``, and
            ``'transformer_degrees'``, ready to be unpacked into
            :meth:`PowerGridGenerator.generate_grid`.
        """
        degrees_by_level = []
        diameters_by_level = []
        
        # --- 1. Voltage Levels ---
        for i, lvl_cfg in enumerate(levels):
            n = int(lvl_cfg['n'])
            avg_k = float(lvl_cfg['avg_k'])
            d = int(lvl_cfg['diam'])
            dist_type = lvl_cfg.get('dist_type', 'poisson')
            max_k = lvl_cfg.get('max_k', min(n - 1, 50))

            print(f"Generating Level {i}: {dist_type.upper()} distribution (Avg={avg_k})")

            if dist_type == 'poisson':
                deg_seq = self._generate_poisson_degrees(n, avg_k)
            else:
                deg_seq = self._generate_optimized_degrees(n, avg_k, max_k, dist_type)
            
            degrees_by_level.append(deg_seq)
            diameters_by_level.append(d)

        # --- 2. Transformer Connections ---
        transformer_degrees = {}
        
        for (i, j), cfg in inter_connections.items():
            if i >= len(levels) or j >= len(levels): continue
                
            n_i = int(levels[i]['n'])
            n_j = int(levels[j]['n'])
            
            conn_type = cfg.get('type', 'simple')
            
            if conn_type == 'k-stars':
                print(f"Generating Transformers {i}<->{j}: k-Stars Model")
                c_val = cfg.get('c', 0.174)
                gamma_val = cfg.get('gamma', 4.15)
                print(gamma_val)
                t_i, t_j = self._generate_transformer_stars(n_i, n_j, c_val, gamma_val)
            else:
                # Simple probabilistic model
                print(f"Generating Transformers {i}<->{j}: Simple Probabilistic")
                p_i = cfg.get('p_i_j', 0.0)
                p_j = cfg.get('p_j_i', 0.0)
                t_i = self._generate_transformer_simple(n_i, p_i)
                t_j = self._generate_transformer_simple(n_j, p_j)
            
            transformer_degrees[(i, j)] = (t_i, t_j)

        return {
            "degrees_by_level": degrees_by_level,
            "diameters_by_level": diameters_by_level,
            "transformer_degrees": transformer_degrees
        }