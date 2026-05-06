r"""
Assigns bus types (Generator, Load, Connection) to a raw power grid topology
using an entropy-based optimization approach from Elyas and Wang (2016),
"Improved Synthetic Power Grid Modeling With Correlated Bus Type Assignments"
(https://doi.org/10.1109/TPWRS.2016.2634318).

Ported from the MATLAB SynGrid toolbox (``sg_bus_type.m``).
"""

import numpy as np
import networkx as nx
import math
import random
from typing import List, Dict, Tuple, Optional


class BusTypeAllocator:
    r"""
    Assigns bus types (Generator, Load, Connection) to a raw power grid
    topology using an Artificial Immune System (AIS) optimization algorithm.

    The method, from Elyas and Wang (2016), exploits the observed non-trivial
    correlations between bus types and topology metrics (node degree, clustering
    coefficient) in realistic grids.  A **bus type entropy** measure quantifies
    these correlations, and a target entropy :math:`W^*` is derived from a
    scaling property fitted to real-world systems.  The AIS then searches for
    an assignment whose entropy matches :math:`W^*`.

    The pipeline is:

    1. Determine target bus type ratios :math:`(r_G, r_L, r_C)` from network
       size.
    2. Estimate :math:`W^*` via Monte Carlo sampling of random assignments
       and the scaling relation :math:`W^* = \mu + \sigma \cdot d(N)`.
    3. Run AIS optimization (clonal selection, hypermutation, receptor editing)
       to find an assignment :math:`\mathbb{T}` such that
       :math:`|W(\mathbb{T}) - W^*| < \epsilon`.

    Two entropy definitions are supported:

    - **Model 0** (:math:`W_0`): standard entropy of bus/link type ratios.
      :math:`\mu` is stable across network sizes.
    - **Model 1** (:math:`W_1`): generalized entropy weighted by :math:`N`
      and :math:`M`.  :math:`\mu` grows with network size, giving better
      discrimination in large grids.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph representing the grid topology (nodes and edges only;
        no bus type attributes required yet).
    entropy_model : int
        Selects the entropy definition: 0 for :math:`W_0`, 1 for :math:`W_1`.
    bus_type_ratio : list of float or None
        Optional target ratios ``[Gen, Load, Conn]``.  Values are normalized
        to sum to 1.  If *None*, default ratios are chosen based on *N*.

    References
    ----------
    .. [1] S. H. Elyas and Z. Wang, "Improved Synthetic Power Grid Modeling
       With Correlated Bus Type Assignments," IEEE Trans. Power Syst.,
       vol. 32, no. 5, pp. 3391–3400, Sept. 2017.
    """
    
    TYPE_GEN = 1
    TYPE_LOAD = 2
    TYPE_CONN = 3

    def __init__(self, graph: nx.Graph, entropy_model: int = 0, bus_type_ratio: Optional[List[float]] = None):
        self.graph = graph
        self.entropy_model = entropy_model
        
        # Pre-calculate graph stats
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.n_edges = graph.number_of_edges()
        
        # Map node to 0-based index for internal matrix logic
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for i, n in enumerate(self.nodes)}
        
        # Edge list as indices (M x 2)
        self.link_ids = []
        for u, v in graph.edges():
            self.link_ids.append([self.node_to_idx[u], self.node_to_idx[v]])
        self.link_ids = np.array(self.link_ids)

        # Pre-calculate degrees for constraint checking (Connection buses shouldn't be leaves)
        self.degrees = np.array([graph.degree(n) for n in self.nodes])
        self.non_leaf_indices = np.where(self.degrees > 1)[0]

        # Determine Ratios [G, L, C]
        if bus_type_ratio is not None:
            self.ratio_types = self._normalize_ratio(bus_type_ratio)
        else:
            self.ratio_types = self._get_ratios(self.n_nodes)
        
        # Store entropy samples for plotting
        self.w_samples: List[float] = []

    def _normalize_ratio(self, ratio: List[float]) -> List[float]:
        """
        Normalize a user-supplied bus type ratio to sum to 1.0.

        Parameters
        ----------
        ratio : list of float
            Three values ``[Gen, Load, Conn]``.

        Returns
        -------
        list of float
            Normalized ratios summing to 1.0.

        Raises
        ------
        ValueError
            If *ratio* does not contain exactly 3 positive values.
        """
        if len(ratio) != 3:
            raise ValueError("Bus type ratio must contain exactly 3 values [Gen, Load, Conn].")
        
        total = sum(ratio)
        if total <= 0:
             raise ValueError("Sum of bus type ratios must be positive.")
             
        return [x / total for x in ratio]
        
    def _get_ratios(self, n: int) -> List[float]:
        r"""
        Return default bus type ratios ``[Gen, Load, Conn]`` based on network
        size, derived from realistic reference systems.

        - :math:`N < 2000` → ``[0.23, 0.55, 0.22]`` (IEEE-300-like)
        - :math:`2000 \le N < 10000` → ``[0.33, 0.44, 0.23]`` (NYISO-like)
        - :math:`N \ge 10000` → ``[0.20, 0.40, 0.40]`` (WECC-like)

        Parameters
        ----------
        n : int
            Network size (number of buses).

        Returns
        -------
        list of float
            ``[r_G, r_L, r_C]`` summing to 1.0.
        """
        if n < 2000:
            return [0.23, 0.55, 0.22] # IEEE-300 like
        elif n < 10000:
            return [0.33, 0.44, 0.23] # NYISO like
        else:
            return [0.2, 0.4, 0.4]    # WECC like

    def _calculate_bus_ratios(self, assignment: np.ndarray) -> List[float]:
        """
        Compute actual bus type ratios from an assignment vector.

        Parameters
        ----------
        assignment : np.ndarray
            Bus type vector of shape ``(N,)`` with values in ``{1, 2, 3}``.

        Returns
        -------
        list of float
            ``[r_G, r_L, r_C]`` — actual ratios computed from the assignment.
        """
        n = len(assignment)
        if n == 0:
            return [0.0, 0.0, 0.0]
        counts = np.bincount(assignment, minlength=4)  # index 0 unused
        return [counts[1] / n, counts[2] / n, counts[3] / n]

    def _get_d_parameter(self, n: int) -> float:
        r"""
        Compute the normalized distance parameter :math:`d(N)` for estimating
        the target entropy :math:`W^*`.

        The piecewise scaling functions were fitted to realistic grids by
        Elyas and Wang (2016).  For entropy model 0:

        .. math::

            d_0(N) = \begin{cases}
              -1.39 \ln N + 6.79 & \text{if } \ln N \le 8 \\
              -6.003 \times 10^{-14} (\ln N)^{15.48} & \text{if } \ln N > 8
            \end{cases}

        For entropy model 1:

        .. math::

            d_1(N) = \begin{cases}
              -1.748 \ln N + 8.576 & \text{if } \ln N \le 8 \\
              -6.053 \times 10^{-22} (\ln N)^{24.1} & \text{if } \ln N > 8
            \end{cases}

        .. note::

           The :math:`d_0` linear-segment coefficients here (``-1.39``,
           ``6.79``) follow the MATLAB SynGrid implementation and differ
           from the values in the paper (``-1.721``, ``8``).

        Parameters
        ----------
        n : int
            Network size.

        Returns
        -------
        float
            The distance parameter :math:`d(N)`.
        """
        log_n = math.log(n)
        if self.entropy_model == 0:
            if log_n <= 8:
                return -1.39 * log_n + 6.79
            else:
                return -6.003e-14 * (log_n**15.48)
        else:
            if log_n <= 8:
                return -1.748 * log_n + 8.576
            else:
                return -6.053e-22 * (log_n**24.1)

    def _generate_random_assignment(self) -> np.ndarray:
        r"""
        Generate a random bus type assignment vector of length :math:`N`.

        Types: 1 = Generator, 2 = Load, 3 = Connection.  The counts are
        determined by ``self.ratio_types``.  Connection buses are preferentially
        placed on non-leaf nodes (degree > 1) to reflect the hub-like role of
        connection buses in realistic grids.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(N,)`` with values in ``{1, 2, 3}``.
        """
        assignment = np.zeros(self.n_nodes, dtype=int)
        
        n_gen = int(round(self.n_nodes * self.ratio_types[0]))
        n_conn = int(round(self.n_nodes * self.ratio_types[2]))
        # Load is remainder
        
        # 1. Assign Connection Buses (Type 3)
        # Prefer nodes with degree > 1
        candidates_conn = list(self.non_leaf_indices)
        if len(candidates_conn) < n_conn:
            # Fallback if not enough non-leafs
            candidates_conn = list(range(self.n_nodes))
            
        chosen_conn = np.random.choice(candidates_conn, size=n_conn, replace=False)
        assignment[chosen_conn] = self.TYPE_CONN
        
        # 2. Assign Generator Buses (Type 1)
        # Choose from remaining empty spots (0)
        available_indices = np.where(assignment == 0)[0]
        if len(available_indices) > 0:
            count = min(n_gen, len(available_indices))
            chosen_gen = np.random.choice(available_indices, size=count, replace=False)
            assignment[chosen_gen] = self.TYPE_GEN
            
        # 3. Assign Load Buses (Type 2) to remainder
        assignment[assignment == 0] = self.TYPE_LOAD
        
        return assignment

    def _calculate_link_ratios(self, assignment: np.ndarray) -> List[float]:
        r"""
        Compute ratios of the 6 link (edge) types for a given assignment.

        Each edge is classified by the sorted pair of its endpoint types:
        GG (1,1), LL (2,2), CC (3,3), GL (1,2), GC (1,3), LC (2,3).
        The ratio is :math:`R_{ij} = M_{ij} / M`.

        Parameters
        ----------
        assignment : np.ndarray
            Bus type vector of shape ``(N,)``.

        Returns
        -------
        list of float
            ``[R_GG, R_LL, R_CC, R_GL, R_GC, R_LC]``.
        """
        # Vectorized link type check
        u_types = assignment[self.link_ids[:, 0]]
        v_types = assignment[self.link_ids[:, 1]]
        
        # Sort pair so (1,2) is same as (2,1)
        pairs = np.sort(np.vstack((u_types, v_types)).T, axis=1)

        counts = {
            (1, 1): 0, (2, 2): 0, (3, 3): 0,
            (1, 2): 0, (1, 3): 0, (2, 3): 0
        }
        
        # Count unique pairs using numpy
        unique, counts_arr = np.unique(pairs, axis=0, return_counts=True)
        for pair, count in zip(unique, counts_arr):
            counts[tuple(pair)] = count
            
        M = self.n_edges if self.n_edges > 0 else 1
        return [
            counts[(1, 1)] / M, # GG
            counts[(2, 2)] / M, # LL
            counts[(3, 3)] / M, # CC
            counts[(1, 2)] / M, # GL
            counts[(1, 3)] / M, # GC
            counts[(2, 3)] / M  # LC
        ]

    def _calculate_entropy_score(self, bus_ratios: List[float], link_ratios: List[float]) -> float:
        r"""
        Compute the bus type entropy :math:`W` for a given assignment.

        **Model 0** (:math:`W_0`):

        .. math::

            W_0 = -\sum_{k=1}^{3} r_k \ln r_k
                  -\sum_{i,j=1}^{3} R_{ij} \ln R_{ij}

        **Model 1** (:math:`W_1`):

        .. math::

            W_1 = -\sum_{k=1}^{3} \ln(r_k) \cdot N_k
                  -\sum_{i,j=1}^{3} \ln(R_{ij}) \cdot M_{ij}

        where :math:`N_k = r_k \cdot N` and :math:`M_{ij} = R_{ij} \cdot M`.

        Parameters
        ----------
        bus_ratios : list of float
            ``[r_G, r_L, r_C]`` — bus type ratios (should reflect the
            **actual** assignment, not necessarily the target ratios).
        link_ratios : list of float
            ``[R_GG, R_LL, R_CC, R_GL, R_GC, R_LC]``.

        Returns
        -------
        float
            Entropy score :math:`W`.
        """
        EPS = 1e-12 # Avoid log(0)
        
        if self.entropy_model == 0:
            # Model 0: - sum(p log p)
            x1 = sum(r * math.log(r + EPS) for r in bus_ratios)
            x2 = sum(r * math.log(r + EPS) for r in link_ratios if r > 0)
        else:
            # Model 1: Weighted by N and M
            x1 = sum(math.log(r + EPS) * (r * self.n_nodes) for r in bus_ratios)
            x2 = sum(math.log(r + EPS) * (r * self.n_edges) for r in link_ratios if r > 0)
            
        return -(x1 + x2)

    def _estimate_w_star(self, monte_carlo_iters: int = 2000) -> Tuple[float, float]:
        r"""
        Estimate the target entropy :math:`W^*` via Monte Carlo sampling.

        Generates ``monte_carlo_iters`` random bus type assignments (with
        fixed target ratios), computes their entropies, and fits a normal
        distribution :math:`\mathcal{N}(\mu, \sigma^2)`.  The target entropy
        is then:

        .. math::

            W^* = \mu + \sigma \cdot d(N)

        where :math:`d(N)` is the piecewise scaling function from
        :meth:`_get_d_parameter`.

        Parameters
        ----------
        monte_carlo_iters : int, optional
            Number of random samples (default 2000).

        Returns
        -------
        w_star : float
            Target entropy value.
        std_w : float
            Standard deviation of the Monte Carlo entropy samples.
        """
        w_samples = []
        for _ in range(monte_carlo_iters):
            assign = self._generate_random_assignment()
            b_ratios = self._calculate_bus_ratios(assign)
            l_ratios = self._calculate_link_ratios(assign)
            w = self._calculate_entropy_score(b_ratios, l_ratios)
            w_samples.append(w)
        
        # Store for plotting
        self.w_samples = w_samples
            
        mean_w = np.mean(w_samples)
        std_w = np.std(w_samples)
        d_param = self._get_d_parameter(self.n_nodes)
        
        w_star = (d_param * std_w) + mean_w
        return w_star, std_w

    def allocate(self, max_iter: int = 100, population_size: int = 20) -> Dict[int, str]:
        r"""
        Run the AIS optimization to find a bus type assignment.

        Implements the Clonal Selection Principle:

        1. **Monte Carlo estimation** of :math:`W^*` and convergence
           threshold :math:`\epsilon` (see :meth:`_estimate_w_star`).
        2. **Initialize** a population of *K* random assignments.
        3. **Iterate** until ``max_iter`` or ``best_error < epsilon``:

           a. Evaluate fitness :math:`|W^* - W(\mathbb{T})|` for each
              individual.
           b. **Clonal selection** — top-ranked individuals produce more
              clones: :math:`N_c = \text{round}(\beta \cdot s / r)` where
              *r* is the rank.
           c. **Hypermutation** — clones are mutated with intensity
              proportional to their parent's rank (worse → more mutations).
           d. **Receptor editing** — inject 10% fresh random solutions to
              maintain diversity and avoid local optima.
           e. **Selection** — combine elite, mutated clones, and fresh
              solutions; keep the top *K*.

        4. Return the best assignment found.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of AIS iterations (default 100).
        population_size : int, optional
            Number of individuals surviving each generation (default 20).

        Returns
        -------
        dict[int, str]
            Mapping from node ID to bus type label
            (``'Gen'``, ``'Load'``, or ``'Conn'``).
        """
        print(f"Starting Bus Type Allocation (N={self.n_nodes}, M={self.n_edges})...")
        
        # 1. Estimate Target W* and Standard Deviation
        w_star, std_w = self._estimate_w_star()
        print(f"  Target Entropy Score (W*): {w_star:.4f}, Std Dev: {std_w:.4f}")
        
        # Determine Convergence Criteria (Logic from MATLAB)
        if self.n_nodes < 50:
            criteria = std_w / 2 if self.entropy_model == 1 else std_w / 10
        else:
            criteria = std_w / 1000

        # 2. Initialize Population
        population = []
        for _ in range(population_size):
            population.append(self._generate_random_assignment())
            
        best_solution = None
        best_error = float('inf')
        
        # 3. Optimization Loop
        for it in range(max_iter):
            # A. Evaluate Fitness
            scores = []
            for indiv in population:
                b_ratios = self._calculate_bus_ratios(indiv)
                l_ratios = self._calculate_link_ratios(indiv)
                w = self._calculate_entropy_score(b_ratios, l_ratios)
                error = abs(w_star - w)
                scores.append((error, indiv))
            
            # Sort by error (Ascending)
            scores.sort(key=lambda x: x[0])
            
            # Update Global Best
            current_best_error, current_best_sol = scores[0]
            if current_best_error < best_error:
                best_error = current_best_error
                best_solution = current_best_sol
            
            # Check Convergence
            if best_error < criteria:
                print(f"  Converged at iteration {it}. Error: {best_error:.6f} < Criteria: {criteria:.6f}")
                break
            
            if it % 10 == 0:
                print(f"  Iter {it}: Best Error = {best_error:.6f}")

            # B. Clonal Selection
            # Select top solutions to clone. 
            # Logic: B=0.4, s=100. nc = round(B*s/rank).
            # MATLAB uses the whole population here as source? Yes, iterates 1:L.
            clones = []
            B, s = 0.4, 100
            
            # Limit cloning to existing population size (sorted)
            for rank, item in enumerate(scores):
                indiv = item[1]
                # Rank is 1-based in formula
                n_clones = int(round((B * s) / (rank + 1)))
                for _ in range(n_clones):
                    clones.append(indiv.copy())
            
            # C. Mutation Operator
            # Mutates the clones.
            # Logic: Split clones into 10 groups. 
            # Worse groups get MORE mutations (higher intensity).
            mutated_clones = []
            L_clones = len(clones)
            chunk_size = L_clones / 10.0
            
            for j in range(L_clones):
                clone = clones[j]
                
                # Determine "group" (b) from 1 to 10
                # j starts at 0 (best clones) to L_clones-1 (worst clones)
                # Actually, clones are generated from best to worst parents, so order is roughly preserved.
                group_idx = int(j / chunk_size) + 1 # 1 to 10
                if group_idx > 10: group_idx = 10
                
                # Apply mutation 'group_idx' times
                for _ in range(group_idx):
                    # Pick random node
                    node_idx = np.random.randint(0, self.n_nodes)
                    
                    # Random flip to any other type (1, 2, or 3)
                    # MATLAB code bias: if rand<0.5 -> 1 else -> 2. 
                    # We will allow all 3 to maintain topological diversity.
                    new_type = np.random.choice([1, 2, 3])
                    clone[node_idx] = new_type
                
                mutated_clones.append(clone)
            
            # D. Diversity Injection
            # Add fresh random solutions. MATLAB adds 10% of new population size.
            n_fresh = max(1, int(len(mutated_clones) / 10))
            fresh_solutions = []
            for _ in range(n_fresh):
                fresh_solutions.append(self._generate_random_assignment())
                
            # E. Selection (Next Generation)
            # Combine: Fresh + Mutated Clones + Original Clones (MATLAB logic combines pop4, pop3, pop2)
            combined_population = fresh_solutions + mutated_clones + clones
            
            # Re-evaluate Combined Population
            combined_scores = []
            for indiv in combined_population:
                b_ratios = self._calculate_bus_ratios(indiv)
                l_ratios = self._calculate_link_ratios(indiv)
                w = self._calculate_entropy_score(b_ratios, l_ratios)
                err = abs(w_star - w)
                combined_scores.append((err, indiv))
            
            # Sort and Truncate to original population size
            combined_scores.sort(key=lambda x: x[0])
            
            # Keep top 'population_size'
            population = [x[1] for x in combined_scores[:population_size]]

        # 4. Final mapping
        bus_types = {}
        type_labels = {1: 'Gen', 2: 'Load', 3: 'Conn'}
        # Use best_solution found across all iterations
        for i, type_code in enumerate(best_solution):
            node_id = self.idx_to_node[i]
            bus_types[int(node_id)] = type_labels[type_code]
            
        return bus_types

    def plot_entropy_pdf(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the empirical PDF of entropy samples from the Monte Carlo
        estimation, with a fitted normal distribution overlay.

        Must be called after :meth:`allocate` or :meth:`_estimate_w_star`
        so that ``self.w_samples`` is populated.

        Parameters
        ----------
        figsize : tuple of int, optional
            Figure size ``(width, height)`` in inches.
        """
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import norm
        except ImportError:
            print("Matplotlib or Scipy not installed. Cannot plot PDF.")
            return

        if not hasattr(self, 'w_samples') or not self.w_samples:
            print("No entropy samples available. Please run allocate() first.")
            return

        plt.figure(figsize=figsize)
        
        # Plot Histogram
        # density=True ensures the area under the histogram sums to 1 (probability density)
        count, bins, ignored = plt.hist(self.w_samples, bins=50, density=True, 
                                      alpha=0.6, color='skyblue', edgecolor='black', 
                                      label='Empirical Data')
        
        # Fit and plot Normal Distribution
        mu, std = norm.fit(self.w_samples)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        
        plt.plot(x, p, 'r', linewidth=2, label=f'Normal Fit\n$\\mu={mu:.3f}$, $\\sigma={std:.3f}$')
        
        plt.title(f'Bus Type Entropy Distribution (N={self.n_nodes})')
        plt.xlabel('Bus Type Entropy (W)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()