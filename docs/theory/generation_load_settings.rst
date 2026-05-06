====================================
Synthetic Generation and Load Settings 
====================================
In previous pages, we explained how the grid topology can be generated using a Chung-Lu-Chain graph model and the bus types can be assigned using statistics from real-world grids. 

In this page, we explain how the settings of generation and load buses in synthetic grid modeling can be done based on :cite:p:`elyas2017statistical`. The idea is to using the exponential distribution of individual generation capacities (or load settings) in a grid, and the non-trivial correlation between the generation capacity (or load setting) and the nodal degree of a generation (or load) bus. The methods are implemented in `capacity_allocator.py <../autoapi/powergrid_synth/capacity_allocator/index.html>`_

Authors in :cite:p:`elyas2017statistical` first analyzed the statistics of the generation capacities and their correlation with topology metrics of the grid, then proposed to determine the generation capcacity via a statistcal approach. A similar approach was doen for the load settings. 

---------------------
Power system modeling
---------------------
Consider an electrical topology of a power grid with $N$ buses and $M$ branches representing transmission lines and transformers. Denote the admittance matrix by $\mathbf{Y}_{N\times N}$, defined as 

.. math:: \mathbf{Y} = \mathbf{A}^\top \Lambda^{-1}(z_l) \mathbf{A}

where $\mathbf{A}$ is the **edge-node incidence matrix**. For a given power network with undirected and unitary links, $\Lambda^{-1}(\cdot)$ is the diagonal inverse matrix with a vector $\cdot$, and $z_l$ the vector of branch impedances in a grid. 

Then the power flow distribuion in a grid follows its network constraints as 

.. math:: \mathbf{P}(t) & = \mathbf{B}^\prime(t) \theta(t) \\ \mathbf{F}(t) & = \Lambda(y_l) \mathbf{A} \theta(t)

where $\mathbf{P}(t) = [\mathbf{P}_g(t), -\mathbf{P}_l(t), \mathbf{P}_l(t)]^\top$ represents the injected real power from generation, load and connection buses; $\theta(t)$ is the vector of phase angles, and $\mathbf{F}(t)$ the real-power delivered along the branches. 

More importantly, grid operations also needs to account for the constraints of generation capcacity, load settings and transmission capcacity such as 

.. math:: P_g^\min \leq & P_g \leq P_g^\max \\ P_l^\min \leq & P_l \leq P_l^\max \\ F^\min \leq & F \leq F_c^\max


--------------------------------
Statistics of Generation setting
--------------------------------
Authors first examined the statistical features of generation capacities in realistic grids in terms of aggregated generation capacity, distribution of individual capacities and their non-trivial correlation with nodal degrees. 

A scaling function of aggreated generation capacity in a grid w.r.t. its network size is derived as 

.. math:: \log P_g^{\text{tot}}(N) = -0.21 \log^2 N + 2.06 \log N + 0.66

where $P_g^{\text{tot}}=\sum_{n=1}^{N_g}P_{g_n}^\max$ denotes the total generation capacity, and $N_g$ is the total number of generation buses. 

This scaling law indicates that the total generation capacity in a grid tends to grow as a power function when the network size is small; whereas as the network size becomes larger, the total generation begins to slow down than the power function. 

For individual generation capacities (or load demands), some study on realistic grid data like NYISO-2935, and the WECC-16944 systems in this work shows that more than 99% of the generation units (and the loads as well) follow an exponential distribution with about 1% having extremely large capacities (or demands) falling out of the normal range defined by the expected exponential distribution. We refer to the Figure 1 in the paper for some empirical observations. (What's the reason behind this? Either an inherent heavy tailed distribution or due to the boundary equalization in a network reduction modeling.)

After analyese on the scaling property and the distribution of generation capacities in a grid, one proceeds further by looking into the correlation between the generation capacities and other topology metrics. 

One first normalize the generation capacity and nodal degree by their maximum as 

.. math:: \bar{P}_{g_n}^\max & = P_{g_n}^\max / \max_i P_{g_i}^\max \in [0,1] \\ \bar{k}_n & = k_n / \max_i k_i \in [0,1]

From a number of realistic grids, it shows there exists a correlation between these two values with a Pearson coefficient of 

.. math:: \rho(\bar{P}_{g_n}^\max, \bar{k}_n) \in [0.25, 0.5]; 

and most data points lie in the region of 

.. math:: \bar{P}_{g_n}^\max \in [0, 0.2] \quad \bar{k}_n \in [0, 0.5]

with few $\bar{P}_{g_n}^\max \geq 0.6$. 

Now we consider a 2D space of points $(\bar{P}_{g_n}^\max, \bar{k}_n)$, and calculate the empirical 2D PDF of the normalized variables in the set $A$

.. math:: \text{Pr}(A) = \text{Pr} \Big( (\bar{P}_{g_n}^\max, \bar{k}_n) \in A \Big)

This 2D PDF is discretized into a $14 \times 14$ probability table, denoted ``Tab_2D_Pgmax``, where rows correspond to 14 capacity classes (from low to high) and columns correspond to 14 nodal degree classes (from low to high). Each entry represents the joint probability of a generator belonging to a particular capacity class and degree class. These tables are extracted from realistic reference systems (e.g., NYISO-2935, WECC-16944) and stored in ``reference_data.py`` (ported from the MATLAB SynGrid toolbox ``sg_refsys_stat.m``).

----------------------------------------------
Algorithm for Generation Capacity Assignment
----------------------------------------------
Given the above statistical observations, the generation capacity assignment proceeds in the following stages.

**Stage 1: Total generation capacity.** Given a synthetic grid of size $N$, compute the total generation capacity using the scaling law:

.. math:: P_g^{\text{tot}} = 10^{-0.21 \log_{10}^2 N + 2.06 \log_{10} N + 0.66}

**Stage 2: Random generation of individual capacities.** Generate a random set of $N_g$ generation capacities from an exponential distribution with mean $\mu = P_g^{\text{tot}} / N_g$:

.. math:: P_{g_n}^\max \sim \text{Exp}(\mu), \quad n = 1, \ldots, N_g

To capture the observed heavy tail, approximately 1% of the units are replaced by "super-large" capacities drawn uniformly from $[\max(P_g), 3 \cdot \max(P_g)]$. If the sum of all capacities deviates beyond a tolerance band (more than 5% above or 10% below $P_g^{\text{tot}}$), the entire set is rescaled proportionally.

The capacities are then normalized:

.. math:: \bar{P}_{g_n}^\max = P_{g_n}^\max / \max_i P_{g_i}^\max

**Stage 3: Correlated assignment via 2D binning.** The assignment maps normalized capacities to generator buses while preserving the empirical degree–capacity correlation encoded in ``Tab_2D_Pgmax``. The algorithm proceeds as follows:

1. **Scale the 2D table to counts.** Multiply each entry of ``Tab_2D_Pgmax`` by $N_g$ and round to obtain integer target counts $n_{rc}$ for each (capacity class $r$, degree class $c$) cell. Adjust rounding errors so that $\sum_{r,c} n_{rc} = N_g$.

2. **Compute marginal targets.** The column sums give the target number of generators in each of the 14 degree bins; the row sums give the target number of generators in each of the 14 capacity bins.

3. **Bin generators by degree.** Sort generator buses by their normalized nodal degree $\bar{k}_n$ in ascending order. Partition the sorted list into 14 degree bins according to the column-sum targets.

4. **Bin capacities by value.** Sort the normalized capacities $\bar{P}_{g_n}^\max$ in ascending order. Partition into 14 capacity bins according to the row-sum targets.

5. **Assign capacities to degree bins.** For each degree bin $c = 1, \ldots, 14$ and for each capacity bin $r = 14, \ldots, 1$ (iterating from highest to lowest capacity class): randomly sample $n_{rc}$ capacities from capacity bin $r$ (without replacement) and assign them to unassigned generators in degree bin $c$.

6. **Denormalize.** Convert each assigned normalized capacity back to its actual value:

   .. math:: P_{g_n}^\max = \bar{P}_{g_n}^\max \cdot \max_i P_{g_i}^\max

The resulting set of generation capacities is statistically consistent with the empirical distribution and the observed degree–capacity correlation from realistic grids.


--------------------------
Statistics of Load setting
--------------------------
The load setting follows a methodology analogous to the generation capacity assignment. The key observations are:

- Individual load demands also follow an exponential distribution, with approximately 1% of loads being "super-large" outliers.
- A non-trivial correlation exists between the load demand and the nodal degree of a load bus, captured by a similar 2D probability table ``Tab_2D_load``.

**Total load computation.** Given the total generation capacity, the total system load is determined by one of four strategies:

- **Deterministic ('D'):** A scaling formula analogous to the generation case:

  .. math:: P_l^{\text{tot}} = 10^{-0.2 \log_{10}^2 N + 1.98 \log_{10} N + 0.58}

- **Light loading ('L'):** 30–40% of total generation capacity.
- **Medium loading ('M'):** 50–60% of total generation capacity.
- **Heavy loading ('H'):** 70–80% of total generation capacity.

The assignment of individual load values to load buses then uses the same 3-stage procedure: (1) generate random load demands from an exponential distribution (with ~1% super-large outliers), (2) normalize, and (3) assign via 2D binning using ``Tab_2D_load``. The methods are implemented in `load_allocator.py <../autoapi/powergrid_synth/load_allocator/index.html>`_.


---------------------------------
Reactive Power Load Allocation
---------------------------------
After active power loads :math:`P_l` are assigned, the corresponding reactive power loads :math:`Q_l` are derived using a power-factor model. For each load bus, a power factor is sampled uniformly from a realistic range:

.. math:: \text{pf}_n \sim \mathcal{U}[\text{pf}_{\min},\; \text{pf}_{\max}], \quad \text{pf}_{\min} = 0.85,\; \text{pf}_{\max} = 0.97

The reactive load is then computed as:

.. math:: Q_{l_n} = P_{l_n} \cdot \tan\!\big(\arccos(\text{pf}_n)\big)

Reference for active/reactive power relation and power-factor context:
`Phase to Phase, Chapter 9.5.2 <https://www.phasetophase.nl/book/book_2_9.html#_9.5.2>`_.

This yields lagging (inductive) reactive loads consistent with typical transmission system characteristics, where power factors fall in the range 0.85–0.97. The resulting :math:`Q_l` values are stored as the ``ql`` node attribute on the graph and are used by the pandapower and pypowsybl exporters when creating load elements.

The method is implemented in :meth:`LoadAllocator.allocate_reactive` and is called automatically during :func:`synthesize`.


.. bibliography::
   :filter: docname in docnames
