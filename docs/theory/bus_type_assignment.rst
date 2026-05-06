#############################################
Bus Type Assignment based on Bus Type Entropy
#############################################

Here we explain the statistical method used to assign the bus types given a grid topology. 
In a typical power grid, $20-40\%$ of the buses are generation buses, $40-60\%$ load buses, and about $20\%$ connection buses. 

Elyas and Wang :cite:p:`elyas2016improved` verified that there exist non-trivial correlations between the three bus types and other topology metrics such as node degrees and clustering coefficients in a real-world power grid. 
A numerical measure, called **Bus Type Entropy**, was proposed to quantify the correlated bus type assignments of realistic power grids. The entropy definition was subsequently improved with redefined formulations and better-fitted scaling functions for the normalized distance parameter.

The methods are implemented in `bus_type_allocator.py <../autoapi/powergrid_synth/bus_type_allocator/index.html>`_ (ported from the MATLAB SynGrid toolbox ``sg_bus_type.m``).

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
What is a bus type assignment?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A bus type assignment vector $\mathbb{T}$ gives one of the following types to each bus node $i$

  .. math:: 
      
      \text{TYPE} := [\mathbb{T}]_i = \begin{cases} 1 & \mbox{for a generator (G)} \\ 2 & \mbox{for a load (L)} \\  3 & \mbox{for a connection (C)} \end{cases}
  
The resulting edge types are defined by the pairs of connected buses:

  .. math::

      \begin{array}{ll}
      {11}: \text{G} \leftrightarrow \text{G} & {12}: \text{G} \leftrightarrow \text{L} \\
      {22}: \text{L} \leftrightarrow \text{L} & {13}: \text{G} \leftrightarrow \text{C} \\
      {33}: \text{C} \leftrightarrow \text{C} & {23}: \text{L} \leftrightarrow \text{C}
      \end{array}

^^^^^^^^^^^^^^^^^^^
Findings
^^^^^^^^^^^^^^^^^^^
The authors performed an analysis on the Pearson correlations 

* $\rho(t,k_t)$ between the bus type $t$ and the average node degree $k_t$, and 
* $\rho(t,C_t)$ between the bus type $t$ and the average local clustering coefficient $C_t$. 

The analysis shows that 

* **Positive correlations:** connection buses tend to have higher average node degree and clustering coefficient. This is consistent with the role of connection (transit) buses as hubs in the transmission network.
* **Negative correlations:** generation or load buses are likely to be less densely connected or clustered, reflecting their more peripheral placement in realistic grids.


^^^^^^^^^^^^^^^^^
Bus type entropy
^^^^^^^^^^^^^^^^^

The bus type entropy is given in two definitions: 
  
  .. math:: 
  
      W_0(\mathbb{T}) = - \sum_{k=1}^3 r_k \cdot \log(r_k) - \sum_{i,j=1}^3 R_{ij} \cdot \log(R_{ij})

      W_1(\mathbb{T}) = -\sum_{k=1}^{3} \log(r_k) \cdot N_k - \sum_{i,j=1}^{3} \log(R_{ij}) \cdot M_{ij} 

where the variables are defined as:

* **Bus Type Ratios** ($r_k$): Represent the ratios of generation (G), load (L), and connection (C) buses, :math:`r_k = \frac{N_k}{N}`
* **Link Type Ratios** ($R_{ij}$): Represent the ratios of the six edge types (GG, LL, CC, GL, GC, LC), :math:`R_{ij} = \frac{M_{ij}}{M}`
* **Network Parameters**:
    * $N$: Total network size (number of buses).
    * $M$: Total number of branches in the grid.
    * $M_{ij}$: Total number of branches of a specific link type $ij$.

.. admonition:: Why do we want an entropy notion? 

  The proposed entropies provide a quantitative means to identify the presence of correlation between different bus type assignments and the power grid topology. They help us to recognize the specific set of bus type assignments, in the spectrum of random ones generated from permutation.

  * $W_0$ is a typical entropy definition of statistical variables, from which the derived entropy value tends to fall within a very stable or restricted numerical region. Because $\mu_{W_0}$ is nearly constant across network sizes, $W_0$ is useful for comparing grids of different sizes on an absolute scale.
  * $W_1$ is a more generalized one, which tends to amplify the scaling impact of the entropy value versus the grid network size, and has the advantage to simplify the approximation procedure of the scaling function. Because $\mu_{W_1}$ grows with $N$, $W_1$ better separates the optimal assignment from random ones in large grids.


^^^^^^^^^^^^^^^^^^^^
Statistical analysis
^^^^^^^^^^^^^^^^^^^^

By investigating the relative difference or distance between the best/original bus type assignment in a realistic power grid and other randomized ones, one can see the effectiveness of the proposed entropy, and further allow us to derive a scaling property of the bus type assignment w.r.t. the grid size. 

1. For the original assignment $\mathbb{T}^*$ in a realistic power grid, compute the entropy $W_{0/1}(\mathbb{T}^*)$. 
2. Randomize the bus type assignments $\tilde{\mathbb{T}}$ while keeping the bus type ratio unchanged, and compute the corresponding entropy. This creates many random samples of the entropy. Denote the sampling size by $k^{\text{max}}$. 

.. admonition:: Notes

  By the central limit theorem (CLT), the empirical PDF of $W_{0/1}(\mathbb{T})$ converges to a normal distribution. The total number of all possible samples would be no greater than $\frac{N!}{N_G! N_L! N_C!}$. For $N\leq 4000$, we may consider $k^{\text{max}}=25,000$; while for even larger grids, we set $k^{\text{max}}=40,000$.

3. Estimate the distribution parameters $(\mu,\sigma)$ to measure the relative location of $\mathbb{T}^*$ and $\tilde{\mathbb{T}}$.

.. admonition:: Statistical results
  
  By analyzing the empirical PDFs, it shows that 
  
  * for $W_0(\mathbb{T})$, the fitting parameter $\mu$ is very stable w.r.t. the network size; and 
  * for $W_1(\mathbb{T})$, $\mu$ grows w.r.t. the network size. 
  * There is a trend for the distance between the original bus type and the mean value $\mu$ --- as the network size increases, the value of $W^*-\mu$ reduces from positive to negative values.

  These statistical results have an implication on building a scaling property for the entropy w.r.t. the grid size. 


^^^^^^^^^^^^^^^^^^^^
The scaling property 
^^^^^^^^^^^^^^^^^^^^

The idea is to capture a scaling property of the optimal entropy $W^*$ w.r.t. the network size with the help of the fitting parameters $(\mu,\sigma)$. With this relationship, we can estimate the optimal entropy $W^*$ w.r.t. a related empirical PDF. 

By examining the realistic power grids systems, the authors found that 

.. admonition:: Observations 

  The relative location of $W^*$ is not stationary but there exists a growing trend in the distance between $W^*$ and $\mu$ w.r.t. the network size $N$, that is, $W^*$ moves from the right to the left side of the distribution as the network size increases. 

This observation presents a possible scaling property of the normalized entropy $d$ in terms of the network size $N$, defined as 

.. math:: 

  d (N) = \frac{W^*-\mu}{\sigma}

From the realistic power grids systems, the authors computed the normalized $d$ and RMSE-fitted piecewise functions (a linear part and a nonlinear part). This leads to the following approximations for $d_0(N)$ 

.. math:: 

  d_0(N) = 
  \begin{cases} 
  - 1.721 \log N + 8 & \mbox{if } \log N \leq 8 \\
  -6.003\times 10^{-14} (\log N)^{15.48} & \mbox{if } \log N>8 
  \end{cases}

and for $d_1(N)$

.. math::

  d_1(N) = 
  \begin{cases} 
  - 1.748 \log N + 8.576 & \mbox{if } \log N \leq 8 \\
  -6.053\times 10^{-22} (\log N)^{24.1} & \mbox{if } \log N>8 
  \end{cases}


Given a generated power grid topology, this allows us to identify the optimal bus type entropy $W^*$ as follows 

.. math:: W^* = \mu + \sigma \cdot d (N)

where $\mu,\sigma$ are estimated from the empirical PDF of by randomizing bus type assignments in the given topology. 


^^^^^^^^^
Algorithm 
^^^^^^^^^

.. admonition:: For a syntheric power grid topology, bus type assignment steps:

  1. build an empirical probability density function (PDF) of the entropy values $W(\mathbb{T})$ by randomizing bus type assignments over the raw grid topology
  2. calculate the distribution parameters $(\mu,\sigma)$
  3. estimate the scaling property $d(N)$ using the proposed approximations 
  4. find the target bus type entropy $W^*$ using the parameters from steps 2 and 3
  5. optimize to search for the desired assignment, with the stopping criteria $\epsilon\leq 0.003$, giving rise to the target entropy 

In step 5, an optimization can be designed with the following objective :math:`\min_{\mathbb{T}} \epsilon = |W(\mathbb{T})-W^*|`. This optimization is achieved by the so-called Artificial Immune Systems (AIS) Algorithm.

.. toctree::
   :maxdepth: 1

   ais_bus_type_assignment


.. bibliography::
   :filter: docname in docnames