########################################
Chung-Lu-Chain grid topology generation
########################################

Here we explain the graph model, Chung-Lu-Chain model, used to generate the raw grid topology. 
The methodology is mainly based on :cite:p:`aksoy2019generative`, which investigated the topological properties of power grids from the US Eastern Internconnection, Texas Interconnection, and Poland transmission system power grids and found

1. subgraphs induced by nodes of the same voltage level exhibit shared structural properties **atypical to small-world networks**, including 

  * low local clustering, 
  * large diameter, and 
  * large average distance 

2. subgraphs induced by transformer edges linking nodes of different voltage types contain **mainly small, disjoint star graphs**. 

The method proposed by the authors includes two phases:

1. the first phase uses the **Chung-Lu random graph model**, taking desired node degrees and desired diameter as inputs, and outputs the subgraphs of same-voltage level; and 
2. the second phase uses a simple **random star graph model** to connect the subgraphs of different voltage levels. 


Preliminaries
^^^^^^^^^^^^^
For a graph $G=(V,E)$ with $n$ nodes, we denote the degree of a node $i$ as $d_i$, and the degree sequence as $\mathbf{d}=[d_1,d_2,\dots,d_n]$. 
Some graph metrics. 

* average distance in a connected graph is the average distance between all pairs of vertices 

.. math:: \text{avgDist} = \frac{1}{n(n-1)} \sum_{i\in V} \sum_{j\in V, j\neq i} d(i,j)

* diameter $\delta$ of a connected graph is the maximum distance over all pairs of nodes 

.. math:: \delta = \max_{i,j\in V, i\neq j} d(i,j)

* local clustering coefficient (LCC) of a vertex is to measure how tightly connected the neighborhood of a vertex is: 

.. math:: lcc(i) = \frac{\# \text{triangles incident to vertex } i}{d_i(d_i -1)/2}

The LCC of a degree-1 vertex is undefined, and the local clustering coefficient of a graph is the average LCC over all vertices for which the LCC is defined. 

Power grid as network-of-networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following :cite:p:`halappanavar2015network`, a power grid graph is defined as $G=(V,E,f,\mathcal{X})$, where $V$ is the set of vertices (substations, buses, generators, loads, etc.), $E$ is the set of edges (transmission lines and transformers), $\mathcal{X}=\{X_1,\dots,X_k\}$ is the set of voltage levels, and $f:V\to \mathcal{X}$ assigns a voltage level to each vertex.

An edge $\{i,j\}$ is called a **transformer edge** if $f(i)\neq f(j)$. The power grid graph can be decomposed into:

* **Same-voltage subgraphs** $G[X]$ for each voltage level $X\in\mathcal{X}$: the subgraph induced by all vertices of voltage $X$ and all edges between them.
* **Transformer subgraphs** $G[T[X_i,X_j]]$: the subgraph induced by all transformer edges between voltage levels $X_i$ and $X_j$.

We denote:

* $n^X = \Vert V^X\Vert$: the number of vertices at voltage level $X$
* $\mathbf{d}^X$: the degree sequence of same-voltage vertices in $G[X]$
* $\mathbf{t}[X_i,X_j]$: the transformer degree sequence of voltage $X_i$ vertices with respect to voltage $X_j$, i.e., the number of transformer edges from each $X_i$ vertex to any $X_j$ vertex

In this way, the entire power grid graph is the union of all same-voltage subgraph edges and all transformer edges.

Key empirical findings from :cite:p:`aksoy2019generative`:

* Same-voltage subgraphs have diameter and average distance on the order of $\sqrt{n}$, much larger than the $\log(n)$ typical of small-world or scale-free models.
* Same-voltage subgraphs have consistently low local clustering coefficients (typically below 0.1), unlike most real-world networks.
* Average degree $\bar{d}\approx 2.425$ is remarkably consistent across all same-voltage subgraphs (CV = 7.6%).
* Transformer subgraphs consist almost entirely of small, disjoint $k$-star graphs.

What is a Chung-Lu model? 
^^^^^^^^^^^^^^^^^^^^^^^^^^
The Chung-Lu random graph model takes as an input a desired node degree sequence $\mathbf{d}=(d_1,\dots,d_n)$, with $d_i$ the desired degree of node $i$.
For every pair of nodes $i,j$, the probability of an edge $(i,j)$ is 

.. math:: \text{Pr}(\{i,j\} \text{ is an edge}) = \frac{d_i d_j}{2m}

where $m=\frac{1}{2}\sum_{i} d_i$ is the desired number of edges. 
In order to ensure that this probability is at most 1, one further requires that $\max_i d_i^2 \leq \sum_{k} d_k$. In expectation, each node achieves its desired degree $d_i$

.. math:: \mathbb{E}(\text{degree of node } i) = \sum_{i} \text{Pr} (\{i,j\} \text{ is an edge}) = \sum_j \frac{d_i d_j}{2m} = d_i.


Pseudo-algorithm
________________

Fast Chung-Lu model chooses the endpoints of $m$ edges by sampling these endpoints proportionally to their desired degree:

.. math:: \text{Pr}(i \text{ selected}) = \frac{d_i}{2m}.

The expected degree of a node is again its desired degree

.. math:: \text{Pr}(\{i,j\} \text{ is an edge}) = 2m \cdot \text{Pr}(i \text{ selected}) \cdot \text{Pr}(j \text{ selected}) = \frac{d_i d_j}{2m}.


Some prior work
^^^^^^^^^^^^^^^
Hines et al. (2010) considered the aggregate power grid graph from IEEE 300-bus test case and the US Eastern Interconnection grid, and found that the structure of these graphs differ substantially from the comparably sized random graph models like Erdõs-Rényi, preferential attachment, and small-world models in terms of the degree distribution, clustering, diameter and assortativity. 


Phase 1: The Chung-Lu Chain (CLC) model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard random graph models (Erdõs-Rényi, Chung-Lu, preferential attachment, small-world) produce graphs with diameter on the order of $\log(n)$, which is far smaller than the $\sqrt{n}$-scale diameters observed in real power grid same-voltage subgraphs. The CLC model addresses this by adapting the Chung-Lu model to additionally take a desired diameter $\delta$ as input.

**Preprocessing (Algorithm 1 --- Setup)**

Given desired degrees $\mathbf{d}=(d_1,\dots,d_n)$ and desired diameter $\delta$, the preprocessing stage:

1. **Diameter adjustment**: Adjusts the input diameter to account for within-box Chung-Lu graph diameter:

   .. math:: \delta \leftarrow \text{round}\Big(\delta - 2\log\frac{\eta}{\delta+1}\Big)

   where $\eta = |\{d\in\mathbf{d}:d>0\}|$ is the number of non-isolated vertices.

2. **Degree sequence inflation** (Lines 4--8): Inflates the degree sequence by randomly duplicating nonzero-degree vertices until the expected number of isolated vertices in a Chung-Lu realization matches the number of zero-degree vertices. The expected number of isolated vertices in a Chung-Lu graph is $\sum_i \exp(-d_i)$ (Chung \& Lu, 2006).

3. **Box assignment** (Lines 9--22): Partitions non-isolated vertices into $\delta+1$ boxes labelled $0,1,\dots,\delta$. If $\eta/(\delta+1) < \max(\mathbf{d})$, some boxes are kept empty (except for path vertices) to ensure enough vertices per non-empty box for the maximum degree to be achievable.

4. **Diameter path selection** (Lines 25--35): Selects $\delta+1$ vertices (preferably of degree $\geq 3$) and assigns each to a distinct box. These form the deterministic diameter path $D$ which guarantees the desired diameter.

5. **Subdiameter path selection** (Lines 37--45): Selects up to $\delta+1$ additional vertices and assigns them to consecutive boxes centered below the diameter path. The subdiameter path $S$ provides alternate paths and cycles, increasing edge connectivity and replicating long-cycle structures observed in real grids.

**Graph generation (Algorithm 2 --- CLC)**

Given the preprocessed outputs $({\mathbf{d}'},\mathbf{v},D,S)$:

1. **Diameter path edges** (Lines 3--7): Connect consecutive diameter path vertices across boxes $0,1,\dots,\delta$ to form a path of length $\delta$.
2. **Subdiameter path edges** (Lines 8--12): Connect consecutive subdiameter path vertices similarly.
3. **Within-box Chung-Lu** (Lines 13--22): For each box $k$, generate a fast Chung-Lu random graph on the vertices assigned to that box. Each edge is formed by sampling two endpoints proportionally to their desired degrees. Self-loops are discarded.

The resulting graph has diameter at least $\delta$ (from the deterministic path), low clustering (inherited from Chung-Lu), and the degree distribution approximately matches the input.


Phase 2: Transformer edges via random star generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The transformer subgraphs between voltage levels consist almost entirely of disjoint $k$-star graphs (Section 3.2 of the paper). Algorithm 3 (STARS) exploits this structure.

**Algorithm 3 --- Stars($\mathbf{t}[X,Y], \mathbf{t}[Y,X]$)**

Given the desired transformer degree sequences for voltage levels $X$ and $Y$:

1. **Partition vertices** into:

   * *Centers*: vertices with transformer degree $\geq 2$ (sets $I_X^c$ and $I_Y^c$)
   * *Leaves*: vertices with transformer degree $= 1$ (sets $I_X^o$ and $I_Y^o$)

2. **Generate $k$-stars** (Lines 7--25): For each center vertex $i\in I_X^c$ with degree $k$:

   * If $\geq k$ leaf vertices remain in $I_Y^o$, connect $i$ to $k$ randomly chosen leaves from $I_Y^o$ (removing them from the pool).
   * Otherwise, place $i$ in a leftover bin $L_X$.
   
   Repeat symmetrically for centers in $I_Y^c$ using leaves from $I_X^o$.

3. **Match remaining degree-1 vertices** (Lines 27--32): Pair remaining leaves from $I_X^o$ and $I_Y^o$ into single edges.

4. **Leftover bipartite Chung-Lu** (Lines 34--42): If leftover vertices remain whose degrees could not be realized via stars, apply a bipartite Chung-Lu model on the leftover bins.

**Sufficient condition for perfect degree match**: If $\sum_{i:t[X,Y]_i\geq 2} t[X,Y]_i \leq |\{j\in Y: t[Y,X]_j=1\}|$ (and symmetrically), then all vertices are allocated to stars and degrees are matched exactly.


Combined CLCStars model (Algorithm 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The full power grid graph $G$ on $k$ voltage levels is generated by:

1. For each voltage level $X_i$ ($i=1,\dots,k$):

   * Run **Setup**$(\mathbf{d}^{X_i}, \delta^{X_i})$ to preprocess.
   * Run **CLC**$(\mathbf{d}', \mathbf{v}, D, S)$ to generate same-voltage subgraph edges.

2. For each pair of voltage levels $(X_i, X_j)$ with $i<j$:

   * Run **Stars**$(\mathbf{t}[X_i,X_j], \mathbf{t}[X_j,X_i])$ to generate transformer edges.

3. Take the union of all edge sets to form $G$.

Phase 1 and Phase 2 operate independently: none of the Phase 1 inputs are required for Phase 2 and vice versa. This allows flexible control over the correlation between same-voltage structure and transformer connectivity.



Validation metrics
^^^^^^^^^^^^^^^^^^

:cite:p:`birchfield2017metric` and :cite:p:`birchfield2016grid` build an extensive list of 18 validation metrics for assessing the realism of systhetic power grid data. 

* metrics of system proportions: substations, load and generation

  * number of buses per substation
  * substation voltage levels
  * percentage of substations containing load 
  * load at each bus 
  * ratio of total generation capacity to total load 
  * percent of substations containing generation 
  * capacities of generators 
  * percent of geneartors committed 
  * generator dispatch percentage 
  * generator reactive power limits 

* metrics of system network: transformers and transmission lines

  * transformer per-unit reactance: $\geq 80\%$ of transformers have a reactance value in the range $[0.05, 0.2]$, and the distribution is roughly normal, centered around 0.12, with some variation.
  * transformer MVA limit and $X/R$ ratio 
  * transmission line reactance 
  * transmission line $X/R$ ratio and MVA limit 
  * ratio of transmission lines to substations, at a single nominal voltage level 
  * percent of lines on the minimum spanning tree 
  * distance of transmission lines along the Delaunay triangulation 
  * topology-related graph theory statistics: 

    * distribution of nodal degrees, clustering coefficient, average shortest path length


Degree distribution comparison statistics
__________________________________________

To quantitatively compare the degree distributions of a synthetic graph against a reference graph, we compute two complementary statistics --- the **Kolmogorov--Smirnov (KS) statistic** and the **Relative Hausdorff (RH) distance** --- both globally and per voltage level. These are implemented in the ``compare_degree_distributions`` method of :class:`~powergrid_synth.GraphComparator`.

**Kolmogorov--Smirnov (KS) statistic**

The two-sample KS test is a nonparametric test for equality of two distributions. Given empirical cumulative distribution functions (ECDFs) $F_1$ and $F_2$ of the degree sequences from the synthetic and reference graphs respectively, the KS statistic is

.. math:: D_{KS} = \sup_{x} |F_1(x) - F_2(x)|

$D_{KS}\in[0,1]$, where values close to 0 indicate that the two distributions are similar. The associated *p*-value tests the null hypothesis that both samples are drawn from the same distribution: a large *p*-value (e.g., $>0.05$) means we cannot reject that hypothesis, providing evidence that the synthetic degree distribution is statistically consistent with the reference.

**Relative Hausdorff (RH) distance**

The Hausdorff distance measures the worst-case discrepancy between two point sets. For two sorted degree sequences $\mathbf{d}^{(1)}$ and $\mathbf{d}^{(2)}$ (treated as 1-D point sets), the undirected Hausdorff distance is

.. math:: d_H(\mathbf{d}^{(1)}, \mathbf{d}^{(2)}) = \max\Big\{\sup_{a\in\mathbf{d}^{(1)}} \inf_{b\in\mathbf{d}^{(2)}} |a-b|, \; \sup_{b\in\mathbf{d}^{(2)}} \inf_{a\in\mathbf{d}^{(1)}} |b-a| \Big\}

We normalize by the maximum degree observed across both sequences to obtain the **Relative Hausdorff distance**:

.. math:: D_{RH} = \frac{d_H(\mathbf{d}^{(1)}, \mathbf{d}^{(2)})}{\max\big(\max(\mathbf{d}^{(1)}), \max(\mathbf{d}^{(2)})\big)}

$D_{RH}\in[0,1]$, where 0 means the degree ranges overlap perfectly. Unlike KS, which compares the shape of distribution functions, RH captures the worst-case mismatch in actual degree values. It is particularly sensitive to differences in maximum degree or the presence of outlier hubs.

**Interpretation guide**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Metric
     - Good range
     - Interpretation
   * - KS Statistic
     - $< 0.15$
     - Degree distribution shapes are similar; the two-sample test cannot distinguish them.
   * - KS *p*-value
     - $> 0.05$
     - Fail to reject the null hypothesis that both degree sequences come from the same distribution.
   * - RH Distance
     - $< 0.15$
     - The worst-case degree mismatch is small relative to the maximum degree in the system.

Both metrics are computed per voltage level, giving a fine-grained view of where the synthetic topology matches or deviates from the reference. 

  * ratio of total lengths of all lines to the length of the minimum spanning tree




Synthetic generation of topology generation inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Diameter $\delta$: The diameter of same-voltage subgraphs is on the order of $\sqrt{n}$, also suggested by :cite:p:`young2018topological`. E.g., one can use the approximate function of the form $f(n) = c\cdot n^k$ to estimate the diameter. In this work, it yields $c\approx 1.301$ and $k\approx 0.574$ using the three sets of data. 


* Same-voltage degree sequence $\mathbf{d}^X$: It was suggested to use the following 

  * a generalized log-normal degree distribution, where the number $n_d$ of degree $d$ nodes follows
      
      .. math:: n_d \propto \exp\Big(-\bigg(\frac{\log d}{\alpha}\bigg)^\beta\Big)
      
      for some parameters $\alpha, \beta$. To optimize $\alpha, \beta$, one needs user-specified target values for average degree $\bar{d}$ and maximum degree $d_\max$. This work finds a consistent average degree across same-voltage subgraphs, with mean $\bar{d}=2.425$ and standard deviation $0.1846$ and coefficient of variation $7.6\%$. 

  * a power law (a scale-free network) where the degree distribution follows a power law

      .. math:: n_d \propto d^{-\gamma}
      
      for some parameter $\gamma$. To optimizer $\gamma$, the maximum degree $d_\max$ is required, which is suggested $d_\max \sim n^{1/\gamma'}$ with some $\gamma'$. Fitting $d_\max$ in the three sets of data of this work using $g(n)=c\cdot n^{1/4}$ yields $c\approx 1.517$.

  Given the required parameter(s), average and/or maximum degree, we can use :cite:p:`kolda2014scalable` to optimize the distribution parameters and obtain the degree sequences. 
  
* Transformer degree sequence $\mathbf{t}[X,Y]$: For any given transformer subgraph between voltage levels $X_i$ and $X_j$, there are two related transformer degree sequences $\mathbf{t}[X_i, X_j]$ and $\mathbf{t}[X_j,X_i].$ The first step to synthetically generating these, one needs to specify how many vertices participate in each transformer graph, i.e., how many vertices of voltage $X_i$ are incident to a transformer edge having the other endpoint in voltage $X_j$, and vice versa for $X_j$. This number can also be approximated based on the number of vertices in the same-voltage subgraphs. 

Given a set of $k$ voltage levels $\mathcal{X}=\{X_1,\dots,X_k\}$, denote the number of voltage $X_i$ vertices by $n_i$, and the number of $X_i$ vertices that are incident via a transformer to a voltage $X_j$ vertex by $t_i^j$. A function $h(n_i,n_j)\approx t_i^j$ of the following form 

.. math:: h(n_i,n_j) = c \cdot \min \{n_i,n_j\}

is used to approximate this number with an optimal $c\approx 0.174$. 

Then, since transformer subgraphs are relatively small graphs consisting almost entirely of disjoint, small $k$-stars, where the number of $k$-stars decreases exponentially in $k$, their degree distributions are short and steep. Thus, one can consider simple power-law degree distributions for some large power-law exponent $\gamma$. An optimal $\gamma\approx4.5$ is found by fitting the data. 

Finally, for each pair of voltage levels $(X_i,X_j)$, one may draw a single power law degree distribution on $h(n_i,n_j)$ vertices. Note that since transformer subgraphs are bipartitie graphs, the degree sequences $\mathbf{t}[X_i, X_j]$ and $\mathbf{t}[X_j,X_i]$ must sum to the same value for each pair of voltage levels (a necessary condition). This constraint can be achieved by iteratively modifying the outputted degree sequences --- nullifying the degree of randomly chosen vertices from the larger-sum sequence while redrawing degrees for null-degreed vertices in the smaller-sum sequence, until the sums are equal. 

.. bibliography::
   :filter: docname in docnames