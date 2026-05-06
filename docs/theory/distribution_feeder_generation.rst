############################################################
Automated generation of synthetic MV distribution feeders
############################################################

This section describes the methodology for automatically generating synthetic medium-voltage (MV) radial distribution feeders, based on the work of :cite:p:`schweitzer2017automated`. The approach treats distribution feeders as tree graphs and identifies statistical patterns in their structural and electrical properties. These patterns are then exploited in a synthesis algorithm that generates feeders statistically similar to real-world data.

The core idea is that many node and edge properties in a radial feeder can be expressed as functions of the **hop distance** from the primary substation. This graph-centric perspective bridges Complex Network Science (CNS) metrics and power engineering design principles. A large dataset from a Dutch DSO (covering :math:`\sim 8200\,\text{km}^2`, :math:`\sim 100` feeders) serves as the empirical basis.

Overview
^^^^^^^^

The generation algorithm proceeds through five sequential steps, each governed by one or more statistical distributions identified from the data:

1. **Node generation** --- determine the number of nodes and assign hop distances using a Negative Binomial distribution.
2. **Feeder connection** --- connect nodes into a tree using a bimodal Gamma degree distribution, with degree clipping based on hop distance.
3. **Node properties** --- assign load/generation to nodes using Beta, Poisson, Normal, and t-Location-Scale distributions.
4. **Cable type assignment** --- select cable types from a library using the Exponential distribution for the current utilization ratio.
5. **Cable length assignment** --- assign branch lengths using a modified Cauchy distribution, with hop-dependent length clipping.

The quality of synthesis is assessed using the **KL-Divergence** between distributions of the synthetic and real feeders.


Notation
^^^^^^^^

The following conventions are used throughout:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`N`
     - Total number of nodes in a feeder
   * - :math:`n`
     - A single node object
   * - :math:`M`
     - Total number of branches in a feeder
   * - :math:`m`
     - A single branch object
   * - :math:`n.h`
     - Hop distance of node :math:`n` from the HV source (number of edges along the unique path)
   * - :math:`n.d`
     - Degree of node :math:`n` (number of incident branches)
   * - :math:`n.P`, :math:`n.Q`
     - Real and reactive power at node :math:`n`
   * - :math:`m.I_\text{est}`
     - Estimated current in branch :math:`m`
   * - :math:`m.I_\text{nom}`
     - Nominal (rated) current of the cable assigned to branch :math:`m`
   * - :math:`m.\ell`
     - Length of branch :math:`m`
   * - :math:`\mathbb{1}(\cdot)`
     - Indicator function (evaluates to 1 if the argument is true, 0 otherwise)
   * - :math:`\mathcal{U}(0,1)`
     - Uniform distribution on the unit interval

Properties are accessed via **dot notation**: e.g., :math:`n.P` is the real power load at node :math:`n`.


Feeder definition and analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **feeder** is defined as the section of the MV distribution system fed by a single primary substation MV bus, plus the HV source bus on the other side of the distribution transformer. The HV node is the **source** (:math:`n.h = 0`), and the MV bus directly connected to it is the **root** (:math:`n.h = 1`).

Feeders are identified from the full system graph :math:`G(V, E)` by performing a **Breadth First Search (BFS)** from each MV neighbor of an HV source, excluding the HV source and its other neighbors. All nodes discovered in the BFS constitute the feeder.

Since distribution feeders are operated radially, the resulting subgraph is a **tree**. This has important consequences:

* The unique path from any node to the source is unambiguous.
* Hop distance :math:`n.h` serves as a proxy for centrality measures (analogous to betweenness centrality in meshed networks).
* Clustering coefficients are zero by definition.
* Power flow can be estimated without knowing line parameters, since downstream power can be computed by summing loads along the tree.

An additional set of **reduced feeders** is obtained by merging nodes connected by negligible impedances (e.g., :math:`R = X = 1\,\mu\Omega`). These are used for most of the analysis, as the distinction between a large busbar and two small busbars connected by near-zero impedance is immaterial for statistical purposes.


Verification methodology: KL-Divergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **Kullback--Leibler (KL) Divergence** is used throughout to quantify how well synthetic distributions match the real data:

.. math::

   D_{KL}(p \| q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} \, dx

where :math:`p(x)` is the empirical distribution from the data and :math:`q(x)` is the fitted model.

The KL-Divergence is operationally interpreted as follows: an observer trying to distinguish samples from :math:`p` versus :math:`q` will err with probability decaying exponentially in the number of observations, at a rate equal to :math:`D_{KL}`. A small :math:`D_{KL}` means that synthetic samples are statistically indistinguishable from real data given a moderate sample size.

Meaningful ranges for :math:`D_{KL}` are determined in two steps:

1. **Functional law selection**: considering aggregate (cumulative) data from all feeders for higher statistical relevance, the distribution with the lowest :math:`D_{KL}` with respect to the data is selected.
2. **Per-feeder range**: the distribution of :math:`D_{KL}` values between each individual feeder and the selected law provides a weighted range for the divergence.


Step 1: Node generation
^^^^^^^^^^^^^^^^^^^^^^^^

.. topic:: Procedure: GenerateNodes

   **Inputs**: power factor CDF, Negative Binomial distribution parameters :math:`(r, p)`.

   1. The first node is the source at :math:`n.h = 0`.
   2. The second node is the root at :math:`n.h = 1`.
   3. For each remaining node :math:`n = 3, 4, \ldots, N`:

      a. Assign :math:`n.\text{power factor}` from the empirical power factor CDF.
      b. Sample :math:`n.h` from the Negative Binomial distribution.

   4. Adjust hop distances to ensure there are no gaps (every hop level :math:`h` from 0 to :math:`\max(h)` has at least one node).

The **Negative Binomial distribution** governs the assignment of hop distances:

.. math::

   f(x; r, p) = \frac{\Gamma(r + x)}{x!\,\Gamma(r)}\, p^r (1 - p)^x, \qquad x = 0, 1, 2, \ldots

where :math:`r > 0` and :math:`0 \leq p \leq 1`. The Negative Binomial is interpreted as an **over-dispersed Poisson**: the variance exceeds the mean, but a mean and variance are sufficient to describe the process. This models the random process of "how far a node is from the source."

Each node also receives a **power factor** from an empirical CDF extracted from the data, simplifying later computations to focus on real power only.


Step 2: Feeder connection
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. topic:: Procedure: ConnectNodes

   **Inputs**: Mixture Gamma distribution parameters, maximum degree clipping function :math:`g_{d_\text{max}}(h)`.

   1. All leaf nodes (at :math:`\max(h)`) and the source (:math:`h = 0`) are assigned degree 1.
   2. The root (:math:`h = 1`) has its degree determined deterministically by the number of nodes at :math:`h = 2` plus one (for the source connection).
   3. For each remaining node :math:`n`:

      a. Sample a degree :math:`d_\text{tmp}` from the mixture Gamma distribution.
      b. Resample until :math:`d_\text{tmp} \leq g_{d_\text{max}}(n.h)`.
      c. Assign :math:`n.d^* \leftarrow d_\text{tmp}`.

   4. **Sort** nodes in ascending order of :math:`h`.
   5. **Connect** from the furthest nodes toward the source: for each node :math:`n` (from :math:`N` down to 2), connect to the predecessor :math:`p` (where :math:`p.h = n.h - 1`) for which :math:`p.d - p.d^*` is most negative (i.e., the predecessor whose actual degree is furthest below its target).

The **degree distribution** of the feeders is fit by a **mixture of two Gamma distributions**:

.. math::

   f(x; \pi, a_1, b_1, a_2, b_2) = \pi \cdot g(x; a_1, b_1) + (1 - \pi) \cdot g(x; a_2, b_2)

where:

.. math::

   g(x; a, b) = \frac{1}{b^a \, \Gamma(a)}\, x^{a-1} e^{-x/b}, \qquad x > 0

The bimodal Gamma reflects two clearly distinct decay rates observed in the empirical degree distribution. This is consistent with findings in the literature that exponential or sum-of-exponential distributions fit the degree distribution of distribution networks, with the Gamma being the conjugate prior of the Exponential.

The radial tree structure imposes deterministic constraints: leaf nodes have degree 1, the source has degree 1, and the root's degree is fully determined by the number of nodes at :math:`h = 2`. The connection algorithm works **bottom-up** (from leaves toward root), always picking the predecessor whose actual degree most needs to increase.


Step 3: Node properties (load and generation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nodes are classified into three types: **intermediate** (no load), **generation** (negative load / power injection), and **consumption** (positive load). Each is handled by a separate sub-procedure.

Intermediate nodes
___________________

.. topic:: Procedure: Intermediate

   **Inputs**: Beta distribution for fraction of intermediate nodes; mixture Poisson distribution for their hop distances.

   1. Sample the fraction of intermediate nodes: :math:`\epsilon \sim \text{Beta}(\alpha, \beta)`, set :math:`N_\text{intermediate} = N \cdot \epsilon`.
   2. The source (:math:`n = 1`) is always marked as intermediate.
   3. For each remaining intermediate node :math:`i = 1, \ldots, N_\text{intermediate} - 1`:

      a. Sample a hop distance from a **mixture Poisson distribution**.
      b. Mark a node at that hop distance as intermediate.

The **Beta distribution** models the fraction of zero-load nodes:

.. math::

   f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)}\, x^{\alpha - 1}(1 - x)^{\beta - 1}, \qquad 0 < x < 1

where :math:`B(\cdot)` is the Beta function. The Beta is a natural choice for modeling fractional quantities bounded between 0 and 1.

The hop distance of each intermediate node follows a **mixture Poisson distribution**:

.. math::

   f(x; \pi, \mu_1, \mu_2) = \pi \frac{\mu_1^x}{x!} e^{-\mu_1} + (1 - \pi) \frac{\mu_2^x}{x!} e^{-\mu_2}, \qquad x = 0, 1, 2, \ldots

The bimodal Poisson reflects the physical observation that junction nodes (no load) occur predominantly close to the primary substation, where main sub-feeders separate, with a secondary mode one-third to halfway down the feeder, reflecting further geographical splitting or voltage-level transitions.


Power injection (generation) nodes
____________________________________

.. topic:: Procedure: PowerInjection

   **Inputs**: Beta distribution for fraction of injection nodes; mixture Normal distribution for normalized hop distance; Normal distribution for injection deviation.

   1. Sample fraction: :math:`\epsilon \sim \text{Beta}`, set :math:`N_\text{inj} = \text{round}(N \cdot \epsilon)`.
   2. For each injection node :math:`i = 1, \ldots, N_\text{inj}`:

      a. Sample normalized hop distance :math:`\epsilon' \sim` mixture Normal.
      b. Select a node :math:`n` with :math:`n.h = \epsilon' \cdot \max(h)`.
      c. Sample deviation :math:`\epsilon'' \sim \mathcal{N}(\mu, \sigma)`, resample until :math:`1/N_\text{inj} + \epsilon'' > 0`.
      d. Set :math:`n.P = -P_\text{inj,total}(1/N_\text{inj} + \epsilon'')`.
      e. Set :math:`n.Q = n.P \cdot \tan(\cos^{-1}(n.\text{power factor}))`.

The **normalized hop distance** (i.e., :math:`h / h_\text{max}`) of injection nodes follows a **mixture of two Normal distributions**:

.. math::

   f(x; \pi, \mu_1, \sigma_1, \mu_2, \sigma_2) = \pi \cdot \mathcal{N}(x; \mu_1, \sigma_1) + (1 - \pi) \cdot \mathcal{N}(x; \mu_2, \sigma_2)

The main mode is close to the substation (small generators, PV installations, wind turbines), with a smaller bump further down the feeder (LV feeders feeding power back).

The **deviation** of each injection from the uniform distribution :math:`1/N_\text{inj}` is:

.. math::

   \epsilon = \frac{n.P_\text{inj}}{\sum_{n} n.P_\text{inj}} - \frac{1}{N_\text{inj}}

which is found to be **Normally distributed**.


Positive load (consumption) nodes
___________________________________

.. topic:: Procedure: PositiveLoad

   **Inputs**: t-Location-Scale distribution parameters.

   For each non-intermediate, non-injection node :math:`n`:

   1. Sample :math:`\epsilon \sim` t-Location-Scale, resample until :math:`1/N + \epsilon > 0`.
   2. Set :math:`n.P = P_\text{total}(1/N + \epsilon)`.
   3. Set :math:`n.Q = n.P \cdot \tan(\cos^{-1}(n.\text{power factor}))`.

The fundamental design principle is that utilities attempt to distribute load **evenly** across a feeder. The deviation from uniform distribution:

.. math::

   \epsilon = \frac{n.P}{\sum_n n.P} - \frac{1}{N}

is tightly centered around zero and follows a **t-Location-Scale distribution**:

.. math::

   f(x; \mu, \sigma, \nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sigma\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\sigma}\right)^2\right)^{-\frac{\nu+1}{2}}

where :math:`\sigma, \nu > 0`. In the limit :math:`\nu = 1`, this reduces to the Cauchy distribution. The empirical fit parameters indicate near-Cauchy behavior, reflecting heavy tails: most loads are close to the uniform value, but occasional significant deviations occur.


Step 4: Cable type assignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. topic:: Procedure: CableType

   **Inputs**: cable library (from k-means clustering), Exponential distribution for :math:`I_\text{est}/I_\text{nom}`.

   Processing from the **furthest branches toward the source**:

   1. For each branch :math:`m` with non-zero estimated current:

      a. With probability :math:`2/3`: set :math:`m.I_\text{nom}` to the maximum nominal current of the downstream node (matching the empirical finding that :math:`\sim 2/3` of nodes have uniform incident cable ratings).
      b. Otherwise: sample :math:`\epsilon \sim \text{Exponential}(\mu)`, compute :math:`I_\text{nom} = I_\text{est}/\epsilon`, and select the cable from the library with the closest :math:`I_\text{nom}`.

   2. For zero-current branches: set :math:`I_\text{nom}` as the average of max and min :math:`I_\text{nom}` on the upstream node, and pick the closest cable.

The **estimated current** in each branch is computed without a full power flow, exploiting the radial tree structure:

.. math::

   m.I_\text{est} = \frac{m.S_\text{downstream}}{\sqrt{3}\, m.V_\text{nom}}

where :math:`m.S_\text{downstream}` is the apparent power of all downstream nodes. This is the key advantage of the radial assumption: power flow is computable from topology alone.

The ratio :math:`I_\text{est}/I_\text{nom}` follows an **Exponential distribution**:

.. math::

   f(x; \mu) = \frac{1}{\mu}\, e^{-x/\mu}, \qquad x \geq 0

The **cable library** is built from the data via k-means clustering on nominal current. It contains per-distance impedance parameters and the empirical frequency of each cable type, which is used to weight the selection.


Step 5: Cable length assignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. topic:: Procedure: CableLength

   **Inputs**: modified Cauchy distribution parameters, maximum length clipping function :math:`g_\text{max}(h)`.

   For each branch :math:`m`:

   1. Sample :math:`\ell_\text{tmp}` from the modified Cauchy distribution.
   2. Resample until :math:`\ell_\text{tmp} \leq g_\text{max}(m.h)`.
   3. Set :math:`m.\ell = \ell_\text{tmp}`.

The **modified Cauchy distribution** is identified by observing a clear exponential decay in the magnitude of the **empirical characteristic function** (Fourier transform of the length histogram):

.. math::

   \varphi_x(t; x_0, \gamma) = e^{j x_0 t - \gamma |t|}

Only the Cauchy distribution exhibits such a decay, leading to the fit:

.. math::

   f(x; x_0, \gamma) = \left[\arctan\!\left(\frac{x_0}{\gamma}\right) + \frac{\pi}{2}\right]^{-1} \frac{\gamma}{(x - x_0)^2 + \gamma^2}

where :math:`x_0 \in \mathbb{R}`, :math:`\gamma > 0`, and the distribution is modified to have support :math:`x > 0` only.


Clipping distributions
^^^^^^^^^^^^^^^^^^^^^^^^

Since several of the fitted distributions have heavy tails or unbounded support, extreme samples can violate physical or engineering constraints. **Clipping functions**, all expressed in terms of hop distance :math:`n.h`, are applied to bound the sampled values.


Maximum degree clipping
_________________________

Branching should decrease with distance from the substation. The maximum degree at each hop level follows a **power law**:

.. math::

   g_{d_\text{max}}(h) = a \cdot h^b

During degree assignment, the bimodal Gamma is resampled until :math:`d \leq g_{d_\text{max}}(h)`.


Maximum nominal current clipping
__________________________________

Larger (more expensive) cables are not used far from the substation. A threshold is applied: for :math:`h \geq 8`, the nominal current is restricted to :math:`I_\text{nom} \leq 450\,\text{A}`.


Maximum cable length clipping
________________________________

Voltage drop constraints physically limit conductor length. The maximum length at each hop distance follows an **exponential function**:

.. math::

   g_\text{max}(h) = a \cdot e^{b \cdot h}

The Cauchy distribution is resampled until :math:`\ell \leq g_\text{max}(h)`.

The effect of clipping is to condition the distribution: :math:`f(x) \to f(x \,|\, x < x_\text{max}(h))`. This redistributes probability mass from outside the constrained domain into the domain, with the support depending on :math:`h`. These clipping trends are directly observed in the real data.


Emergent validation distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two key distributions **emerge naturally** from the synthesis without being explicitly modeled, providing strong validation:

Downstream power distribution
________________________________

The downstream power of a node :math:`n_i` is the sum of all real power flowing past it toward the leaves:

.. math::

   n_i.P_\text{downstream} = \sum_{n_j \leftarrow n_i} n_j.P

where :math:`n_j \leftarrow n_i` denotes all nodes downstream of :math:`n_i`. When normalized by total feeder load, this follows a **Generalized Pareto distribution**:

.. math::

   f(x; k, \sigma, \theta) = \frac{1}{\sigma}\left(1 + k \cdot \frac{x - \theta}{\sigma}\right)^{-1 - 1/k}

where :math:`x > \theta` and :math:`k > 0`.


Per-unit voltage drop distribution
_____________________________________

The estimated voltage drop across each branch, expressed as a fraction of nominal voltage:

.. math::

   m.\Delta V = \frac{m.I_\text{est} \cdot m.Z}{m.V_\text{nom}}

where :math:`m.Z` is computed from the per-distance cable parameters and length :math:`m.\ell`. This also follows a **Generalized Pareto distribution**. The fact that these distributions emerge from the interplay of the individually modeled properties indicates that the complex interactions within the feeder are correctly captured.


Algorithm inputs and evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The generation algorithm requires three scalar inputs:

1. :math:`N` --- number of nodes
2. Total load (MVA)
3. Total generation (MVA)

These are sampled from a **three-dimensional Kernel Density Estimate (KDE)** fitted to the real data vector :math:`(N, \text{Load}, \text{Generation})`.

Performance is evaluated by generating a large ensemble (e.g., 427 synthetic feeders) and comparing distributions across all properties. When inputs are similar to the training data, the :math:`D_{KL}` values remain low. When inputs deviate (e.g., doubled load), :math:`D_{KL}` increases by roughly an order of magnitude, thus serving as an indicator of how "extreme" a generated case is relative to the data.


Summary of statistical distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Property
     - Distribution
     - Equation
   * - Hop distance
     - Negative Binomial
     - :math:`f(x; r, p) = \frac{\Gamma(r+x)}{x!\,\Gamma(r)} p^r (1-p)^x`
   * - Degree
     - Mixture Gamma
     - :math:`f(x) = \pi g(x; a_1, b_1) + (1-\pi) g(x; a_2, b_2)`
   * - Fraction of intermediate nodes
     - Beta
     - :math:`f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}`
   * - Hop of intermediate nodes
     - Mixture Poisson
     - :math:`f(x) = \pi \frac{\mu_1^x e^{-\mu_1}}{x!} + (1-\pi) \frac{\mu_2^x e^{-\mu_2}}{x!}`
   * - Fraction of injection nodes
     - Beta
     - (same form as above)
   * - Normalized hop of injection nodes
     - Mixture Normal
     - :math:`f(x) = \pi \mathcal{N}(x;\mu_1,\sigma_1) + (1-\pi) \mathcal{N}(x;\mu_2,\sigma_2)`
   * - Injection power deviation
     - Normal
     - :math:`\mathcal{N}(\mu, \sigma)`
   * - Load deviation
     - t-Location-Scale
     - :math:`f(x;\mu,\sigma,\nu) \propto (1 + (x-\mu)^2/(\nu\sigma^2))^{-(\nu+1)/2}`
   * - Current ratio :math:`I_\text{est}/I_\text{nom}`
     - Exponential
     - :math:`f(x;\mu) = \mu^{-1} e^{-x/\mu}`
   * - Cable length
     - Modified Cauchy
     - :math:`f(x; x_0, \gamma) \propto \gamma / [(x - x_0)^2 + \gamma^2]`
   * - Max degree (clipping)
     - Power law
     - :math:`g_{d_\text{max}}(h) = a \cdot h^b`
   * - Max length (clipping)
     - Exponential function
     - :math:`g_\text{max}(h) = a \cdot e^{b \cdot h}`
   * - Downstream power (emergent)
     - Generalized Pareto
     - :math:`f(x;k,\sigma,\theta) = \sigma^{-1}(1+k(x-\theta)/\sigma)^{-1-1/k}`
   * - Voltage drop (emergent)
     - Generalized Pareto
     - (same form as above)


Generality and future directions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The authors note that while the distributions were calibrated on a Dutch DSO dataset, the **methodology is general**. Preliminary analysis of two U.S. feeders (IEEE 8500-bus and a California utility feeder) shows that similar statistical laws hold with different parameter values. The modular structure of the algorithm allows distributions to be swapped or reparametrized for different regions without changing the construction logic.

Current omissions acknowledged:

* **Transformers** are minimal in the single-feeder model (only the HV/MV transformer). Future multi-voltage-level models will incorporate them via a library approach.
* **Capacitors** are absent from the Dutch dataset (underground cables provide natural capacitance), but would need inclusion for other regions.
* **Meshed/reconfigurable topologies**: the algorithm focuses on radial feeders. Normally-open switches enabling reconfiguration, and the cycle distribution of the full system, are left for future work.

The increasing :math:`D_{KL}` when inputs deviate from the training data provides a built-in metric for assessing how "normal" a generated feeder is, enabling quality control in Monte Carlo applications.


Software implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The synthesis algorithm is implemented in the ``powergrid_synth.distribution`` subpackage.

**Graph classes.** Generated feeders can be wrapped as
:class:`~powergrid_synth.core.grid_graph.DistributionGrid` objects, a subclass of
:class:`~powergrid_synth.core.grid_graph.PowerGridGraph` (itself a ``networkx.Graph``
subclass).  ``DistributionGrid`` provides convenience properties for radial tree grids:

* ``root`` — the node at hop distance 0
* ``max_hop`` — maximum hop distance in the feeder
* ``is_radial`` — whether the graph is a connected tree
* ``total_load_mw`` / ``total_gen_mw`` — aggregate power quantities
* ``nodes_at_hop(h)`` / ``nodes_by_type(t)`` — node selection helpers

The companion :class:`~powergrid_synth.core.grid_graph.TransmissionGrid` class provides
analogous helpers for meshed, multi-voltage-level transmission networks (``voltage_levels``,
``n_levels``).

**Reference grid conversion.** To calibrate the algorithm from a real distribution grid
rather than the default Table III parameters, the module
:mod:`~powergrid_synth.distribution.distribution_converter` provides:

* :func:`~powergrid_synth.distribution.distribution_converter.pandapower_to_feeders` —
  converts a pandapower network to Schweitzer-format feeder graphs, handling lines,
  transformers, and closed bus–bus switches.
* :func:`~powergrid_synth.distribution.distribution_converter.pypowsybl_to_feeders` —
  converts a pypowsybl Network (loaded from CGMES, XIIDM, MATPOWER, PSS/E, etc.)
  to the same Schweitzer-format feeder graphs.
* :func:`~powergrid_synth.distribution.distribution_converter.feeder_summary` —
  returns a summary dict per feeder.

For transmission grids loaded from industry-standard formats,
:func:`~powergrid_synth.core.data_format_converter.load_grid` and
:func:`~powergrid_synth.core.data_format_converter.pypowsybl_to_nx` provide
direct conversion from any pypowsybl-supported file format to a NetworkX graph
compatible with the synthesis pipeline.

The fitted parameters can be fed directly to
:class:`~powergrid_synth.distribution.SchweetzerFeederGenerator` for synthesis.
See the ``DistributionSynthFromRef`` example notebook for a complete workflow.

.. bibliography::
   :filter: docname in docnames