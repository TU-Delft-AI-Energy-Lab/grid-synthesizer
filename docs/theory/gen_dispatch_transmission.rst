########################################################
Generation Dispatch and Transmission Capacity Assignment
########################################################

In previous pages, we explained how the grid topology can be generated, bus types assigned, and generation/load capacities set. In this page, we explain how the **generation dispatch** and **transmission line capacity limits** are determined for a synthetic grid model, based on :cite:p:`sadeghian2018novel`.

As noted in :cite:p:`sadeghian2018novel`, a valid synthetic grid model requires at least three components:

a) the electrical grid topology;
b) the generation and load settings (correlated placement and sizing);
c) the transmission constraints (capacity limits of transmission lines and transformers).

This page addresses component (c), together with the generation dispatch needed to compute power flows. The methods are implemented in `generation_dispatcher.py <../autoapi/powergrid_synth/generation_dispatcher/index.html>`_, `transmission.py <../autoapi/powergrid_synth/transmission/index.html>`_, and `dcpf.py <../autoapi/powergrid_synth/dcpf/index.html>`_.


------------
System model
------------

Consider a power grid with $N$ buses and $M$ branches (transmission lines and transformers). The admittance matrix $\mathbf{Y}_{N \times N}$ is

.. math:: \mathbf{Y}_{N \times N} = \mathbf{A}^\top \Lambda^{-1}(z_l) \mathbf{A}

where $\mathbf{A}$ is the branch–node incidence matrix, $\Lambda^{-1}(\cdot)$ the diagonal inverse matrix, and $z_l$ the vector of branch impedances (per-unit).

By neglecting power losses (the DC power flow approximation), the power balance and flow distribution follow

.. math::

   \mathbf{P}(t) &= \mathbf{B}'(t)\,\boldsymbol{\theta}(t) \\
   \mathbf{F}(t) &= \Lambda(y_l)\,\mathbf{A}\,\boldsymbol{\theta}(t)

where $\boldsymbol{\theta}(t)$ is the vector of bus voltage phase angles, $\mathbf{P}(t)$ the vector of net real power injections, $\mathbf{B}'$ the susceptance matrix, and $\mathbf{F}(t)$ the vector of branch real-power flows.

Grid operations must also satisfy:

.. math::

   P_g^{\min} \leq P_g \leq P_g^{\max}, \qquad P_L^{\min} \leq P_L \leq P_L^{\max}, \qquad |F_l| \leq F_l^{\max}

Two key normalized parameters are introduced:

- **Dispatch factor** of generator $i$:

  .. math:: \alpha_i = \frac{P_{g_i}}{P_{g_i}^{\max}}, \qquad i = 1, \ldots, N_G

  where $\alpha_i = 0$ means uncommitted and $\alpha_i = 1$ means fully committed.

- **Transmission gauge ratio** of branch $l$:

  .. math:: \beta_l = \frac{F_l}{F_l^{\max}}, \qquad l = 1, \ldots, M

  where $\beta_l \in (0,1]$ under normal operation, and $\beta_l > 1$ indicates overloaded (emergency) operation.


------------------------------------
Statistics of generation dispatch
------------------------------------

Correlation between capacity and dispatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Statistical analysis of real-world grids (NYISO-2935, WECC-16994, PEGASE-13659, ERCOT) reveals a non-trivial correlation between generation capacity $P_{g_n}^{\max}$ and actual dispatch $P_{g_n}$, with Pearson coefficient

.. math:: \rho(P_{g_n}^{\max}, P_{g_n}) \in [0.75, 0.95]

Three categories of generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All generators in a grid can be divided into three categories:

* **(A) Uncommitted units** ($\alpha_g = 0$): about 10–20% of total generators. These tend to be small or medium-size units. Reasons include market requirements, annual overhaul, or low load levels.

* **(B) Partially committed units** ($0 < \alpha_g < 1$): about 40–50% of total. Their output power varies between minimum and maximum generation capacity.

* **(C) Fully committed units** ($\alpha_g \approx 1$): the remaining generators operate very close to their maximum capacity.

Normalized capacity and dispatch factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The normalized generation capacity is defined as

.. math:: \bar{P}_{g_n}^{\max} = \frac{P_{g_n}^{\max}}{\max_i P_{g_i}^{\max}} \in [0, 1]

and there exists a significant correlation between $\bar{P}_{g_n}^{\max}$ and $\alpha_n$ with Pearson coefficient

.. math:: \rho(\bar{P}_{g_n}^{\max}, \alpha_n) \in [0.15, 0.55]

Small and mid-size units tend to have a wider range of dispatch factor compared with larger units. A small number of units may have negative dispatch factor (e.g., pumped-storage hydro acting as load).

2-D empirical PMF
^^^^^^^^^^^^^^^^^^

A joint probability density $f(\bar{P}_{g_n}^{\max}, \alpha_n)$ is estimated from real grid data and discretized into a 2-D probability table ``Tab_2D_Pg`` of size $14 \times 10$ (14 capacity bins $\times$ 10 dispatch-factor bins). This table captures the observed correlation structure. Its purpose is to ensure that the synthetic grid reproduces a similar correlation between generation dispatch and capacity as found in real systems.

Distribution of committed units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For committed units (category B), the analysis shows:

* More than 99% of committed units have capacities following an **exponential distribution** with parameter $\mu_{\text{committed}}$.
* About 1% have extremely large capacities falling outside the normal exponential range.
* The dispatch factor distribution depends on the **loading level** of the system $\alpha^\Sigma = \sum_{i=1}^{N_G} P_{g_i} / \sum_{i=1}^{N_G} P_{g_i}^{\max}$. For high loading levels (e.g., NYISO: $\alpha^\Sigma = 0.74$), it tends toward a generalized extreme value distribution; for low loading levels (e.g., PEGASE: $\alpha^\Sigma = 0.38$), it approaches a uniform distribution.


-----------------------------------------
Algorithm for generation dispatch
-----------------------------------------

The generation dispatch algorithm assigns a dispatch factor $\alpha_i$ to each generator bus, producing the active power output $P_{g_i} = \alpha_i \cdot P_{g_i}^{\max}$. The algorithm proceeds in four stages.

Stage 1: Uncommitted units ($\alpha = 0$)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Select 10–20% of generators as uncommitted. The target capacities are drawn from Uniform$[0, 0.6]$, and units whose normalized capacity is closest to each target are selected. These units receive $\alpha = 0$ (zero dispatch).

Stage 2: Committed units ($0 < \alpha < 1$)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the remaining units, select 40–50% as partially committed:

1. **Capacity selection**: 99% of selected units follow an exponential distribution with parameter $\mu_{\text{committed}}$, and 1% are drawn from the extreme tail (Uniform$[0.5, 1.0]$).

2. **Dispatch factor generation**: Generate a random set of $\alpha$ values. When the loading level flag ``alpha_mod = 0``, all values are Uniform$[0, 1]$. Otherwise, 99.5% are Uniform$[0, 1]$ and 0.5% are negative (representing pumped-storage or similar units).

3. **2-D bin-matching assignment**: Assign dispatch factors to units using the ``Tab_2D_Pg`` table. The table is scaled to integer counts matching the number of committed units. Units are sorted by normalized capacity into 14 bins; alphas are sorted into 10 bins. The assignment loops from high capacity to low, randomly pairing units with alpha values from the corresponding bins. This reproduces the observed 2-D correlation between capacity and dispatch factor.

Stage 3: Fully committed units ($\alpha = 1$)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All remaining generators are assigned dispatch factor $\alpha = 1$.

Stage 4: Balancing
^^^^^^^^^^^^^^^^^^^^

The total generation must match total load (within a tolerance of 1%). An iterative loop adjusts the dispatch:

* **Excess generation**: scale down committed alphas, or turn off fully committed and uncommitted units.
* **Generation deficit**: scale up committed alphas (capped at 1.0), or turn on uncommitted/full-load units.

This guarantees a feasible power balance for the subsequent DC power flow.


---------------------------------------
DC power flow solver
---------------------------------------

After generation dispatch, a DC power flow (DCPF) is solved to determine the flow distribution  across all branches. The DCPF implementation in ``dcpf.py`` proceeds as follows:

1. **Susceptance matrix** $\mathbf{B}$: For each branch $(i,j)$ with reactance $x_{ij}$, the susceptance is $b_{ij} = 1/x_{ij}$. Off-diagonals: $B_{ij} = -b_{ij}$; diagonals: $B_{ii} = \sum_j b_{ij}$.

2. **Power injection vector** $\mathbf{P}$: Net injection at bus $i$ is $P_i = (P_{g_i} - P_{L_i}) / S_{\text{base}}$, where $S_{\text{base}} = 100$ MVA.

3. **Slack bus selection**: The bus with the largest generator is set as the slack (reference angle $\theta_{\text{slack}} = 0$).

4. **Solve**: Remove the slack bus row/column from $\mathbf{B}$ and $\mathbf{P}$, then solve the reduced system

   .. math:: \mathbf{B}_{\text{red}} \, \boldsymbol{\theta}_{\text{red}} = \mathbf{P}_{\text{red}}

   using a sparse direct solver.

5. **Branch flows**: For each branch $(i,j)$ with reactance $x_{ij}$:

   .. math:: F_{ij} = \frac{\theta_i - \theta_j}{x_{ij}} \times S_{\text{base}} \quad \text{(MW)}


-----------------------------------------
Branch impedance generation
-----------------------------------------

Before the DCPF can be run, each branch needs an impedance $z_l = r_l + jx_l$. The impedance assignment follows the SynGrid MATLAB toolbox (``sg_line.m``).

Impedance magnitude (LogNormal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The per-unit impedance magnitude $Z$ is drawn from a LogNormal distribution:

.. math:: Z \sim \text{LogNormal}(\mu_Z, \sigma_Z)

with parameters $\mu_Z$ and $\sigma_Z$ extracted from reference system data (e.g., $\mu_Z = -2.38$, $\sigma_Z = 1.99$ for NYISO). Values are clipped to $[0.001, 0.5]$ p.u. for numerical stability.

Line angle via Lévy stable distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The impedance angle $\phi$ (determining the X/R ratio) is generated from a Lévy $\alpha$-stable distribution:

.. math:: \phi \sim S(\alpha_s, \beta_s, \gamma_s, \delta_s)

with parameters $(\alpha_s, \beta_s, \gamma_s, \delta_s)$ from reference data (e.g., $(1.374, -0.838, 2.965, 85.801)$ for NYISO). The angle is clipped to $[0.01°, 89.99°]$, then the reactance and resistance are computed as:

.. math::

   x_l &= Z_l \sin(\phi_l) \\
   r_l &= Z_l \cos(\phi_l)

DCPF-based impedance swapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure that the impedance assignment is consistent with realistic flow patterns, the following iterative procedure is applied:

1. Run DCPF to obtain flow magnitudes $|F_l|$.
2. Sort impedance magnitudes $Z$ in ascending order; sort flow indices in descending order of $|F_l|$.
3. **Assign low-$Z$ to high-flow lines**: lines carrying more power should have lower impedance (shorter/thicker conductors). This is the physical principle that lower impedance paths carry more current.
4. **Random perturbation**: to avoid an overly deterministic assignment, random swaps are performed among neighboring entries. The swap rate and neighborhood size depend on grid size:

   * For $N > 1200$: swap rate $a_s = 0.3$, neighborhood $a_n = 0.8$
   * Otherwise: $a_s = 0.2$, $a_n = 0.2$

5. Reassign $Z_l$, $x_l$, $r_l$ to each branch using the original line angle $\phi_l$.

This process may be iterated (2 times for $N \geq 300$, once otherwise).


----------------------------------------------
Transmission capacity statistics and assignment
----------------------------------------------

Scaling property
^^^^^^^^^^^^^^^^^

The aggregate transmission capacity in a grid follows a scaling law with network size:

.. math:: \log_{10} F_l^{\text{tot}}(N) = 1.03 \log_{10} N + 2.52

where $F_l^{\text{tot}} = \sum_{m=1}^{M} F_{l_m}^{\max}$ is the total transmission capacity.

Exponential distribution of gauge ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The transmission gauge ratio $\beta_l = F_l / F_l^{\max}$ follows an exponential distribution with parameter $\mu_\beta$ (e.g., $\mu_\beta \approx 0.27$ for NYISO, $0.28$ for WECC):

.. math:: \beta_l \sim \text{Exp}(\mu_\beta)

Values are clipped to $(0, 1]$ for normal operation. A small fraction (about 0.08–0.11%) of lines are designated as **overloaded** with $\beta_l \in (1.0, 1.2]$, representing short-term or emergency ratings.

Correlation between gauge ratio and flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There exists a considerable correlation between the gauge ratio $\beta_l$ and the normalized flow $\bar{F}_l$, with Pearson coefficient $\rho \in [0.35, 0.65]$. This is captured in a 2-D probability table ``Tab_2D_FlBeta`` (size varies by reference system: e.g., $18 \times 16$ for NYISO, $16 \times 14$ for WECC), where rows correspond to flow classes and columns to gauge-ratio classes.


-------------------------------------------
Algorithm for transmission capacity assignment
-------------------------------------------

After DCPF gives the flow distribution $F_l$, the transmission capacity assignment proceeds:

1. **Normalize flows**: $\bar{F}_l = |F_l| / \max_l |F_l| \in [0, 1]$.

2. **Generate gauge ratios**: Draw $M$ values from $\text{Exp}(\mu_\beta)$, inject overloads, and sort.

3. **2-D bin-matching**: Using the ``Tab_2D_FlBeta`` table, assign gauge ratios to lines based on their normalized flow magnitude. The procedure is analogous to the dispatch factor assignment: flows are sorted into bins (rows), betas into bins (columns), and pairing proceeds from high to low within each 2-D cell.

4. **Calculate capacity**: For each line $l$:

   .. math:: F_l^{\max} = \frac{|F_l|}{\beta_l}

   If $F_l^{\max} \leq 2$ MW (negligible flow), apply a minimum capacity of $5 + 100 \cdot U(0,1)$ MW to avoid degenerate limits.

.. note::

   The paper specifies a scaling check: if $\sum F_m^{\max}$ deviates more than 5% from $F_l^{\text{tot}}(N)$, all capacities should be rescaled proportionally. This is not currently implemented in the code.


------------------------------------------
Topology refinement (optional)
------------------------------------------

After impedance and capacity assignment, an optional topology refinement step reduces excessive phase-angle spread in the grid. This is based on the SynGrid heuristic (``sg_flow_lim.m``):

1. Run DCPF and compute the angle spread $\Delta\theta = \max(\theta) - \min(\theta)$ (in degrees).
2. Compare against a target threshold $\text{TT} = 10^{0.3196 \log_{10} N + 0.8324}$.
3. If $\Delta\theta > \text{TT} + 2°$:

   a. **Add a strengthening line**: connect the bus pair with the maximum angle difference, using low impedance ($r \approx 0.001$–$0.002$, $x \approx 0.002$–$0.005$ p.u.).
   b. **Remove a weak line**: from the top 20% highest-impedance lines, randomly remove one (ensuring both endpoints have degree $\geq 3$ for grids with $N \geq 40$). Revert if the removal disconnects the grid.

4. Repeat up to 10 iterations.


.. bibliography::
   :filter: docname in docnames