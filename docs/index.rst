PowerGridSynth
==============

**PowerGridSynth** is an open-source Python package for generating realistic **synthetic power grids** at both the transmission and distribution level.

- **Transmission grids** are synthesized using a `Chung-Lu-Chain (CLC) <https://arxiv.org/abs/1711.11098>`_ graph model that reproduces prescribed degree distributions and diameters across multiple voltage levels, then layers on bus-type assignment, generation/load capacity allocation (active *and* reactive power), generation dispatch, and transmission-line impedance/capacity assignment drawn from empirical statistics of real grids (NYISO, WECC).
- **Distribution feeders** are synthesized using the algorithm of `Schweitzer et al. (2017) <https://ieeexplore.ieee.org/document/7895177>`_, producing radial MV/LV tree graphs with realistic cable types, lengths, and load/generation profiles.

Synthesised grids can be **exported to 12+ industry-standard formats** via `pandapower <https://www.pandapower.org/>`_ and `pypowsybl <https://pypowsybl.readthedocs.io/>`_ and validated with **DC and AC power-flow solvers** from both libraries.

The goal of the project is to provide synthetic yet realistic power grids for grid modeling, simulation and analysis, with the ultimate goal of building **Foundation Models** for power grids. This project is supported by `AI-EFFECT <https://ai-effect.eu/>`_ (Artificial Intelligence Experimentation Facility For the Energy Sector).


Transmission Grid Synthesis
----------------------------

Synthesis Pipeline
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 5 30 65
   :header-rows: 1

   * - Step
     - Module
     - Method
   * - 1
     - Topology generation
     - CLC graph model with prescribed degree distribution and diameter per voltage level; k-stars inter-level transformer model
   * - 2
     - Bus-type assignment
     - Entropy-based Artificial Immune System (AIS) optimisation reproducing empirical 2-D joint distribution of bus types and node degrees
   * - 3
     - Generation capacity
     - Exponential/extreme-value sampling with capacity–degree correlation via 2-D PMF table
   * - 4
     - Load allocation
     - Empirical 2-D probability table matching load–degree joint distribution; reactive loads via power-factor model
   * - 5
     - Generation dispatch
     - Three-category partitioning (uncommitted / partially committed / fully committed) with 2-D bin matching and iterative balancing
   * - 6
     - Transmission lines
     - LogNormal impedance magnitudes, Lévy-stable line angles, DCPF-based swapping, exponential gauge-ratio assignment via 2-D table


Quick Start
~~~~~~~~~~~

The high-level ``synthesize()`` function runs the entire CLC transmission pipeline in one call:

.. code-block:: python

   from powergrid_synth import synthesize

   # Mode I — clone an existing grid's statistical profile
   grid = synthesize(
       mode="reference",
       reference_case="case118",
       seed=42,
       export_formats=["json", "cgmes", "matpower"],
   )

   # Mode II — fully synthetic from user-specified parameters
   grid = synthesize(
       mode="synthetic",
       level_specs=[
           {"n": 50,  "avg_k": 3.5, "diam": 10, "dist_type": "dgln"},
           {"n": 150, "avg_k": 2.5, "diam": 15, "dist_type": "dpl"},
           {"n": 300, "avg_k": 2.0, "diam": 20, "dist_type": "poisson"},
       ],
       connection_specs={
           (0, 1): {"type": "k-stars", "c": 0.174, "gamma": 4.15},
           (1, 2): {"type": "k-stars", "c": 0.15,  "gamma": 4.15},
       },
       seed=42,
       export_formats=["json", "matpower"],
   )

See :doc:`examples/Synthesize.nblink` for a full walkthrough.

Step-by-Step Usage
~~~~~~~~~~~~~~~~~~

For fine-grained control over individual pipeline stages:

.. code-block:: python

   from powergrid_synth import (
       InputConfigurator, PowerGridGenerator, BusTypeAllocator,
       CapacityAllocator, LoadAllocator, GenerationDispatcher,
       TransmissionLineAllocator, GridVisualizer,
   )

   # 1. Configure voltage levels and inter-level connections
   level_specs = [
       {'n': 50,  'avg_k': 3.5, 'diam': 10, 'dist_type': 'dgln'},
       {'n': 150, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'dpl'},
       {'n': 300, 'avg_k': 2.0, 'diam': 20, 'dist_type': 'poisson'},
   ]
   connection_specs = {
       (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
       (1, 2): {'type': 'k-stars', 'c': 0.15,  'gamma': 4.15},
   }

   config = InputConfigurator(seed=42)
   params = config.create_params(level_specs, connection_specs)

   # 2. Generate topology (CLC model)
   gen = PowerGridGenerator(seed=42)
   grid = gen.generate_grid(
       degrees_by_level=params['degrees_by_level'],
       diameters_by_level=params['diameters_by_level'],
       transformer_degrees=params['transformer_degrees'],
   )

   # 3. Assign bus types, generation/load, dispatch, and line parameters
   BusTypeAllocator(grid).allocate()
   CapacityAllocator(grid).allocate()
   LoadAllocator(grid).allocate()
   GenerationDispatcher(grid).dispatch()
   TransmissionLineAllocator(grid).allocate()

   # 4. Visualize
   GridVisualizer().plot_grid(grid, layout='yifan_hu', title="Synthetic Grid")

See the :doc:`examples/index` for Jupyter notebooks covering each step in detail.


Distribution Grid Synthesis
----------------------------

Quick Start
~~~~~~~~~~~

The high-level ``synthesize_distribution()`` function generates realistic radial MV/LV feeders in one call:

.. code-block:: python

   from powergrid_synth import synthesize_distribution

   # Mode I — fit parameters from a reference distribution network
   feeders = synthesize_distribution(
       mode="reference",
       reference_case="cigre_lv",
       n_feeders=5,
       n_nodes=20,
       total_load_mw=0.5,
       seed=42,
       output_dir="output",
       export_formats=["json"],
   )

   # Mode II — use default Table III parameters (no reference needed)
   feeders = synthesize_distribution(
       mode="default",
       n_feeders=10,
       n_nodes=30,
       total_load_mw=0.8,
       seed=7,
   )

See :doc:`examples/DistributionSynth.nblink` and :doc:`examples/DistributionSynthFromRef.nblink` for detailed walkthroughs.


Protecting the Reference Grid
------------------------------

In reference mode the synthesiser reads statistics from a real network. When
that network is confidential, ``perturbation_config`` controls how much of it
reaches the synthetic output. Each registered input is handled in one of three
modes: ``raw`` (read and used unchanged), ``perturb`` (read and displaced by
calibrated noise), or ``off`` (never reaches the output; a public value is used
instead).

.. code-block:: python

   from powergrid_synth import synthesize

   grid = synthesize(
       mode="reference",
       reference_case="case118",
       seed=42,
       perturbation_config={
           "strength": 1.0,
           "seed": 7,                      # fix per release; see below
           "parameters": {
               "degrees_by_level": {"mode": "perturb"},
               "node_count": {"mode": "perturb"},
               "base_kv_map": {"mode": "off"},
           },
       },
   )

   # What was protected, and what was not.
   print(grid.graph["perturbation_report"])

A TOML file or a ``PerturbationSettings`` object works in place of the mapping;
``perturbation.example.toml`` in the repository root is a worked example. The
same argument and the same file work for ``synthesize_distribution()``.

**Every parameter defaults to** ``raw``, so passing nothing reproduces the
unprotected behaviour exactly.

Every run writes an audit record --- mode, strength, noise scale, resolved seed
and any fallback, per parameter --- next to the exported grid as
``<output_name>_perturbation_report.json``. Retain it alongside a release.

.. warning::

   Noise is drawn once per run, so pooling grids *within* one release cannot
   average it away. Publishing several batches from the same reference with
   fresh entropy breaks that property: fix the seed so every batch carries the
   identical offset.

Protection is evidenced by measurement, not by a theorem, and the measurements
establish it for some parameters and not others. See :doc:`theory/perturbation`
for the mechanism, the threat model, the registered parameters, and how to
measure recovery against your own reference.


Supported Data Formats & Power-Flow Solvers
--------------------------------------------

Synthetic grids live as **NetworkX graphs** internally and can be converted / exported via
``GridExporter`` and ``data_format_converter``.

**Export formats**

.. list-table::
   :widths: 15 30 55
   :header-rows: 1

   * - Via
     - Format
     - Method
   * - pandapower
     - JSON, Excel, SQLite, Pickle
     - ``to_json()``, ``to_excel()``, ``to_sqlite()``, ``to_pickle()``
   * - pypowsybl
     - **CGMES, XIIDM, MATPOWER, PSS/E**, UCTE, AMPL, BIIDM, JIIDM
     - ``to_cgmes()``, ``to_matpower()``, ``to_psse()``, ``to_pypowsybl(format=...)``

**Conversion chain**::

    Export:  NetworkX  ↔  pandapower  →  pypowsybl  →  [CGMES / XIIDM / MATPOWER / PSS·E / …]
    Import:  [CGMES / XIIDM / MATPOWER / PSS·E / …]  →  pypowsybl  →  NetworkX  (via load_grid / pypowsybl_to_nx)

Converter functions: ``pandapower_to_nx``, ``nx_to_pandapower``,
``pandapower_to_pypowsybl``, ``pypowsybl_to_nx``, ``load_grid``
(in ``data_format_converter.py``).

**Power-flow solvers**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - Solver
     - Library
     - Type
     - Call
   * - Newton-Raphson AC
     - pandapower
     - AC
     - ``pp.runpp(net)``
   * - Linear DC
     - pandapower
     - DC
     - ``pp.rundcpp(net)``
   * - AC load-flow
     - pypowsybl
     - AC
     - ``pypowsybl.loadflow.run_ac(net)``
   * - DC load-flow
     - pypowsybl
     - DC
     - ``pypowsybl.loadflow.run_dc(net)``
   * - Built-in DCPF
     - powergrid_synth
     - DC
     - ``DCPowerFlow(graph).run()``


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   Getting Started  <README.md>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   Examples  <examples/index>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   Theory  <theory/index>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   API Reference <autoapi/index>