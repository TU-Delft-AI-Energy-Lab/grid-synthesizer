# PowerGridSynth

![CI Status](https://github.com/TU-Delft-AI-Energy-Lab/grid-synthesizer/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Python package for generating realistic **synthetic power grids** at both the transmission and distribution level.

- **Transmission grids** are synthesized using a [Chung-Lu-Chain (CLC)](https://arxiv.org/abs/1711.11098) graph model that reproduces prescribed degree distributions and diameters across multiple voltage levels, then layers on bus-type assignment, generation/load capacity allocation (active *and* reactive power), generation dispatch, and transmission-line impedance/capacity assignment drawn from empirical statistics of real grids (NYISO, WECC).
- **Distribution feeders** are synthesized using the algorithm of [Schweitzer et al. (2017)](https://ieeexplore.ieee.org/document/7895177), producing radial MV/LV tree graphs with realistic cable types, lengths, and load/generation profiles.

Synthesised grids can be **exported to 12+ industry-standard formats** via [pandapower](https://www.pandapower.org/) and [pypowsybl](https://pypowsybl.readthedocs.io/) and validated with **DC and AC power-flow solvers** from both libraries.

## Documentation

Full documentation (theory, API reference, tutorials): [power-grid-synthesizer.readthedocs.io](https://power-grid-synthesizer.readthedocs.io/en/latest/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TU-Delft-AI-Energy-Lab/grid-synthesizer.git
cd Chung_Lu_Chain-synthesizer
```
2. Create and activate a virtual environment (e.g. with `uv`):
```bash
uv venv powergrid_studio
source powergrid_studio/bin/activate
```
3. Install in editable mode:
```bash
uv pip install -e ".[dev]"
```

For one-call synthesis, converters, exporters, and power-flow integrations, install the optional extras as well:

```bash
uv pip install -e ".[dev,export]"
```

Core dependencies: `numpy`, `scipy`, `networkx`, `matplotlib`, `pandas`.

Optional export/power-flow dependencies: `pandapower`, `numba`, `lightsim2grid`, `pypowsybl`, `pypowsybl-jupyter`.

---

## Transmission Grid Synthesis

### Quick Start

The high-level `synthesize()` function runs the entire CLC transmission pipeline in one call:

```python
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
```

See `examples/transmission/Synthesize.ipynb` for a full walkthrough.

### Step-by-Step Usage

For fine-grained control over each pipeline stage:

```python
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
```

### Synthesis Pipeline

| Step | Module | Method |
|------|--------|--------|
| **1. Topology generation** | `generator.py`, `preprocessing.py`, `edge_creation.py`, `transformer_edges.py` | CLC graph model with prescribed degree distribution and diameter per voltage level; k-stars inter-level transformer model |
| **2. Bus-type assignment** | `bus_type_allocator.py` | Entropy-based Artificial Immune System (AIS) optimisation reproducing empirical 2-D joint distribution of bus types and node degrees |
| **3. Generation capacity** | `capacity_allocator.py` | Exponential/extreme-value sampling with capacity–degree correlation via 2-D PMF table |
| **4. Load allocation** | `load_allocator.py` | Empirical 2-D probability table matching load–degree joint distribution; reactive loads via power-factor model |
| **5. Generation dispatch** | `generation_dispatcher.py` | Three-category partitioning (uncommitted / partially committed / fully committed) with 2-D bin matching and iterative balancing |
| **6. Transmission lines** | `transmission.py`, `dcpf.py` | LogNormal impedance magnitudes, Lévy-stable line angles, DCPF-based swapping, exponential gauge-ratio assignment via 2-D table, optional topology refinement |

Statistical parameters are stored in `reference_data.py` for reference systems (NYISO-2935, WECC-16994).

### Modules

| Module | Description |
|--------|-------------|
| `generator.py` | Main orchestrator — `PowerGridGenerator` |
| `preprocessing.py` | Backbone setup and spatial-box assignment (Algorithm 1) |
| `edge_creation.py` | Intra-level edge creation via CLC model (Algorithm 2) |
| `transformer_edges.py` | Inter-level connections via k-stars model (Algorithm 3) |
| `input_configurator.py` | High-level specification → detailed input parameters |
| `deg_dist_optimizer.py` | Optimises distribution parameters (DGLN α,β / Power-Law γ) to match target average degree |
| `bus_type_allocator.py` | AIS-based Generator / Load / Connection bus-type assignment |
| `capacity_allocator.py` | Generation capacity (Pg_max) allocation |
| `load_allocator.py` | Active-power load (Pl) and reactive-power load (Ql) allocation |
| `generation_dispatcher.py` | Dispatch factor (α) assignment and generation–load balancing |
| `transmission.py` | Branch impedance (R, X) and thermal capacity (F_max) allocation |
| `dcpf.py` | Lightweight DC power-flow solver |
| `reference_data.py` | Pre-computed empirical statistics for NYISO, WECC reference systems |

### Examples

| Notebook | Topic | Colab |
|----------|-------|-------|
| `examples/transmission/Synthesize.ipynb` | **High-level synthesis** — full pipeline in one call (Mode I & II) | |
| `examples/transmission/SynthesizePypowsybl.ipynb` | **pypowsybl formats** — load IEEE-118 from built-in and CGMES | |
| `examples/transmission/TopologyGeneration.ipynb` | CLC topology generation and input configuration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/TopologyGeneration_colab.ipynb) |
| `examples/transmission/BusTypeAssignment.ipynb` | AIS bus-type allocation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/BusTypeAssignment_colab.ipynb) |
| `examples/transmission/GenLoadSettings.ipynb` | Generation capacity and load allocation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/GenLoadSettings_colab.ipynb) |
| `examples/transmission/ieee_test.ipynb` | Validation against IEEE118 test cases | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/ieee_test_colab.ipynb) |
| `examples/tranmission/pegase9241_test.ipynb` | Large-scale test | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/pegase9241_test_colab.ipynb) |
| `examples/tranmission/rte7k_from_source_file.ipynb` | Large-scale test using RTE7000 data | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/rte7k_from_source_file_colab.ipynb) |

---

## Distribution Grid Synthesis

### Quick Start

The high-level `synthesize_distribution()` function generates realistic radial MV/LV feeders in one call:

```python
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
```

See `examples/distribution/DistributionSynth.ipynb` and `examples/distribution/DistributionSynthFromRef.ipynb` for detailed walkthroughs.

### Step-by-Step Usage

```python
from powergrid_synth import DistributionGrid
from powergrid_synth.distribution import (
    SchweetzerFeederGenerator,
    pandapower_to_feeders,
    fit_params_from_feeders,
)

# Option A: generate with default (Table III) parameters
gen = SchweetzerFeederGenerator(seed=42)
feeder = DistributionGrid.from_nx(
    gen.generate_feeder(n_nodes=50, total_load_mw=8.0, total_gen_mw=1.0)
)
print(feeder.max_hop, feeder.total_load_mw, feeder.is_radial)

# Option B: fit parameters from a real pandapower network
import pandapower.networks as pn
ref_feeders = pandapower_to_feeders(pn.create_cigre_network_lv())
params = fit_params_from_feeders(ref_feeders)
gen = SchweetzerFeederGenerator(params=params, seed=0)
feeder = DistributionGrid.from_nx(
    gen.generate_feeder(n_nodes=44, total_load_mw=0.7)
)
```

### Synthesis Pipeline

| Step | Description |
|------|-------------|
| **1. Tree topology** | Random radial tree generation with branching and hop-count constraints |
| **2. Cable types** | Assign cable types from a realistic catalogue (weighted by frequency) |
| **3. Cable lengths** | Sample cable lengths from a Cauchy distribution fitted to reference data |
| **4. Loads** | Distribute total load across buses |
| **5. Generation** | Assign distributed generation (PV, etc.) to buses |

### Modules

| Module | Description |
|--------|-------------|
| `distribution_synthesis.py` | `SchweetzerFeederGenerator` — generates a single radial feeder graph |
| `distribution_params.py` | `FeederParams` dataclass with default Table III values |
| `distribution_analysis.py` | `fit_params_from_feeders` — extract statistical parameters from reference feeders |
| `distribution_converter.py` | `pandapower_to_feeders`, `pypowsybl_to_feeders` — convert real networks to feeder graphs |
| `distribution_input_model.py` | Input configuration and validation |
| `distribution_validation.py` | Structural and electrical validation of generated feeders |
| `synthesize.py` | `synthesize_distribution` — one-call high-level function |

### Examples

| Notebook | Topic |
|----------|-------|
| `examples/distribution/DistributionSynth.ipynb` | **Feeder synthesis** — Schweitzer algorithm with default/custom parameters |
| `examples/distribution/DistributionSynthFromRef.ipynb` | **Synthesis from reference** — fit parameters from a pandapower network |
| `examples/distribution/DistributionSynthPypowsybl.ipynb` | **pypowsybl formats** — load CIGRE LV via CGMES, full pipeline + one-liner |

---

## Protecting the Reference Grid

In reference mode the synthesiser reads statistics from a real network. When
that network is confidential, `perturbation_config` controls how much of it
reaches the synthetic output. Each registered input is handled in one of three
modes:

| Mode | Behaviour |
|------|-----------|
| `raw` | Read from the reference and used unchanged. **Default for every parameter.** |
| `perturb` | Read and displaced by calibrated noise before use. |
| `off` | Never reaches the output; a public built-in value is used instead. |

```python
from powergrid_synth import synthesize

grid = synthesize(
    mode="reference",
    reference_case="case118",
    seed=42,
    perturbation_config={
        "strength": 1.0,
        "seed": 7,                      # fix per release, see the warning below
        "parameters": {
            "degrees_by_level": {"mode": "perturb"},
            "node_count": {"mode": "perturb"},
            "base_kv_map": {"mode": "off"},
        },
    },
)

print(grid.graph["perturbation_report"])   # what was protected, and what was not
```

A TOML path or a `PerturbationSettings` object works in place of the mapping —
see [`perturbation.example.toml`](perturbation.example.toml). The same argument
works for `synthesize_distribution()`.

Since every parameter defaults to `raw`, passing nothing reproduces the
unprotected behaviour exactly.

Every run writes an audit record — mode, strength, noise scale, resolved seed
and any fallback, per parameter — next to the exported grid as
`<output_name>_perturbation_report.json`. Retain it alongside a release.

> **Publish one batch per reference.** Noise is drawn once per run, so pooling
> grids *within* a release cannot average it away. Publishing several batches
> from the same reference with fresh entropy breaks that: fix the seed so every
> batch carries the identical offset.

Protection is evidenced by measurement, not by a theorem, and the measurements
establish it for some parameters and not others. Simulate a pooling adversary
against your own reference before relying on a configuration:

```bash
python scripts/measure_topology_recovery.py --reference-case case118 --n-grids 30
python scripts/make_recovery_figure.py
```

See the [theory page](https://power-grid-synthesizer.readthedocs.io/en/latest/theory/perturbation.html)
for the mechanism, the threat model, the registered parameters, and what the
measurements do and do not establish.

### Modules

| Module | Description |
|--------|-------------|
| `privacy/registry.py` | `REGISTRY` — every reference-derived input, its shape, transform and permitted modes |
| `privacy/settings.py` | `PerturbationSettings` — configuration parsing, validation and mode resolution |
| `privacy/engine.py` | Shape-dispatched perturbation: scalar families, probability vectors, integers, degree sequences, graph node counts |
| `privacy/noise.py` | Laplace draws and the logit / log / affine-logit transforms |
| `privacy/rng.py` | Per-stream seeding, so each parameter draws independently and reproducibly |
| `privacy/report.py` | `build_report` — the per-run audit record |
| `privacy/recovery.py` | Adversary estimators used to measure what perturbation actually protects |

---

## Shared Infrastructure

### Analysis, Export & Visualisation

| Module | Description |
|--------|-------------|
| `grid_graph.py` | `PowerGridGraph`, `TransmissionGrid`, `DistributionGrid` — NetworkX Graph subclasses with voltage-level filtering and radial-tree helpers |
| `analysis.py` | Topological metrics (diameter, clustering, path length) |
| `hierarchical_analysis.py` | Per-voltage-level analysis |
| `comparison.py` | Side-by-side comparison of synthetic vs. reference grids |
| `visualization.py` | Plotting routines (Yifan Hu layout, per-level views, interactive dashboards) |
| `exporter.py` | Export to **JSON, Excel, SQLite, Pickle** (pandapower) and **CGMES, XIIDM, MATPOWER, PSS/E, UCTE, AMPL** (pypowsybl) |
| `data_format_converter.py` | Conversion between NetworkX ↔ pandapower ↔ pypowsybl; import from CGMES/XIIDM/MATPOWER/PSS·E via `load_grid` |

### Export Formats

| Via | Format | Method |
|----|--------|--------|
| **pandapower** | JSON | `GridExporter.to_json()` |
| | Excel (.xlsx) | `GridExporter.to_excel()` |
| | SQLite | `GridExporter.to_sqlite()` |
| | Pickle | `GridExporter.to_pickle()` |
| **pypowsybl** | CGMES | `GridExporter.to_cgmes()` |
| | XIIDM | `GridExporter.to_pypowsybl(format='XIIDM')` |
| | MATPOWER | `GridExporter.to_matpower()` |
| | PSS/E | `GridExporter.to_psse()` |
| | UCTE, AMPL, BIIDM, JIIDM | `GridExporter.to_pypowsybl(format=...)` |

### Conversion Chain

```
Export:  NetworkX  ↔  pandapower  →  pypowsybl  →  [CGMES / XIIDM / MATPOWER / PSS·E / …]
Import:  [CGMES / XIIDM / MATPOWER / PSS·E / …]  →  pypowsybl  →  NetworkX  (via load_grid / pypowsybl_to_nx)
```

### Power-Flow Solvers

| Solver | Library | Type | Call |
|--------|---------|------|------|
| Newton-Raphson AC | **pandapower** | AC | `pp.runpp(net)` |
| Linear DC | **pandapower** | DC | `pp.rundcpp(net)` |
| AC load-flow | **pypowsybl** | AC | `pypowsybl.loadflow.run_ac(net)` |
| DC load-flow | **pypowsybl** | DC | `pypowsybl.loadflow.run_dc(net)` |
| Built-in DCPF | **powergrid_synth** | DC | `DCPowerFlow(graph).run()` |

See `examples/transmission/ieee_test.ipynb` for a full demonstration of export and power-flow validation.

## Testing

```bash
pytest tests/ -v
```

## References

1. Aksoy et al. (2018). "A Generative Graph Model for Electrical Infrastructure Networks." *Journal of Complex Networks*, 7(1), 128–162. [arXiv:1711.11098](https://arxiv.org/abs/1711.11098)
2. Elyas & Wang (2017). "Improved Synthetic Power Grid Modeling With Correlated Bus Type Assignments." *IEEE Trans. Power Syst.*, 32(5), 3391–3402. [DOI:10.1109/TPWRS.2016.2636165](https://ieeexplore.ieee.org/document/7763878)
3. Elyas et al. (2017). "On the Statistical Settings of Generation and Load in Synthetic Grid Modeling." [arXiv:1706.09294](https://arxiv.org/abs/1706.09294)
4. Sadeghian et al. (2018). "A Novel Algorithm for Statistical Assignment of Transmission Capacities in Synthetic Grid Modeling." *IEEE PESGM*. [DOI:10.1109/PESGM.2018.8585532](https://ieeexplore.ieee.org/document/8585532)
5. Schweitzer et al. (2017). "Automated Generation Models for Synthetic Power Distribution Grids." *IEEE Trans. Power Syst.*, 32(5), 3974–3985. [DOI:10.1109/TPWRS.2017.2694839](https://ieeexplore.ieee.org/document/7895177)

## License

MIT
