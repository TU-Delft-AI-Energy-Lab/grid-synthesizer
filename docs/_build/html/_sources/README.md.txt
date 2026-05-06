# Getting Started

**Source code:** [GitHub repository](https://github.com/cookbook-ms/Chung_Lu_Chain-synthesizer)

## Installation

### From PyPI

```bash
pip install powergridsynth
```

### From source (development mode)

1. Clone the repository:
```bash
git clone https://github.com/cookbook-ms/Chung_Lu_Chain-synthesizer.git
cd Chung_Lu_Chain-synthesizer
```

2. Create and activate a virtual environment (e.g. with `uv`):
```bash
uv venv powergrid
source powergrid/bin/activate
```

3. Install in editable mode with development dependencies:
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
| **2. Bus-type assignment** | `bus_type_allocator.py` | Entropy-based AIS optimisation reproducing empirical 2-D joint distribution of bus types and node degrees |
| **3. Generation capacity** | `capacity_allocator.py` | Exponential/extreme-value sampling with capacity–degree correlation via 2-D PMF table |
| **4. Load allocation** | `load_allocator.py` | Empirical 2-D probability table matching load–degree joint distribution; reactive loads via power-factor model |
| **5. Generation dispatch** | `generation_dispatcher.py` | Three-category partitioning with 2-D bin matching and iterative balancing |
| **6. Transmission lines** | `transmission.py`, `dcpf.py` | LogNormal impedance magnitudes, Lévy-stable line angles, DCPF-based swapping, exponential gauge-ratio assignment |

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

### Power-Flow Solvers

| Solver | Library | Type | Call |
|--------|---------|------|------|
| Newton-Raphson AC | **pandapower** | AC | `pp.runpp(net)` |
| Linear DC | **pandapower** | DC | `pp.rundcpp(net)` |
| AC load-flow | **pypowsybl** | AC | `pypowsybl.loadflow.run_ac(net)` |
| DC load-flow | **pypowsybl** | DC | `pypowsybl.loadflow.run_dc(net)` |
| Built-in DCPF | **powergrid_synth** | DC | `DCPowerFlow(graph).run()` |

## Testing

```bash
pytest tests/ -v
```

## Building Documentation

```bash
cd docs
make html
```
