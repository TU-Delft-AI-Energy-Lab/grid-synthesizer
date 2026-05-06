# API Reference

## Transmission Grid Synthesis

### High-Level Function

```{eval-rst}
.. autofunction:: powergrid_synth.synthesize
```

### Topology Generation

```{eval-rst}
.. autoclass:: powergrid_synth.InputConfigurator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.PowerGridGenerator
   :members:
   :undoc-members:
   :show-inheritance:
```

### Electrical Assignment

```{eval-rst}
.. autoclass:: powergrid_synth.BusTypeAllocator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.CapacityAllocator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.LoadAllocator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.GenerationDispatcher
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.TransmissionLineAllocator
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Distribution Grid Synthesis

### High-Level Function

```{eval-rst}
.. autofunction:: powergrid_synth.synthesize_distribution
```

### Core Classes

```{eval-rst}
.. autoclass:: powergrid_synth.distribution.SchweetzerFeederGenerator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.distribution.FeederParams
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.distribution.DistributionSynthParams
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.distribution.DistributionInputModel
   :members:
   :undoc-members:
   :show-inheritance:
```

### Analysis & Conversion

```{eval-rst}
.. autofunction:: powergrid_synth.distribution.fit_params_from_feeders
```

```{eval-rst}
.. automodule:: powergrid_synth.distribution.distribution_converter
   :members:
   :undoc-members:
```

---

## Graph Classes

```{eval-rst}
.. autoclass:: powergrid_synth.PowerGridGraph
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.TransmissionGrid
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.DistributionGrid
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Analysis, Export & Visualisation

```{eval-rst}
.. autoclass:: powergrid_synth.GraphComparator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.GridVisualizer
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.GridExporter
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Format Converters

```{eval-rst}
.. autofunction:: powergrid_synth.pandapower_to_nx
```

```{eval-rst}
.. autofunction:: powergrid_synth.nx_to_pandapower
```

```{eval-rst}
.. autofunction:: powergrid_synth.pandapower_to_pypowsybl
```

```{eval-rst}
.. autofunction:: powergrid_synth.pypowsybl_to_nx
```

```{eval-rst}
.. autofunction:: powergrid_synth.load_grid
```
