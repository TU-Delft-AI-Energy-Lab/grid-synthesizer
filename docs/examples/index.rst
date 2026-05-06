.. include:: intro.rst

Each notebook is also available as a self-contained **Google Colab** version with
package installation included — click the |colab| badge to open directly in your browser.

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :height: 18px

Transmission Grid Synthesis
---------------------------

The **recommended starting point** is the high-level ``synthesize()`` notebook, which runs the
entire pipeline in a single function call.  The remaining notebooks provide step-by-step
control over individual pipeline stages.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Notebook
     - Open in Colab
   * - :doc:`High-Level Synthesis <Synthesize.nblink>`
     - (recommended starting point)
   * - :doc:`Synthesis with pypowsybl <SynthesizePypowsybl.nblink>`
     -
   * - :doc:`Topology Generation <TopologyGeneration.nblink>`
     - |colab_topo|
   * - :doc:`Bus Type Assignment <BusTypeAssignment.nblink>`
     - |colab_bus|
   * - :doc:`Generation and Load Settings <GenLoadSettings.nblink>`
     - |colab_gen|
   * - :doc:`IEEE Test <IEEETest.nblink>`
     - |colab_ieee|
   * - :doc:`PEGASE 9241 Test <Pegase9241Test.nblink>`
     - |colab_pegase|
   * - :doc:`RTE 7000 Test <RTE7000Test.nblink>`
     - |colab_rte7k|



.. |colab_topo| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/TopologyGeneration_colab.ipynb

.. |colab_bus| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/BusTypeAssignment_colab.ipynb

.. |colab_gen| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/GenLoadSettings_colab.ipynb

.. |colab_ieee| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/ieee_test_colab.ipynb

.. |colab_pegase| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/pegase9241_test_colab.ipynb

.. |colab_rte7k| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/TU-Delft-AI-Energy-Lab/grid-synthesizer/blob/main/examples/colab/rte7k_from_source_file_colab.ipynb

Here is a brief example showing how to configure the user input, and then generate, and visualize a synthetic power grid.

.. code-block:: python
   :linenos:

    from powergrid_synth.generator import PowerGridGenerator
    from powergrid_synth.input_configurator import InputConfigurator
    from powergrid_synth.visualization import GridVisualizer

    # 1. Configuration
    # Define the voltage levels (Node count, Avg Degree, Diameter, Distribution Type)
    level_specs = [
        {'n': 50,  'avg_k': 3.5, 'diam': 10, 'dist_type': 'dgln'},    # Backbone (Log-Normal)
        {'n': 150, 'avg_k': 2.5, 'diam': 15, 'dist_type': 'dpl'},     # Distribution (Power Law)
        {'n': 300, 'avg_k': 2.0, 'diam': 20, 'dist_type': 'poisson'}  # Local (Poisson)
    ]

    # Define connections between levels (k-stars model)
    connection_specs = {
        (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
        (1, 2): {'type': 'k-stars', 'c': 0.15, 'gamma': 4.15}
    }

    # 2. Generate Parameters
    config = InputConfigurator(seed=42)
    params = config.create_params(level_specs, connection_specs)

    # 3. Generate Topology
    gen = PowerGridGenerator(seed=42)
    grid = gen.generate_grid(
        degrees_by_level=params['degrees_by_level'],
        diameters_by_level=params['diameters_by_level'],
        transformer_degrees=params['transformer_degrees']
    )

Distribution Grid Synthesis
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Notebook
     - Description
   * - :doc:`Distribution Feeder Synthesis <DistributionSynth.nblink>`
     - Schweitzer algorithm with default/custom parameters
   * - :doc:`Distribution Synthesis from Reference Grid <DistributionSynthFromRef.nblink>`
     - Fit parameters from a pandapower network
   * - :doc:`Distribution Synthesis with pypowsybl <DistributionSynthPypowsybl.nblink>`
     - Load CIGRE LV via CGMES, run full pipeline + one-liner


.. toctree::
   :maxdepth: 1

   transmission
   distribution