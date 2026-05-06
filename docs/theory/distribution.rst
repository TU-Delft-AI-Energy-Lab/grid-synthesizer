Distribution Grid Synthesis
===========================

Distribution feeders are synthesized as radial MV/LV tree graphs using the algorithm of :cite:p:`schweitzer2017automated`, which stochastically grows branches with realistic cable types, lengths, and load/generation profiles. Parameters can be fitted from reference pandapower networks or specified directly.

.. toctree::
   :maxdepth: 1

   Distribution Feeder Generation <distribution_feeder_generation.rst>

.. bibliography::
   :filter: docname in docnames