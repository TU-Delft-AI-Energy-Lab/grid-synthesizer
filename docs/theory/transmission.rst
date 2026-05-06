Transmission Grid Synthesis
===========================

Transmission grids are synthesized by first generating a multi-level topology using the Chung-Lu-Chain (CLC) random graph model, then assigning bus types (generator, load, connection) and allocating active/reactive power capacities based on empirical statistics from real grids. Finally, generation is dispatched and transmission-line impedances and thermal limits are drawn from reference distributions.

.. toctree::
   :maxdepth: 1

   Topology Generation <topology_generation.rst>
   Bus Type Assignment <bus_type_assignment.rst>
   Generation and Load Settings <generation_load_settings.rst>
   Generation Dispatch and Transmission Capacity <gen_dispatch_transmission.rst>
