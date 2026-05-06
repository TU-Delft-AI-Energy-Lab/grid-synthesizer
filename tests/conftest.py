import pytest
import numpy as np
import networkx as nx

@pytest.fixture(autouse=True)
def set_random_seed():
    """
    Automatically resets random seeds before every test function 
    to ensure deterministic and reproducible tests.
    """
    np.random.seed(42)
    # If modules use their own RNG instances, this global seed might not catch everything,
    # but the package components generally allow passing a seed or use numpy global.
    yield


@pytest.fixture(scope="session")
def fully_parameterised_grid():
    """
    Generate a small fully-parameterised grid once per test session.
    This grid has bus types, generation capacities, loads, dispatch,
    and transmission-line parameters assigned — suitable for testing
    converters, exporters, and the DCPF solver.
    """
    from powergrid_synth.transmission.input_configurator import InputConfigurator
    from powergrid_synth.transmission.generator import PowerGridGenerator
    from powergrid_synth.transmission.bus_type_allocator import BusTypeAllocator
    from powergrid_synth.transmission.capacity_allocator import CapacityAllocator
    from powergrid_synth.transmission.load_allocator import LoadAllocator
    from powergrid_synth.transmission.generation_dispatcher import GenerationDispatcher
    from powergrid_synth.transmission.transmission import TransmissionLineAllocator

    np.random.seed(42)
    configurator = InputConfigurator(seed=42)
    level_specs = [
        {"n": 30, "avg_k": 2.5, "diam": 6, "dist_type": "poisson"},
        {"n": 40, "avg_k": 2.0, "diam": 10, "dist_type": "poisson"},
    ]
    connection_specs = {
        (0, 1): {"type": "k-stars", "c": 0.174, "gamma": 4.15},
    }
    params = configurator.create_params(level_specs, connection_specs)

    gen = PowerGridGenerator(seed=42)
    grid = gen.generate_grid(
        params["degrees_by_level"],
        params["diameters_by_level"],
        params["transformer_degrees"],
        keep_lcc=True,
    )

    bus_types = BusTypeAllocator(grid).allocate(max_iter=10)
    nx.set_node_attributes(grid, bus_types, name="bus_type")

    caps = CapacityAllocator(grid, ref_sys_id=1).allocate()
    nx.set_node_attributes(grid, caps, name="pg_max")

    loads = LoadAllocator(grid, ref_sys_id=1).allocate(loading_level="M")
    nx.set_node_attributes(grid, loads, name="pl")

    dispatch = GenerationDispatcher(grid, ref_sys_id=1).dispatch()
    nx.set_node_attributes(grid, dispatch, name="pg")

    TransmissionLineAllocator(grid, ref_sys_id=1).allocate()

    return grid