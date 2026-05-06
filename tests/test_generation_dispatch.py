import pytest
import numpy as np
import networkx as nx
from powergrid_synth.transmission.generation_dispatcher import GenerationDispatcher

@pytest.fixture
def sample_grid():
    G = nx.Graph()
    # 10 Generators, 100 MW each = 1000 MW total
    for i in range(10):
        G.add_node(i, bus_type='Gen', pg_max=100.0)
    
    # Loads
    G.add_node(10, bus_type='Load', pl=200.0)
    G.add_node(11, bus_type='Load', pl=200.0)
    # Total Load = 400 MW (40% Loading)
    return G

class TestGenerationDispatcher:

    def test_initialization(self, sample_grid):
        dispatcher = GenerationDispatcher(sample_grid, ref_sys_id=1)
        assert dispatcher.mu_committed is not None

    def test_select_uncommitted(self, sample_grid):
        dispatcher = GenerationDispatcher(sample_grid)
        norm_pg = np.array([[i, 1.0] for i in range(10)])
        
        uncomm, remaining = dispatcher._select_uncommitted(norm_pg)
        assert 1 <= len(uncomm) <= 2
        assert uncomm.shape[1] == 3
        assert np.all(uncomm[:, 2] == 0.0)

    def test_select_committed(self, sample_grid):
        dispatcher = GenerationDispatcher(sample_grid)
        norm_pg = np.array([[i, 1.0] for i in range(8)])
        
        # Pass total units count (10) explicitly
        comm, remaining = dispatcher._select_committed(norm_pg, 10)
        # 40-50% of 10 is 4-5. 
        # But random factors apply. Just check bounds.
        assert 3 <= len(comm) <= 6

    def test_dispatch_balancing_light(self, sample_grid):
        """Test dispatch with 400 MW load (Excess scenario initial)."""
        dispatcher = GenerationDispatcher(sample_grid)
        result = dispatcher.dispatch()
        
        total_gen = sum(result.values())
        total_load = 400.0
        
        print(f"Target: {total_load}, Actual: {total_gen}")
        assert abs(total_gen - total_load) < 0.1 * total_load

    def test_dispatch_balancing_heavy(self, sample_grid):
        """Test dispatch with 800 MW load (Deficit scenario initial)."""
        sample_grid.nodes[10]['pl'] = 400.0
        sample_grid.nodes[11]['pl'] = 400.0 
        
        dispatcher = GenerationDispatcher(sample_grid)
        result = dispatcher.dispatch()
        
        total_gen = sum(result.values())
        total_load = 800.0
        
        print(f"Target: {total_load}, Actual: {total_gen}")
        assert abs(total_gen - total_load) < 0.1 * total_load

    def test_dispatch_no_generators(self):
        """dispatch returns empty dict when grid has no Gen nodes."""
        grid = nx.Graph()
        for i in range(5):
            grid.add_node(i, bus_type='Load', pl=10.0)
        dispatcher = GenerationDispatcher(grid, ref_sys_id=1)
        result = dispatcher.dispatch()
        assert result == {}

    def test_select_uncommitted_empty(self, sample_grid):
        """_select_uncommitted returns empty arrays for empty input."""
        dispatcher = GenerationDispatcher(sample_grid, ref_sys_id=1)
        empty = np.array([]).reshape(0, 2)
        uncomm, remaining = dispatcher._select_uncommitted(empty)
        assert len(uncomm) == 0
        assert len(remaining) == 0

    def test_generate_alphas_alpha_mod_nonzero(self, sample_grid):
        """_generate_alphas with alpha_mod != 0 produces 99.5% positive, 0.5% negative."""
        dispatcher = GenerationDispatcher(sample_grid, ref_sys_id=1)
        dispatcher.alpha_mod = 1
        np.random.seed(42)
        n_comm = 200
        alphas = dispatcher._generate_alphas(n_comm)
        assert len(alphas) == n_comm
        assert np.any(alphas < 0)

    def test_generate_alphas_zero_count(self, sample_grid):
        """_generate_alphas with n_comm=0 returns empty array."""
        dispatcher = GenerationDispatcher(sample_grid, ref_sys_id=1)
        alphas = dispatcher._generate_alphas(0)
        assert len(alphas) == 0

    def test_generate_alphas_alpha_mod_nonzero_small(self, sample_grid):
        """_generate_alphas with alpha_mod != 0 and small n_comm hits n_005=0 path."""
        dispatcher = GenerationDispatcher(sample_grid, ref_sys_id=1)
        dispatcher.alpha_mod = 1
        # n_comm=1: n_995=round(1*0.995)=1, n_005=0 → returns a1 directly (line 198)
        alphas = dispatcher._generate_alphas(1)
        assert len(alphas) == 1
        assert alphas[0, 0] >= 0  # no negative dispatch for such a small set

    def test_invalid_ref_sys_fallback(self):
        """Invalid ref_sys_id falls back to ref_sys_id=1 without error."""
        grid = nx.Graph()
        grid.add_node(0, bus_type='Load', pl=10.0)
        dispatcher = GenerationDispatcher(grid, ref_sys_id=99)
        assert dispatcher.stats is not None