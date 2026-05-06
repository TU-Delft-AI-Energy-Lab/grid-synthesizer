import pytest
import numpy as np
from powergrid_synth.transmission.input_configurator import InputConfigurator

class TestInputConfig:
    def setup_method(self):
        self.config = InputConfigurator(seed=42)

    def test_poisson_generation(self):
        """Test basic degree list generation."""
        n = 100
        avg = 3.0
        degrees = self.config._generate_poisson_degrees(n, avg)
        
        assert len(degrees) == n
        assert abs(np.mean(degrees) - avg) < 1.0 # Rough statistical check

    def test_k_stars_generation(self):
        """Test the k-stars transformer logic."""
        n_i = 50
        n_j = 100
        c = 0.174
        gamma = 4.15
        
        t_i, t_j = self.config._generate_transformer_stars(n_i, n_j, c, gamma)
        
        assert len(t_i) == n_i
        assert len(t_j) == n_j
        
        # Bipartite constraint check:
        # Sum of degrees leaving I must equal sum of degrees leaving J
        assert sum(t_i) == sum(t_j)

    def test_create_params_structure(self):
        """Test the full dictionary creation."""
        levels = [{'n': 10, 'avg_k': 2, 'diam': 3}]
        conns = {}
        
        params = self.config.create_params(levels, conns)
        
        assert 'degrees_by_level' in params
        assert 'diameters_by_level' in params
        assert 'transformer_degrees' in params
        assert len(params['degrees_by_level']) == 1

    def test_generate_optimized_degrees_dgln(self):
        degrees = self.config._generate_optimized_degrees(
            n_nodes=50, avg_degree=3.0, max_degree=15, dist_type='dgln'
        )
        assert len(degrees) == 50
        assert all(1 <= d <= 15 for d in degrees)

    def test_generate_optimized_degrees_dpl(self):
        degrees = self.config._generate_optimized_degrees(
            n_nodes=50, avg_degree=3.0, max_degree=15, dist_type='dpl'
        )
        assert len(degrees) == 50
        assert all(1 <= d <= 15 for d in degrees)

    def test_generate_transformer_simple(self):
        result = self.config._generate_transformer_simple(n_nodes=100, prob=0.3)
        assert len(result) == 100
        assert all(v in (0, 1) for v in result)

    def test_create_params_with_kstars_connection(self):
        levels = [
            {'n': 20, 'avg_k': 2.5, 'diam': 4, 'dist_type': 'poisson'},
            {'n': 30, 'avg_k': 2.0, 'diam': 5, 'dist_type': 'poisson'},
        ]
        conns = {
            (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15}
        }
        params = self.config.create_params(levels, conns)
        assert (0, 1) in params['transformer_degrees']
        t_i, t_j = params['transformer_degrees'][(0, 1)]
        assert len(t_i) == 20
        assert len(t_j) == 30

    def test_create_params_with_simple_connection(self):
        levels = [
            {'n': 20, 'avg_k': 2.5, 'diam': 4, 'dist_type': 'poisson'},
            {'n': 30, 'avg_k': 2.0, 'diam': 5, 'dist_type': 'poisson'},
        ]
        conns = {
            (0, 1): {'type': 'simple', 'p_i_j': 0.2, 'p_j_i': 0.1}
        }
        params = self.config.create_params(levels, conns)
        assert (0, 1) in params['transformer_degrees']
        t_i, t_j = params['transformer_degrees'][(0, 1)]
        assert len(t_i) == 20
        assert len(t_j) == 30
        assert all(v in (0, 1) for v in t_i)
        assert all(v in (0, 1) for v in t_j)

    def test_create_params_multiple_levels(self):
        levels = [
            {'n': 15, 'avg_k': 2.0, 'diam': 3, 'dist_type': 'poisson'},
            {'n': 25, 'avg_k': 2.5, 'diam': 4, 'dist_type': 'poisson'},
            {'n': 10, 'avg_k': 2.0, 'diam': 3, 'dist_type': 'poisson'},
        ]
        conns = {
            (0, 1): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15},
            (1, 2): {'type': 'simple', 'p_i_j': 0.15, 'p_j_i': 0.1},
        }
        params = self.config.create_params(levels, conns)
        assert len(params['degrees_by_level']) == 3
        assert len(params['diameters_by_level']) == 3
        assert (0, 1) in params['transformer_degrees']
        assert (1, 2) in params['transformer_degrees']

    def test_create_params_invalid_connection_index(self):
        levels = [{'n': 10, 'avg_k': 2, 'diam': 3}]
        conns = {
            (0, 5): {'type': 'k-stars', 'c': 0.174, 'gamma': 4.15}
        }
        params = self.config.create_params(levels, conns)
        assert (0, 5) not in params['transformer_degrees']

    def test_poisson_degrees_all_positive_or_zero(self):
        degrees = self.config._generate_poisson_degrees(200, 2.0)
        assert all(d >= 0 for d in degrees)