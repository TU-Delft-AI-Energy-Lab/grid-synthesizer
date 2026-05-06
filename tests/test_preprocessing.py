import pytest
import numpy as np
import math
from powergrid_synth.transmission.preprocessing import Preprocessor

class TestPreprocessor:
    def setup_method(self):
        self.prep = Preprocessor()

    def test_degree_inflation(self):
        """Test that degree inflation works and doesn't return fewer nodes."""
        initial_degrees = [3, 3, 2, 1, 1] # 5 nodes
        target_diameter = 4
        
        d_prime, v, D, S = self.prep.run_setup(initial_degrees, target_diameter)
        
        # Check that we didn't lose nodes
        assert len(d_prime) >= len(initial_degrees)
        
        # Check that we have a numpy array
        assert isinstance(d_prime, np.ndarray)

    def test_diameter_path_constraints(self):
        """Test that the Diameter Path (D) nodes are assigned to distinct boxes."""
        # Setup input
        initial_degrees = [4]*20 # 20 nodes with degree 4
        target_diameter = 5
        
        # --- Calculate EXPECTED diameter adjustment based on Algorithm 1 ---
        eta = len([x for x in initial_degrees if x > 0]) # 20
        term = 2 * math.log(eta / (target_diameter + 1)) # 2 * ln(20/6) ~= 2.407
        expected_delta = int(round(target_diameter - term)) # 5 - 2 = 3
        expected_delta = max(1, expected_delta)
        
        # Run Code
        d_prime, v, D, S = self.prep.run_setup(initial_degrees, target_diameter)
        
        # Assertion
        # The size of the diameter path set D should be equal to (adjusted_delta + 1)
        assert len(D) == expected_delta + 1, \
            f"Expected diameter set size {expected_delta + 1} (adjusted from {target_diameter}), but got {len(D)}"
        
        # Check distinct box assignment
        boxes_for_D = [v[i] for i in D]
        assert len(boxes_for_D) == len(set(boxes_for_D)), "Nodes in D must have distinct boxes"

    def test_subdiameter_path_constraints(self):
        """Test Subdiameter Path (S) logic."""
        initial_degrees = [4]*50
        target_diameter = 10
        
        d_prime, v, D, S = self.prep.run_setup(initial_degrees, target_diameter)
        
        # S should be disjoint from D
        assert len(D.intersection(S)) == 0
        
        if len(S) > 0:
            boxes_for_S = [v[i] for i in S]
            assert len(boxes_for_S) == len(set(boxes_for_S)), "Nodes in S must have distinct boxes"