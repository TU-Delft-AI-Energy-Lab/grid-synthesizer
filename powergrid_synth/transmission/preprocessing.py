r"""
Preprocessing stage (Algorithm 1) for the Chung-Lu Chain (CLC) model.

Prepares the degree sequence, box assignments, and path vertex selections
needed by :class:`~powergrid_synth.edge_creation.EdgeCreator`.

See `Aksoy et al. (2018) <https://doi.org/10.1093/comnet/cny016>`_
(arXiv:1711.11098, Appendix A.2–A.3) for the full algorithm description.
"""
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Set

class Preprocessor:
    r"""
    Preprocessing for a single same-voltage subgraph (Algorithm 1).

    Given desired degrees :math:`\mathbf{d}` and diameter :math:`\delta`,
    produces an inflated degree sequence, box assignments, and
    diameter/subdiameter path vertex sets used by the CLC edge generator.

    See `Aksoy et al. (2018) <https://doi.org/10.1093/comnet/cny016>`_,
    Section 4.3 and Appendix A.3.
    """

    def assign_boxes(self, indices: np.ndarray, boxes: List[int]) -> Dict[int, int]:
        r"""
        Randomly assign vertices to boxes (Lines 43–49 of Algorithm 1).

        Each vertex index is assigned to a uniformly random box from the
        provided list.

        Parameters
        ----------
        indices : numpy.ndarray
            Vertex indices to assign.
        boxes : list of int
            Available box IDs.

        Returns
        -------
        dict
            Mapping ``{vertex_index: box_id}``.
        """
        mapping = {}
        # Line 44: for each i in indices
        for i in indices:
            # Line 45: Randomly choose b in boxes
            if not boxes:
                continue 
            b = random.choice(boxes)
            # Line 46: v_i <- b
            mapping[int(i)] = b
        return mapping

    def run_setup(self, desired_degrees: List[int], input_diameter: int) -> Tuple[np.ndarray, np.ndarray, Set[int], Set[int]]:
        r"""
        Execute the Setup procedure (Algorithm 1).

        The preprocessing performs five steps:

        1. **Diameter adjustment** (Line 3): :math:`\delta \leftarrow
           \text{round}\!\big(\delta - 2\log(\eta/(\delta+1))\big)`.
        2. **Degree-sequence inflation** (Lines 4–8): duplicate random
           nonzero-degree entries until the expected number of Chung-Lu
           isolated vertices :math:`\sum_i e^{-d_i}` matches the number of
           zero-degree vertices :math:`n - \eta`.
        3. **Box assignment** (Lines 9–22): distribute non-isolated vertices
           into :math:`\delta+1` boxes, optionally keeping some boxes empty
           so each non-empty box can support the maximum degree.
        4. **Diameter path** (Lines 25–35): select :math:`\delta+1` vertices
           (degree :math:`\geq 3` preferred) and place one in each box.
        5. **Subdiameter path** (Lines 37–45): select up to
           :math:`\delta+1` additional vertices and assign to centred
           consecutive boxes.

        Parameters
        ----------
        desired_degrees : list of int
            Input degree sequence :math:`\mathbf{d}`.
        input_diameter : int
            Desired diameter :math:`\delta`.

        Returns
        -------
        d_prime : numpy.ndarray
            Inflated degree sequence :math:`\mathbf{d}'`.
        v : numpy.ndarray
            Vertex-to-box mapping (``v[i]`` = box ID; ``-1`` = unassigned).
        D : set of int
            Diameter path vertex indices.
        S : set of int
            Subdiameter path vertex indices.
        """
        # Work with a list for easy inflation, convert to numpy later
        d = list(desired_degrees)
        
        # Line 2: eta <- |{d in d : d > 0}|
        eta = sum(1 for x in d if x > 0)
        
        # Line 3: delta adjustment
        # Formula: delta <- round(delta - 2 * log(eta / (delta + 1)))
        if input_diameter + 1 > 0 and eta > 0:
            term = 2 * math.log(eta / (input_diameter + 1))
            delta = int(round(input_diameter - term))
        else:
            delta = input_diameter
            
        # Ensure delta is valid (at least 1)
        delta = max(1, delta)

        # --- Lines 4-8: Inflate degree sequence ---
        while True:
            current_n = len(d)
            exp_sum = sum(math.exp(-x) for x in d)
            
            lhs = current_n - exp_sum
            
            if lhs > eta:
                break

            # Line 6: Randomly select nonzero d from d
            non_zeros = [x for x in d if x > 0]
            if not non_zeros:
                break # Avoid infinite loop if all zeros
            
            chosen_d = random.choice(non_zeros)
            d.append(chosen_d)

        d = np.array(d)
        n_total = len(d)

        # --- Lines 9-11: Randomly distribute non-isolated vertices into boxes ---
        I_O = np.where(d >= 1)[0]
        B = list(range(delta + 1))

        # --- Lines 13-19: Select Box Subset C ---
        max_d = np.max(d) if len(d) > 0 else 0
        C = []

        if (delta + 1) > 0 and max_d > 0 and (eta / (delta + 1)) < max_d:
            c_size = int(math.ceil(eta / max_d)) 
            c_size = max(1, min(c_size, len(B)))
            C = random.sample(B, c_size)
        else:
            C = list(B)

        # Line 20: v = ASSIGNBOXES(I_O, C)
        v = np.full(n_total, -1, dtype=int)
        
        assignments = self.assign_boxes(I_O, C)
        for idx, box_id in assignments.items():
            v[idx] = box_id

        # --- Lines 21-31: Diameter Path Selection ---
        candidates_ge_3 = np.where(d >= 3)[0]
        
        if len(candidates_ge_3) >= (delta + 1):
            I_P = candidates_ge_3
        else:
            I_P = np.where(d >= 2)[0]
            
        # Ensure I_P contains standard Python ints for set operations later
        I_P = [int(x) for x in I_P]

        # Line 27: Randomly choose D subset I_P with |D| = delta + 1
        if len(I_P) < (delta + 1):
            # Fallback: if not enough nodes for diameter, take all available
            D = set(I_P)
        else:
            D = set(random.sample(I_P, delta + 1))

        # Lines 28-31: Assign each i in D to distinct box b in B
        available_boxes_D = list(B)
        random.shuffle(available_boxes_D)
        
        for i in D:
            if available_boxes_D:
                b = available_boxes_D.pop()
                v[i] = b

        # --- Lines 32-40: Subdiameter Path Selection ---
        # Line 33: alpha calculation
        I_P_minus_D = list(set(I_P) - D)
        alpha = min(delta + 1, len(I_P_minus_D)) - 1
        
        if alpha >= 0:
            # Line 34: beta calculation
            term_beta_1 = math.floor((delta + 1) / 2)
            term_beta_2 = math.floor((alpha - 1) / 2)
            beta = term_beta_1 - term_beta_2
            
            # Line 35: B <- {beta, ..., beta + alpha}
            # The formula gives 1-based indices (relative to 1..delta+1)
            # We must shift to 0-based indices for Python (0..delta)
            B_sub_indices = [b - 1 for b in range(beta, beta + alpha + 1)]
            
            # Filter B_sub to ensure it's within valid box range (0 to delta)
            B_sub = [b for b in B_sub_indices if 0 <= b <= delta]

            # Line 36: Randomly choose S subset I_P \ D with |S| = alpha + 1
            S = set()
            
            sample_size = min(len(I_P_minus_D), alpha + 1)
            if sample_size > 0:
                S = set(random.sample(I_P_minus_D, sample_size))

            # Lines 37-40: Assign each i in S to distinct box in B_sub
            available_boxes_S = list(B_sub)
            random.shuffle(available_boxes_S)
            
            for i in S:
                if available_boxes_S:
                    b = available_boxes_S.pop()
                    v[i] = b
        else:
            S = set()

        return d, v, D, S