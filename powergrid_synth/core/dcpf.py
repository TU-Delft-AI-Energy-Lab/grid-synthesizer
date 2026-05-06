import numpy as np
import networkx as nx
from scipy.sparse import csc_matrix, linalg

class DCPowerFlow:
    r"""Lightweight DC Power Flow (DCPF) solver.

    Solves the linearised power flow equations

    .. math::

        \mathbf{P} = \mathbf{B}\,\boldsymbol{\theta}, \qquad
        F_{ij} = \frac{\theta_i - \theta_j}{x_{ij}}

    where :math:`\mathbf{B}` is the bus susceptance matrix,
    :math:`\boldsymbol{\theta}` the vector of voltage angles (radians),
    :math:`\mathbf{P}` the vector of net real-power injections (per-unit on
    a 100 MVA base), and :math:`F_{ij}` the real-power flow on branch
    :math:`(i, j)`.

    The slack bus (reference angle :math:`\theta = 0`) is chosen as the bus
    hosting the largest generator.  The solver uses a sparse direct
    factorisation via ``scipy.sparse.linalg.spsolve``.

    This class is used by
    :class:`~powergrid_synth.transmission.TransmissionLineAllocator` to
    compute the flow distribution needed for impedance swapping and
    capacity assignment, following the methodology of
    `Sadeghian et al. (2018) <https://ieeexplore.ieee.org/document/8585532>`_.

    Parameters
    ----------
    graph : networkx.Graph
        Power grid graph.  Each edge must have an ``'x'`` attribute
        (per-unit reactance).  Each node may have ``'pg'`` (generation MW)
        and ``'pl'`` (load MW) attributes.

    See Also
    --------
    powergrid_synth.transmission.TransmissionLineAllocator : Uses DCPF for
        impedance swapping and capacity assignment.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.n_bus = len(self.nodes)
        
    def run(self):
        r"""Execute the DC power flow and return branch flows and bus angles.

        Algorithm
        ---------
        1. Build the susceptance matrix :math:`\mathbf{B}` from branch
           reactances: :math:`B_{ij} = -1/x_{ij}`,
           :math:`B_{ii} = \sum_j 1/x_{ij}`.
        2. Form the net injection vector
           :math:`P_i = (P_{g_i} - P_{L_i}) / S_{\text{base}}`.
        3. Select the slack bus (largest generator), remove its row/column,
           and solve :math:`\mathbf{B}_{\text{red}}\,
           \boldsymbol{\theta}_{\text{red}} = \mathbf{P}_{\text{red}}`.
        4. Compute branch flows
           :math:`F_{ij} = (\theta_i - \theta_j) / x_{ij} \times
           S_{\text{base}}` (MW).

        Returns
        -------
        flows : dict[tuple[int, int], float]
            Branch power flows in MW, keyed by ``(u, v)`` node pairs.
        angles : dict[int, float]
            Bus voltage angles in radians, keyed by node id.
        """
        # 1. Build B Matrix (Susceptance)
        # B_ij = -1/x_ij
        # B_ii = sum(1/x_ij) for all j connected to i
        
        row_ind = []
        col_ind = []
        data = []
        
        # Diagonal elements accumulator
        b_diag = np.zeros(self.n_bus)
        
        edges = []
        
        for u, v, d in self.graph.edges(data=True):
            idx_u = self.node_to_idx[u]
            idx_v = self.node_to_idx[v]
            
            # Reactance x. Avoid division by zero.
            x = d.get('x', 0.1) 
            if x < 1e-6: x = 1e-6
            
            b_val = 1.0 / x
            
            # Off-diagonals (B_ij = -1/x)
            row_ind.extend([idx_u, idx_v])
            col_ind.extend([idx_v, idx_u])
            data.extend([-b_val, -b_val])
            
            # Add to diagonals
            b_diag[idx_u] += b_val
            b_diag[idx_v] += b_val
            
            edges.append((u, v, x))

        # Add diagonal elements
        row_ind.extend(range(self.n_bus))
        col_ind.extend(range(self.n_bus))
        data.extend(b_diag)
        
        B = csc_matrix((data, (row_ind, col_ind)), shape=(self.n_bus, self.n_bus))
        
        # 2. Build P Vector (Injections)
        # P_inj = P_gen - P_load
        P_inj = np.zeros(self.n_bus)
        
        # Identify Slack Bus (usually largest generator or first generator)
        max_pg = -1
        slack_idx = 0
        
        for i, n in enumerate(self.nodes):
            pg = self.graph.nodes[n].get('pg', 0.0)
            pl = self.graph.nodes[n].get('pl', 0.0)
            
            # Net injection (normalized to 100 MVA base if values are MW)
            # Assuming attributes are in MW and BaseMVA=100
            P_inj[i] = (pg - pl) / 100.0
            
            if pg > max_pg:
                max_pg = pg
                slack_idx = i
                
        # 3. Solve for Theta
        # Remove slack row/col to make system non-singular
        # We solve: B_reduced * theta_reduced = P_reduced
        
        # Mask for non-slack buses
        mask = np.ones(self.n_bus, dtype=bool)
        mask[slack_idx] = False
        
        B_red = B[mask, :][:, mask]
        P_red = P_inj[mask]
        
        try:
            theta_red = linalg.spsolve(B_red, P_red)
        except RuntimeError:
            # Fallback for singular matrix (islanded components)
            theta_red = np.zeros_like(P_red)

        # Reconstruct full theta vector
        theta = np.zeros(self.n_bus)
        theta[mask] = theta_red
        theta[slack_idx] = 0.0 # Slack angle reference
        
        # 4. Calculate Flows
        # F_ij = (theta_i - theta_j) / x_ij
        flows = {}
        angles = {self.nodes[i]: theta[i] for i in range(self.n_bus)}
        
        for u, v, x in edges:
            idx_u = self.node_to_idx[u]
            idx_v = self.node_to_idx[v]
            
            th_u = theta[idx_u]
            th_v = theta[idx_v]
            
            # Flow from u to v
            f = (th_u - th_v) / x
            flows[(u, v)] = f * 100.0 # Convert pu back to MW
            
        return flows, angles