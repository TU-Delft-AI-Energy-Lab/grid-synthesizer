import pytest
import numpy as np
import networkx as nx
from powergrid_synth.core.dcpf import DCPowerFlow


class TestDCPowerFlow:

    @pytest.fixture()
    def simple_3bus(self):
        """3-bus system: Gen at 0 (slack), Load at 1 and 2."""
        G = nx.Graph()
        G.add_node(0, pg=100.0, pl=0.0, pg_max=200.0)
        G.add_node(1, pg=0.0, pl=50.0)
        G.add_node(2, pg=0.0, pl=40.0)
        G.add_edge(0, 1, x=0.1)
        G.add_edge(1, 2, x=0.2)
        G.add_edge(0, 2, x=0.15)
        return G

    def test_run_returns_flows_and_angles(self, simple_3bus):
        dcpf = DCPowerFlow(simple_3bus)
        flows, angles = dcpf.run()
        assert isinstance(flows, dict)
        assert isinstance(angles, dict)

    def test_flows_keys_are_edge_tuples(self, simple_3bus):
        dcpf = DCPowerFlow(simple_3bus)
        flows, _ = dcpf.run()
        for key in flows:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_angles_keys_are_node_ids(self, simple_3bus):
        dcpf = DCPowerFlow(simple_3bus)
        _, angles = dcpf.run()
        assert set(angles.keys()) == {0, 1, 2}

    def test_slack_bus_angle_is_zero(self, simple_3bus):
        """Slack bus (largest gen) should have angle = 0."""
        dcpf = DCPowerFlow(simple_3bus)
        _, angles = dcpf.run()
        assert angles[0] == pytest.approx(0.0)

    def test_power_balance(self, simple_3bus):
        """Net injection should approximately equal generation - load."""
        dcpf = DCPowerFlow(simple_3bus)
        flows, _ = dcpf.run()
        # Total injection at slack = sum of loads = 90 MW
        total_flow_from_slack = sum(
            f for (u, v), f in flows.items() if u == 0
        ) - sum(
            f for (u, v), f in flows.items() if v == 0
        )
        assert abs(total_flow_from_slack - 90.0) < 1.0

    def test_single_line_system(self):
        """2-bus: gen at 0, load at 1."""
        G = nx.Graph()
        G.add_node(0, pg=50.0, pl=0.0, pg_max=100.0)
        G.add_node(1, pg=0.0, pl=50.0)
        G.add_edge(0, 1, x=0.1)
        dcpf = DCPowerFlow(G)
        flows, angles = dcpf.run()
        # Flow on the single line should be ~50 MW
        flow_val = list(flows.values())[0]
        assert abs(abs(flow_val) - 50.0) < 1.0
