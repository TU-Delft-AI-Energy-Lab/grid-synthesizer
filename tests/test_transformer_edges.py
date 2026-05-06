import pytest
import numpy as np
from powergrid_synth.transmission.transformer_edges import TransformerConnector


class TestTransformerConnector:

    def test_basic_star_generation(self):
        """Center in X with degree 3 should create 3 edges to Y leaves."""
        # X has 4 nodes: node 0 wants 3 connections, nodes 1-3 want 1 each
        t_xy = [3, 1, 1, 1]
        # Y has 6 nodes: all degree 1
        t_yx = [1, 1, 1, 1, 1, 1]

        connector = TransformerConnector()
        edges = connector.generate_transformer_edges(t_xy, t_yx)
        assert isinstance(edges, list)
        assert len(edges) > 0

    def test_edges_are_tuples(self):
        t_xy = [2, 2, 1]
        t_yx = [1, 1, 1, 1, 1]
        connector = TransformerConnector()
        edges = connector.generate_transformer_edges(t_xy, t_yx)
        for e in edges:
            assert isinstance(e, tuple)
            assert len(e) == 2

    def test_symmetric_degree_one(self):
        """All degree-1: should produce matched edges."""
        t_xy = [1, 1, 1, 1]
        t_yx = [1, 1, 1, 1]
        connector = TransformerConnector()
        edges = connector.generate_transformer_edges(t_xy, t_yx)
        assert len(edges) >= 1

    def test_zero_degree_nodes_produce_no_edges(self):
        t_xy = [0, 0, 0]
        t_yx = [0, 0, 0]
        connector = TransformerConnector()
        edges = connector.generate_transformer_edges(t_xy, t_yx)
        assert len(edges) == 0

    def test_leftover_x_chung_lu_fallback(self):
        """X centers want more leaves than Y has, forcing L_x population."""
        # 3 centers in X each want 5 leaves, but Y only has 2 leaves
        t_xy = [5, 5, 5]
        t_yx = [1, 1]
        connector = TransformerConnector()
        edges = connector.generate_transformer_edges(t_xy, t_yx)
        assert isinstance(edges, list)
        for e in edges:
            assert 0 <= e[0] < len(t_xy)
            assert 0 <= e[1] < len(t_yx)

    def test_leftover_y_chung_lu_fallback(self):
        """Y centers want more leaves than X has, forcing L_y population."""
        t_xy = [1, 1]
        t_yx = [5, 5, 5]
        connector = TransformerConnector()
        edges = connector.generate_transformer_edges(t_xy, t_yx)
        assert isinstance(edges, list)
        for e in edges:
            assert 0 <= e[0] < len(t_xy)
            assert 0 <= e[1] < len(t_yx)

    def test_both_leftovers_trigger_chung_lu(self):
        """Both L_x and L_y get populated, triggering bipartite Chung-Lu."""
        # Both sides have centers wanting 5 leaves, but few degree-1 nodes
        t_xy = [5, 5]
        t_yx = [5, 5]
        connector = TransformerConnector()
        edges = connector.generate_transformer_edges(t_xy, t_yx)
        assert isinstance(edges, list)
        for e in edges:
            assert 0 <= e[0] < len(t_xy)
            assert 0 <= e[1] < len(t_yx)

    def test_large_center_insufficient_leaves(self):
        """One very large center with insufficient leaves on opposite side."""
        # Single center in X wants 20 leaves; Y has only 3 leaves
        t_xy = [20, 1, 1]
        t_yx = [1, 1, 1]
        connector = TransformerConnector()
        edges = connector.generate_transformer_edges(t_xy, t_yx)
        assert isinstance(edges, list)
        assert len(edges) > 0
        for e in edges:
            assert 0 <= e[0] < len(t_xy)
            assert 0 <= e[1] < len(t_yx)
