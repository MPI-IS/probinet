"""
Unit tests for the plot module.
"""

import unittest

import matplotlib.pyplot as plt
import networkx as nx

from probinet.visualization.plot import mapping, plot_hard_membership


class TestPlotHardMembership(unittest.TestCase):
    """
    Test cases for the plot module.
    """

    def setUp(self):
        # Set up a sample graph for testing
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])

        # Sample hard memberships and positions
        self.communities = {"Community1": [1, 2, 3, 4]}
        self.pos = nx.circular_layout(self.graph)
        self.node_size = 200
        self.colors = {1: "red", 2: "blue", 3: "green", 4: "yellow"}
        self.edge_color = "gray"

    # Skip this test for now
    def test_plot_hard_membership(self):
        # Test plot_hard_membership function

        # Capture the plot evaluation
        with self.subTest():
            plt.figure()
            plot_hard_membership(
                self.graph,
                self.communities,
                self.pos,
                self.node_size,
                self.colors,
                self.edge_color,
            )
            plt.close()

    def test_mapping(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)])
        A = nx.DiGraph()
        A.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)])
        # Assert that the nodes are mapped correctly
        self.assertEqual(list(mapping(G, A).nodes()), [0, 1, 2, 3, 4])
        # Assert that the nodes are mapped correctly
        self.assertEqual(list(mapping(G, A).nodes()), [0, 1, 2, 3, 4])
        # Assert that the edges are mapped correctly
        for edge in mapping(G, A).edges():
            self.assertIn(edge, G.edges())
        for edge in G.edges():
            self.assertIn(edge, mapping(G, A).edges())
