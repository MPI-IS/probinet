"""
Test cases for the input module.
"""

import networkx as nx

from probinet.input.loader import build_adjacency_from_networkx
from probinet.models.classes import GraphData

from .fixtures import BaseTest


class TestInput(BaseTest):
    def test_build_adjacency_from_networkx(self):
        G = nx.erdos_renyi_graph(n=14, p=0.5, seed=7, directed=False)
        # Add weights
        for i, edge in enumerate(G.edges()):
            G.edges[edge]["weight"] = 3
            G.edges[edge]["weight2"] = i

        g_data = build_adjacency_from_networkx(
            G, weight_list=["weight", "weight2"], file_name=self.folder + "edges.csv"
        )

        self.assertIsInstance(g_data, GraphData)
