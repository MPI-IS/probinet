import os

import networkx as nx
import numpy as np

from probinet.input.loader import build_adjacency_from_file
from probinet.models.classes import GraphData

from .fixtures import BaseTest


class TestBuildAdjacencyFromFile(BaseTest):
    def setUp(self):
        # Initialize node labels as instance attributes
        self.node1 = "A"
        self.node2 = "B"
        self.node3 = "C"

        # Initialize weights as random positive integers
        self.weight_ab = np.random.randint(1, 10)
        self.weight_bc = np.random.randint(1, 10)
        self.weight_ca = np.random.randint(1, 10)

        # Create the graph
        self.create_graph()

    def create_graph(self):
        # Create a simple undirected graph with integer weights
        self.graph = nx.Graph()
        self.graph.add_edge(self.node1, self.node2, weight=self.weight_ab)
        self.graph.add_edge(self.node2, self.node3, weight=self.weight_bc)
        self.graph.add_edge(self.node3, self.node1, weight=self.weight_ca)

    # This attribute should be updated for each test.
    @property
    def edge_list(self):
        return [
            (self.node1, self.node2, {"weight": self.weight_ab}),
            (self.node1, self.node3, {"weight": self.weight_ca}),
            (self.node2, self.node3, {"weight": self.weight_bc}),
        ]

    def save_graph_to_csv(self, graph, file_path):
        """
        Save the graph to a CSV file.
        """
        df = nx.to_pandas_edgelist(graph)
        df.to_csv(file_path, index=False)
        return df

    def helper_test_build_adjacency_from_file(
        self, binary, undirected, expected_tensor
    ):
        """
        Helper function to test the build_adjacency_from_file function.
        """
        # Save the graph to a CSV file in the temporary folder
        csv_file = os.path.join(self.folder, "test_graph.csv")
        df = self.save_graph_to_csv(self.graph, csv_file)

        # Call the function and check the output
        result = build_adjacency_from_file(
            path_to_file=csv_file, sep=",", undirected=undirected, binary=binary
        )
        self.assertIsInstance(result, GraphData)
        np.testing.assert_array_equal(result.adjacency_tensor, expected_tensor)

        self.assertEqual(len(result.graph_list), 1)
        graph = result.graph_list[0]
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertEqual(
            sorted(graph.edges(data=True)),
            self.edge_list,
        )

    def test_build_adjacency_from_file_with_binary_false(self):
        expected_tensor = np.array(
            [
                [
                    [0.0, self.weight_ab, self.weight_ca],
                    [self.weight_ab, 0.0, self.weight_bc],
                    [self.weight_ca, self.weight_bc, 0.0],
                ]
            ]
        )
        self.helper_test_build_adjacency_from_file(
            binary=False, undirected=True, expected_tensor=expected_tensor
        )

    def test_build_adjacency_from_file_with_binary_true(self):
        self.weight_ab = self.weight_bc = self.weight_ca = 1
        expected_tensor = np.array(
            [
                [
                    [0.0, self.weight_ab, self.weight_ca],
                    [self.weight_ab, 0.0, self.weight_bc],
                    [self.weight_ca, self.weight_bc, 0.0],
                ]
            ]
        )
        self.helper_test_build_adjacency_from_file(
            binary=True, undirected=True, expected_tensor=expected_tensor
        )

    def test_build_adjacency_from_file_with_undirected_false(self):
        expected_tensor = np.array(
            [
                [
                    [0.0, self.weight_ab, self.weight_ca],
                    [0.0, 0.0, self.weight_bc],
                    [0.0, 0.0, 0.0],
                ]
            ]
        )
        self.helper_test_build_adjacency_from_file(
            binary=False, undirected=False, expected_tensor=expected_tensor
        )

    def test_build_adjacency_from_file_with_additional_weight(self):
        # Modify the graph to add a new weight called 'length'
        for edge in self.graph.edges:
            self.graph.edges[edge]["length"] = self.graph.edges[edge]["weight"] + 2

        # Save the graph to a CSV file in the temporary folder
        csv_file = os.path.join(self.folder, "test_graph.csv")
        self.save_graph_to_csv(self.graph, csv_file)

        # Call the function and check the output
        result = build_adjacency_from_file(
            path_to_file=csv_file,
            sep=",",
            binary=False,
            undirected=False,
        )
        self.assertIsInstance(result, GraphData)

        graph1 = result.graph_list[0]
        graph2 = result.graph_list[1]

        # We now check that the tensor and the graphs are correctly built. Given that the order
        # of the graphs is not guaranteed, we will check both possibilities. The graph is not
        # guaranteed because networkx lists the edge attributes in different orders depending on
        # the call to the function.
        try:
            np.testing.assert_array_equal(
                result.adjacency_tensor,
                np.array(
                    [
                        [
                            [0.0, self.weight_ab, self.weight_ca],
                            [0.0, 0.0, self.weight_bc],
                            [0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, self.weight_ab + 2, self.weight_ca + 2],
                            [0.0, 0.0, self.weight_bc + 2],
                            [0.0, 0.0, 0.0],
                        ],
                    ]
                ),
            )
        except AssertionError:
            expected_tensor = np.array(
                [
                    [
                        [0.0, self.weight_ab + 2, self.weight_ca + 2],
                        [0.0, 0.0, self.weight_bc + 2],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, self.weight_ab, self.weight_ca],
                        [0.0, 0.0, self.weight_bc],
                        [0.0, 0.0, 0.0],
                    ],
                ]
            )
            np.testing.assert_array_equal(result.adjacency_tensor, expected_tensor)

        try:
            self.assertEqual(sorted(graph1.edges(data=True)), self.edge_list)

            self.assertEqual(
                sorted(graph2.edges(data=True)),
                [
                    (self.node1, self.node2, {"weight": self.weight_ab + 2}),
                    (self.node1, self.node3, {"weight": self.weight_ca + 2}),
                    (self.node2, self.node3, {"weight": self.weight_bc + 2}),
                ],
            )
        except AssertionError:
            self.assertEqual(sorted(graph2.edges(data=True)), self.edge_list)

            self.assertEqual(
                sorted(graph1.edges(data=True)),
                [
                    (self.node1, self.node2, {"weight": self.weight_ab + 2}),
                    (self.node1, self.node3, {"weight": self.weight_ca + 2}),
                    (self.node2, self.node3, {"weight": self.weight_bc + 2}),
                ],
            )

    def test_build_adjacency_from_file_with_negative_weights(self):
        self.weight_ab = self.weight_bc = self.weight_ca = -1
        self.create_graph()
        expected_tensor = np.array(
            [
                [
                    [0.0, self.weight_ab, self.weight_ca],
                    [self.weight_ab, 0.0, self.weight_bc],
                    [self.weight_ca, self.weight_bc, 0.0],
                ]
            ]
        )
        with self.assertRaises(ValueError) as context:
            self.helper_test_build_adjacency_from_file(
                binary=False, undirected=True, expected_tensor=expected_tensor
            )
        self.assertEqual(str(context.exception), "There are negative weights.")
