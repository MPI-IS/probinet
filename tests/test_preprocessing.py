import unittest

import networkx as nx
import numpy as np
import sktensor as skt

from pgm.input import tools
from pgm.input.preprocessing import build_B_from_A, build_sparse_B_from_A, preprocess


class TestPreprocessing(unittest.TestCase):
    def test_build_B_from_A(self):
        # Test case for build_B_from_A
        G1 = nx.MultiDiGraph()
        G1.add_edges_from([(0, 1, {'weight': 1}), (1, 2, {'weight': 2}), (2, 1, {'weight': 3})])
        G2 = nx.MultiDiGraph()
        G2.add_edges_from([(0, 1, {'weight': 1}), (0, 2, {'weight': 2}), (2, 0, {'weight': 2})])

        A = [G1, G2]
        nodes = [0, 1, 2]
        expected_B = np.array([[[0, 1, 0], [0, 0, 2], [0, 3, 0]],
                               [[0, 1, 2], [0, 0, 0], [2, 0, 0]]])
        expected_rw = [2, 1.6]

        # Now, the test should pass
        B, rw = build_B_from_A(A, nodes=nodes)
        np.testing.assert_array_equal(B, expected_B)
        np.testing.assert_array_almost_equal(rw, expected_rw)

    def test_build_B_from_A_mismatched_nodes(self):
        # Test case for build_B_from_A with mismatched nodes
        G1 = nx.MultiDiGraph()
        G1.add_edges_from([(1, 2, {'weight': 1}), (2, 3, {'weight': 1})])
        G2 = nx.MultiDiGraph()
        G2.add_edges_from([(1, 3, {'weight': 1})])

        A = [G1, G2]
        nodes = [1, 2, 3]

        # This test should raise an AssertionError due to the mismatched set of vertices
        with self.assertRaises(AssertionError):
            build_B_from_A(A, nodes=nodes)

    def test_build_B_from_A_non_int_weighed_nodes(self):
        # Test case for build_B_from_A with mismatched nodes
        G1 = nx.MultiDiGraph()
        G1.add_edges_from([(1, 2, {'weight': 0.5}), (2, 3, {'weight': 0.8})])

        A = [G1]
        nodes = [1, 2, 3]

        # This test should raise an AssertionError due to the mismatched set of vertices
        with self.assertRaises(AssertionError):
            build_B_from_A(A, nodes=nodes)

    def test_build_sparse_B_from_A(self):
        # Test case for build_sparse_B_from_A
        G1 = nx.MultiDiGraph()
        G1.add_edges_from([(0, 1, {'weight': 1}), (1, 2, {'weight': 3})])
        G2 = nx.MultiDiGraph()
        G2.add_edges_from([(0, 2, {'weight': 2}), (0, 2, {'weight': 2})])
        A = [G1, G2]

        # Create expected sparse tensors using provided indices and values
        expected_data_indices = (
            (np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 2, 1])))
        expected_data_values = np.array([1.0, 2.0, 3.0])
        expected_data_shape = (2, 3, 3)
        expected_data = skt.sptensor(expected_data_indices, expected_data_values,
                                     shape=expected_data_shape)

        expected_data_T_indices = (
            (np.array([0, 0, 1]), np.array([1, 2, 1]), np.array([0, 1, 0])))
        expected_data_T_values = np.array([1.0, 2.0, 3.0])
        expected_data_T_shape = (2, 3, 3)
        expected_data_T = skt.sptensor(expected_data_T_indices, expected_data_T_values,
                                       shape=expected_data_T_shape)

        expected_v_T = np.array([0.0, 0.0, 0.0])
        expected_rw = [0.0, 0.0]

        data, data_T, v_T, rw = build_sparse_B_from_A(A)

        # Use np.testing.assert_array_almost_equal for comparing arrays with floating-point values
        np.testing.assert_array_almost_equal(data.subs, expected_data.subs)
        np.testing.assert_array_almost_equal(data_T.subs, expected_data_T.subs)
        np.testing.assert_array_almost_equal(v_T, expected_v_T)
        np.testing.assert_array_almost_equal(rw, expected_rw)

    def test_preprocess_dense_array(self):
        # Test case for preprocess with dense array
        A = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
        expected_result = skt.dtensor(A)
        result = preprocess(A)
        self.assertEqual(type(result), type(expected_result))
        np.testing.assert_array_equal(result, expected_result)

    def test_preprocess_sparse_array(self):
        # Test case for preprocess with sparse array
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        expected_result = tools.sptensor_from_dense_array(A)
        result = preprocess(A)
        self.assertEqual(type(result), type(expected_result))
        self.assertTrue(np.array_equal(result.subs, expected_result.subs))
        np.testing.assert_array_equal(result.vals, expected_result.vals)
