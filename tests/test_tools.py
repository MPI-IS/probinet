"""
Test cases for the tools module.
"""
import os
import unittest

import numpy as np
from scipy import sparse
import sktensor as skt

from pgm.input.tools import (
    can_cast, Exp_ija_matrix, get_item_array_from_subs, is_sparse, normalize_nonzero_membership,
    output_adjacency, sptensor_from_dense_array, transpose_ij2, transpose_ij3)

from .fixtures import decimal, rtol


class TestToolsModule(unittest.TestCase):
    """
    Test cases for the tools module.
    """

    def test_can_cast(self):
        self.assertTrue(can_cast("123"))
        self.assertFalse(can_cast("12.34"))
        self.assertTrue(can_cast(42))
        self.assertFalse(can_cast("abc"))

    def test_normalize_nonzero_membership(self):
        # Test when input is a 2x2 matrix
        input_matrix = np.array([[1, 2], [3, 4]])
        normalized_matrix = normalize_nonzero_membership(input_matrix)
        expected_result = np.array([[0.33333, 0.66666], [0.428571, 0.571429]])
        np.testing.assert_array_almost_equal(normalized_matrix, expected_result, decimal=decimal)

    def test_is_sparse_sparse_tensor(self):
        # Test case where the tensor is considered sparse
        sparse_tensor = np.array([[[0, 1], [0, 0]], [[0, 0], [0, 0]]])
        result = is_sparse(sparse_tensor)
        self.assertTrue(result)

    def test_is_sparse_dense_tensor(self):
        # Test case where the tensor is considered dense
        dense_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = is_sparse(dense_tensor)
        self.assertFalse(result)

    def test_is_sparse_edge_case(self):
        # Test case with an edge case (empty tensor)
        empty_tensor = np.array([])
        result = is_sparse(empty_tensor)
        self.assertFalse(result)

    def assertSptensorEqual(self, result, expected_result):
        self.assertTrue(np.allclose(result.subs, expected_result.subs))
        self.assertTrue(np.allclose(result.vals, expected_result.vals))
        self.assertEqual(result.shape, expected_result.shape)
        self.assertEqual(result.dtype, expected_result.dtype)

    def test_sptensor_from_dense_array(self):
        # Test case where the input ndarray is valid
        dense_array = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        expected_result = skt.sptensor(([0, 1, 2], [0, 1, 2]), [1, 2, 3], shape=(3, 3))
        result = sptensor_from_dense_array(dense_array)
        self.assertSptensorEqual(result, expected_result)

    def test_get_item_array_from_subs(self):
        # Test case for get_item_array_from_subs
        A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        ref_subs = ([0, 0], [0, 1], [1, 1])
        expected_result = np.array([2, 4])
        result = get_item_array_from_subs(A, ref_subs)
        np.testing.assert_array_equal(result, expected_result)

    def test_transpose_ij3(self):
        # Test case for transpose_ij3
        M = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        expected_result = np.array([[[1, 3], [2, 4]], [[5, 7], [6, 8]]])
        result = transpose_ij3(M)
        np.testing.assert_array_equal(result, expected_result)

    def test_transpose_ij2(self):
        # Test case for transpose_ij2
        M = np.array([[1, 2], [3, 4]])
        expected_result = np.array([[1, 3], [2, 4]])
        result = transpose_ij2(M)
        np.testing.assert_array_equal(result, expected_result)

    def test_Exp_ija_matrix(self):
        # Test case for Exp_ija_matrix
        u = np.array([[0.1, 0.2], [0.3, 0.4]])
        v = np.array([[0.5, 0.6], [0.7, 0.8]])
        w = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected_result = np.array([[0.95, 1.29], [2.07, 2.81]])
        result = Exp_ija_matrix(u, v, w)
        np.testing.assert_allclose(result, expected_result, rtol=rtol)

    @unittest.skip("Reason: Not implemented yet")
    def test_output_adjacency(self):
        # Test output adjacency function.
        # Create an empty scipy sparse matrix

        i = [0, 0, 0, 1, 1, 1]
        j = [0, 1, 2, 0, 3, 4]
        A = [sparse.csr_matrix((np.ones_like(j), (i, j)))]

        out_folder = 'tests/'
        label = 'test_output_adjacency'
        # This gives: AttributeError: 'DataFrame' object has no attribute 'append'
        output_adjacency(A, out_folder, label)

        self.assertTrue(os.path.exists(out_folder + label + '.dat'))
