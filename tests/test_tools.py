"""
Test cases for the tools module.
"""

import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sparse import COO

from probinet.utils.matrix_operations import (
    Exp_ija_matrix,
    normalize_nonzero_membership,
    sp_uttkrp,
    sp_uttkrp_assortative,
    transpose_matrix,
    transpose_tensor,
)
from probinet.utils.tools import (
    build_edgelist,
    can_cast_to_int,
    get_item_array_from_subs,
    is_sparse,
    output_adjacency,
    sptensor_from_dense_array,
    write_adjacency,
    write_design_matrix,
)

from .constants import DECIMAL, RTOL
from .fixtures import BaseTest


class TestTensors(unittest.TestCase):
    """
    Test cases for the tools module.
    """

    def setUp(self):
        # Parameters for the non-assortative case
        self.vals_ = np.array([1.0, 2.0, 3.0])
        self.subs_ = (np.array([0, 0, 1]), np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.u_ = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self.v_ = np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])
        self.w_ = np.array(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
                [[0.9, 1.0], [1.1, 1.2]],
            ]
        )
        # Parameters for the assortative case
        self.vals_a = np.array(
            [1.0, 2.0, 3.0]
        )  # of size n, number of nodes; these are the non-zero values of the tensor
        self.subs_a = (
            np.array([0, 0, 0]),
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
        )  # list of arrays of indices, each one of size n, lxixj
        self.u_a = np.array([[0.1, 0.2], [0.4, 0], [0.5, 0.6]])  # of size nxk,
        self.v_a = np.array([[0.7, 0], [0.9, 0], [1.1, 0]])  # of size n
        self.w_a = np.array([[0.1, 0.2, 0]])  # lxk #in the other case, this is lxkxk

    def test_can_cast(self):
        self.assertTrue(can_cast_to_int("123"))
        self.assertFalse(can_cast_to_int("12.34"))
        self.assertTrue(can_cast_to_int(42))
        self.assertFalse(can_cast_to_int("abc"))

    def test_normalize_nonzero_membership(self):
        # Test when input is a 2x2 matrix
        input_matrix = np.array([[1, 2], [3, 4]])
        normalized_matrix = normalize_nonzero_membership(input_matrix)
        expected_result = np.array([[0.33333, 0.66666], [0.428571, 0.571429]])
        np.testing.assert_array_almost_equal(
            normalized_matrix, expected_result, decimal=DECIMAL
        )

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
        self.assertTrue(np.allclose(result.coords, expected_result.coords))
        self.assertTrue(np.allclose(result.data, expected_result.data))
        self.assertEqual(result.shape, expected_result.shape)
        self.assertEqual(result.dtype, expected_result.dtype)

    def test_sptensor_from_dense_array(self):
        # Test case where the input ndarray is valid
        dense_array = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        # Define indices and values
        indices = ([0, 1, 2], [0, 1, 2])
        values = np.array([1, 2, 3], dtype=np.float64)
        shape = (3, 3)

        # Create the sparse tensor
        expected_result = COO(indices, values, shape=shape)
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
        result = transpose_tensor(M)
        np.testing.assert_array_equal(result, expected_result)

    def test_transpose_ij2(self):
        # Test case for transpose_ij2
        M = np.array([[1, 2], [3, 4]])
        expected_result = np.array([[1, 3], [2, 4]])
        result = transpose_matrix(M)
        np.testing.assert_array_equal(result, expected_result)

    def test_Exp_ija_matrix(self):
        # Test case for Exp_ija_matrix
        u = np.array([[0.1, 0.2], [0.3, 0.4]])
        v = np.array([[0.5, 0.6], [0.7, 0.8]])
        w = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected_result = np.array([[0.95, 1.29], [2.07, 2.81]])
        result = Exp_ija_matrix(u, v, w)
        np.testing.assert_allclose(result, expected_result, rtol=RTOL)

    def test_sp_uttkrp_mode_1(self):
        mode = 1
        result = sp_uttkrp(self.vals_, self.subs_, mode, self.u_, self.v_, self.w_)
        expected_result = np.array([[0.23, 0.53], [0.58, 1.34], [3.81, 5.19]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_sp_uttkrp_mode_2(self):
        mode = 2
        result = sp_uttkrp(self.vals_, self.subs_, mode, self.u_, self.v_, self.w_)
        expected_result = np.array([[0.07, 0.1], [0.3, 0.44], [2.01, 2.34]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_sp_uttkrp_assortative_mode_1(self):
        mode = 1
        result = sp_uttkrp_assortative(
            self.vals_a, self.subs_a, mode, self.u_a, self.v_a, self.w_a
        )  # this is nxk
        expected_result = np.array([[0.07, 0.0], [0.18, 0], [0.33, 0]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_sp_uttkrp_assortative_mode_2(self):
        mode = 2
        result = sp_uttkrp_assortative(
            self.vals_a, self.subs_a, mode, self.u_a, self.v_a, self.w_a
        )  # this is nxk
        expected_result = np.array([[0.01, 0.04], [0.08, 0], [0.15, 0.36]])
        self.assertTrue(np.allclose(result, expected_result))


class TestWriteDesignMatrix(BaseTest):
    def setUp(self):
        self.metadata = {"node1": "metadata1", "node2": "metadata2"}
        self.perc = 0.5
        self.fname = "test_X_"
        self.nodeID = "Node"
        self.attr_name = "Metadata"
        df = pd.DataFrame.from_dict(
            self.metadata, orient="index", columns=[self.attr_name]
        )
        df.reset_index(inplace=True)
        df.columns = [self.nodeID, self.attr_name]
        self.expected_output = df

    def test_write_design_Matrix(self):
        write_design_matrix(
            self.metadata,
            self.perc,
            self.folder,
            self.fname,
            self.nodeID,
            self.attr_name,
        )
        output_file = (
            self.folder
            + self.fname
            + str(self.perc)[0]
            + "_"
            + str(self.perc)[2]
            + ".csv"
        )
        output_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(output_df, self.expected_output)


class TestAdjacencyFunctions(BaseTest):
    def setUp(self):
        self.A = [nx.MultiDiGraph(), nx.MultiDiGraph()]
        self.G = [nx.MultiDiGraph(), nx.MultiDiGraph()]
        # self.out_folder = "./"
        self.label = "test_output"
        self.fname = "test_write.csv"
        self.ego = "source"
        self.alter = "target"

    def test_write_adjacency(self):
        write_adjacency(self.G, self.folder, self.fname, self.ego, self.alter)
        self.assertTrue((Path(self.folder) / self.fname).is_file())

    def test_build_edgelist(self):
        # Create a sparse tensor with known data

        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4, 5, 7, 9])
        A = coo_matrix((data, (row, col)), shape=(4, 4), dtype=np.int32)

        # Call the function with the tensor and a known layer index
        result = build_edgelist(A, 1)

        # Create the expected result
        expected_result = pd.DataFrame(
            {"source": [0, 3, 1, 0], "target": [0, 3, 1, 2], "L1": [4, 5, 7, 9]},
            dtype=np.int32,
        )

        # Assert that the result is as expected
        pd.testing.assert_frame_equal(result, expected_result)


class TestOutputAdjacency(BaseTest):
    def setUp(self):
        # Create a list of sparse matrices
        self.A = [
            coo_matrix(([1, 2, 3], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
            for _ in range(3)
        ]
        self.label = "test_output_adjacency"

    def test_output_adjacency(self):
        # Call the function with the test inputs
        output_adjacency(self.A, self.folder, self.label)

        # Check if the evaluation file exists
        self.assertTrue(Path(self.folder + self.label + ".dat").is_file())

        # Load the evaluation file into a DataFrame
        df = pd.read_csv(self.folder + self.label + ".dat", sep=" ")

        # Create the expected DataFrame
        df_list = [
            pd.DataFrame(
                {"source": [0, 1, 2], "target": [0, 1, 2], "L" + str(layer): [1, 2, 3]}
            )
            for layer in range(len(self.A))
        ]

        expected_df = pd.concat(df_list).reset_index(drop=True)

        # Check if the evaluation DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(df, expected_df)
