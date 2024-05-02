"""
This file contains the fixtures for the tests.
"""
import tempfile
import unittest

import numpy as np

RTOL = 1e-2
DECIMAL = 5
TOLERANCE_1 = 1e-3
TOLERANCE_2 = 1e-3


class BaseTest(unittest.TestCase):
    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.folder = temp_output_folder + '/'
            # Call the parent class's run method to execute the test
            super().run(result)

def flt(x, d=3):
    return round(x, d)

def expected_Aija(U, V, W): #TODO: future refactoring ticket: use a similar function from pgm,
    # and avoid defining this new one here
    if W.ndim == 1:
        M = np.einsum('ik,jk->ijk', U, V)
        M = np.einsum('ijk,k->ij', M, W)
    else:
        M = np.einsum('ik,jq->ijkq', U, V)
        M = np.einsum('ijkq,kq->ij', M, W)
    return M

def check_shape_and_sum(matrix, expected_shape, expected_sum, matrix_name):
    assert matrix.shape == expected_shape, f"Expected {matrix_name} to have shape {expected_shape}, but got {matrix.shape}"
    assert np.isclose(np.sum(matrix), expected_sum, atol=TOLERANCE_1), f"Expected sum of {matrix_name} to be {expected_sum}, but got {np.sum(matrix)}"
