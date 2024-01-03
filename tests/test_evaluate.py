import unittest

import numpy as np

from pgm.output.evaluate import (
    _lambda0_full, calculate_AUC, calculate_conditional_expectation, calculate_expectation,
    PSloglikelihood)


class TestEvaluateFunctions(unittest.TestCase):

    def setUp(self):
        # Set up variables for testing
        self.N = 5
        self.L = 3
        self.K = 2
        self.eta = 0.2
        self.rseed = 42

        # Generate random data for testing
        np.random.seed(self.rseed)
        self.u = np.random.rand(self.N, self.K)
        self.v = np.random.rand(self.N, self.K)
        self.w = np.random.rand(self.L, self.K, self.K)
        self.B = np.random.randint(2, size=(self.L, self.N, self.N))
        self.mask = np.random.choice([True, False], size=(self.L, self.N, self.N))

    def test_calculate_AUC(self):
        # Test calculate_AUC function

        # Generate random prediction values
        pred = np.random.rand(self.L, self.N, self.N)

        # Calculate AUC
        auc_result = calculate_AUC(pred, self.B, mask=self.mask)

        # Check if AUC result is a number
        self.assertIsInstance(auc_result, float)

    def test_calculate_AUC_1(self):
        # Test calculate_AUC function

        # Set up a perfect prediction
        perfect_pred = self.B.astype(float)

        # Calculate AUC for perfect prediction
        auc_result = calculate_AUC(perfect_pred, self.B, mask=self.mask)

        # Check if AUC result is 1.0
        self.assertEqual(auc_result, 1.0)

    def test_lambda0_full(self):
        # Example input values
        u = np.array([[0.1, 0.2], [0.3, 0.4]])
        v = np.array([[0.5, 0.6], [0.7, 0.8]])
        w = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Updated expected output for the given inputs
        expected_result = np.array([[[0.29, 0.39], [0.63, 0.85]],
                                    [[0.63, 0.85], [1.41, 1.91]]])

        # Call the _lambda0_full function
        result = _lambda0_full(u, v, w)

        # Check if the result matches the updated expected output
        self.assertTrue(np.allclose(result, expected_result))

    def test_calculate_conditional_expectation(self):
        # Test calculate_conditional_expectation function

        # Calculate conditional expectation
        cond_expectation = calculate_conditional_expectation(self.B, self.u, self.v, self.w,
                                                             eta=self.eta)

        # Check the shape of the result
        self.assertEqual(cond_expectation.shape, (self.L, self.N, self.N))

    def test_calculate_expectation(self):
        # Test calculate_expectation function

        # Calculate expectation
        expectation = calculate_expectation(self.u, self.v, self.w, eta=self.eta)

        # Check the shape of the result
        self.assertEqual(expectation.shape, (self.L, self.N, self.N))

    @unittest.skip("Reason: Not implemented yet")
    def test_PSloglikelihood(self):
        # Test PSloglikelihood function

        # Calculate pseudo log-likelihood
        psloglikelihood_result = PSloglikelihood(self.B, self.u, self.v, self.w, self.eta,
                                                 mask=self.mask)

        # Check if psloglikelihood_result is a number
        self.assertIsInstance(psloglikelihood_result, float)

    @unittest.skip("Reason: Not implemented yet")
    def test_calculate_opt_func(self):
        # Test calculate_opt_func function

        # Calculate optimal value for pseudo log-likelihood
        opt_func_result = calculate_opt_func(self.B, mask=self.mask)

        # Check if opt_func_result is a number
        self.assertIsInstance(opt_func_result, float)
