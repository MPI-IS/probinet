"""
Test cases for the evaluate module.
"""

import unittest

import numpy as np
from tests.constants import RANDOM_SEED_REPROD

from probinet.evaluation.expectation_computation import (
    calculate_conditional_expectation,
    calculate_expectation,
    compute_mean_lambda0,
)
from probinet.evaluation.link_prediction import compute_link_prediction_AUC
from tests.constants import RANDOM_SEED_REPROD


class TestEvaluateFunctions(unittest.TestCase):
    """
    Test cases for the evaluate module.
    """

    def setUp(self):
        # Set up variables for testing
        self.N = 5
        self.L = 3
        self.K = 2
        self.eta = 0.2
        self.rseed = RANDOM_SEED_REPROD

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
        auc_result = compute_link_prediction_AUC(pred, self.B, mask=self.mask)

        # Check if AUC result is a number
        self.assertIsInstance(auc_result, float)

    def test_calculate_AUC_1(self):
        # Test calculate_AUC function

        # Set up a perfect prediction
        perfect_pred = self.B.astype(float)

        # Calculate AUC for perfect prediction
        auc_result = compute_link_prediction_AUC(perfect_pred, self.B, mask=self.mask)

        # Check if AUC result is 1.0
        self.assertEqual(auc_result, 1.0)

    def test_lambda0_full(self):
        # Example input values
        u = np.array([[0.1, 0.2], [0.3, 0.4]])
        v = np.array([[0.5, 0.6], [0.7, 0.8]])
        w = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Updated expected evaluation for the given inputs
        expected_result = np.array(
            [[[0.29, 0.39], [0.63, 0.85]], [[0.63, 0.85], [1.41, 1.91]]]
        )

        # Call the _lambda0_full function
        result = compute_mean_lambda0(u, v, w)

        # Check if the result matches the updated expected evaluation
        self.assertTrue(np.allclose(result, expected_result))

    def test_calculate_conditional_expectation(self):
        # Test calculate_conditional_expectation function

        # Calculate conditional expectation
        cond_expectation = calculate_conditional_expectation(
            self.B, self.u, self.v, self.w, eta=self.eta
        )

        # Check the shape of the result
        self.assertEqual(cond_expectation.shape, (self.L, self.N, self.N))

    def test_calculate_expectation(self):
        # Test calculate_expectation function

        # Calculate expectation
        expectation = calculate_expectation(self.u, self.v, self.w, eta=self.eta)

        # Check the shape of the result
        self.assertEqual(expectation.shape, (self.L, self.N, self.N))
