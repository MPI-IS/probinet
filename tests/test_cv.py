"""
Test cases for the cv module.
"""

import unittest

import numpy as np
from tests.constants import RANDOM_SEED_REPROD

from probinet.model_selection.masking import (
    extract_mask_kfold,
    shuffle_indices_all_matrix,
)


class TestCV(unittest.TestCase):
    """
    Test cases for the cv module.
    """

    def setUp(self):
        # Set up variables for testing
        self.N = 5
        self.L = 3
        self.NFold = 5
        self.rseed = RANDOM_SEED_REPROD

    def test_extract_mask_kfold(self):
        # Test extract_mask_kfold function

        # Define rng
        rng = np.random.default_rng(seed=self.rseed)

        # Generate shuffled indices
        indices = shuffle_indices_all_matrix(self.N, self.L, rng=rng)

        # Test for each fold
        for fold in range(self.NFold):
            mask = extract_mask_kfold(indices, self.N, fold=fold, NFold=self.NFold)

            # Check the dimensions of the mask
            self.assertEqual(mask.shape, (self.L, self.N, self.N))

            # Check if the mask is non-symmetric
            for l in range(self.L):
                self.assertTrue(np.any(mask[l] != mask[l].T))

            # Count the number of ones in the mask and check against the expected count
            ones_count = np.sum(mask)
            expected_ones_count = (self.N * self.N // self.NFold) * self.L
            self.assertEqual(ones_count, expected_ones_count)

    def test_shuffle_indices_all_matrix(self):
        # Test shuffle_indices_all_matrix function

        # Define rng
        rng = np.random.default_rng(seed=self.rseed)

        # Generate shuffled indices
        indices = shuffle_indices_all_matrix(self.N, self.L, rng=rng)

        # Check if the indices are shuffled for each layer
        for l in range(self.L):
            self.assertTrue(np.any(indices[l] != np.arange(self.N * self.N)))
