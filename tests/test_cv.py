import unittest

import numpy as np

from pgm.model.cv import extract_mask_kfold, shuffle_indices_all_matrix


class TestCV(unittest.TestCase):

    def setUp(self):
        # Set up variables for testing
        self.N = 5
        self.L = 3
        self.NFold = 5
        self.rseed = 10

    def test_extract_mask_kfold(self):
        # Test extract_mask_kfold function

        # Generate shuffled indices
        indices = shuffle_indices_all_matrix(self.N, self.L, rseed=self.rseed)

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

        # Generate shuffled indices
        indices = shuffle_indices_all_matrix(self.N, self.L, rseed=self.rseed)

        # Check if the indices are shuffled for each layer
        for l in range(self.L):
            self.assertTrue(np.any(indices[l] != np.arange(self.N * self.N)))

