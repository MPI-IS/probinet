import unittest

import numpy as np
from sklearn import metrics

from probinet.evaluation.link_prediction import calculate_f1_score


class TestCalculateF1Score(unittest.TestCase):
    def test_calculate_f1_score_no_mask(self):
        pred = np.array([[[0.2, 0.8], [0.6, 0.4]]])
        data0 = np.array([[0, 1], [1, 0]])
        expected_f1 = 0.6666666666666666
        self.assertAlmostEqual(calculate_f1_score(data0, pred), expected_f1)

    def test_calculate_f1_score_with_threshold(self):
        pred = np.array([[[0.2, 0.8], [0.6, 0.4]]])
        data0 = np.array([[0, 1], [1, 0]])
        threshold = 0.5
        Z_pred = np.array([[0, 1], [1, 0]])
        expected_f1 = metrics.f1_score(data0.flatten(), Z_pred.flatten())
        self.assertAlmostEqual(
            calculate_f1_score(data0, pred, threshold=threshold), expected_f1
        )
