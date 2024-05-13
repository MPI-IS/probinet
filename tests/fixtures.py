"""
This file contains the fixtures for the tests.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

RTOL = 1e-2
DECIMAL = 5
TOLERANCE_1 = 1e-3
TOLERANCE_2 = 1e-3
current_file_path = Path(__file__)
PATH_FOR_INIT = current_file_path.parent / 'inputs/'
INIT_STR = '_for_initialization'


class BaseTest(unittest.TestCase):
    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.folder = temp_output_folder + '/'
            # Call the parent class's run method to execute the test
            super().run(result)

    def _fit_model_to_data(self, conf):
        _ = self.model.fit(data=self.B,
                           data_T=self.B_T,
                           data_T_vals=self.data_T_vals,
                           nodes=self.nodes,
                           **conf)

    def _load_model_results(self, path=None):
        if path is None:
            path = Path(self.model.out_folder) / str('theta' + self.model.end_file)
        return np.load(path.with_suffix('.npz'))

    def _load_ground_truth_results(self, thetaGT_path=None):
        if thetaGT_path is None:
            thetaGT_path = Path(__file__).parent / 'outputs' / ('theta_GT_' + self.algorithm)
        return np.load(thetaGT_path.with_suffix('.npz'))

    def _assert_model_information(self, theta):
        self.assertTrue(np.allclose(self.model.u_f, theta['u']))
        self.assertTrue(np.allclose(self.model.v_f, theta['v']))
        self.assertTrue(np.allclose(self.model.w_f, theta['w']))
        if self.algorithm == 'CRep' or self.algorithm == 'JointCRep':
            self.assertTrue(np.allclose(self.model.eta_f, theta['eta']))
        else:
            self.assertTrue(np.allclose(self.model.beta_f, theta['beta']))

    def _assert_dictionary_keys(self, theta):
        assert all(key in theta for key in self.keys_in_thetaGT), ("Some keys are missing in the "
                                                                   "theta "
                                                                   "dictionary")

    def _assert_ground_truth_information(self, theta, thetaGT):
        self.assertTrue(np.allclose(thetaGT['u'], theta['u']))
        self.assertTrue(np.allclose(thetaGT['v'], theta['v']))
        self.assertTrue(np.allclose(thetaGT['w'], theta['w']))
        if self.algorithm == 'CRep' or self.algorithm == 'JointCRep':
            self.assertTrue(np.allclose(thetaGT['eta'], theta['eta']))
        else:
            self.assertTrue(np.allclose(thetaGT['beta'], theta['beta']))


class ModelTestMixin:
    def test_running_algorithm_from_mixin(self):
        """
        Test running algorithm function.
        """

        # Fit the model to the data
        self._fit_model_to_data(self.conf)

        # Load the model results
        theta = self._load_model_results()

        # Load the ground truth results
        thetaGT = self._load_ground_truth_results()

        # Assert the model information
        self._assert_model_information(theta)

        # Assert the dictionary keys
        self._assert_dictionary_keys(theta)

        # Asserting GT information
        self._assert_ground_truth_information(theta, thetaGT)

    def test_running_algorithm_initialized_from_file_from_mixin(self):

        with (PATH_FOR_INIT.joinpath('setting_' + self.algorithm + INIT_STR + '.yaml').open('rb')
              as fp):
            self.conf = yaml.safe_load(fp)

        # Saving the outputs of the tests inside the tests dir
        self.conf['out_folder'] = self.folder

        self.conf['end_file'] = '_OUT_' + self.algorithm  # Adding a suffix to the output files

        self.conf['initialization'] = 1

        self._fit_model_to_data(self.conf)

        # Load the model results
        theta_path = Path(self.model.out_folder) / str('theta' + self.model.end_file)
        theta = self._load_model_results(theta_path)

        # Load the ground truth results
        thetaGT_path = Path(__file__).parent / 'outputs' / ('theta_GT_' + self.algorithm + INIT_STR)
        thetaGT = self._load_ground_truth_results(thetaGT_path)

        # Assert the model information
        self._assert_model_information(theta)

        # Assert the dictionary keys
        self._assert_dictionary_keys(theta)

        # Asserting GT information
        self._assert_ground_truth_information(theta, thetaGT)


def flt(x, d=3):
    return round(x, d)


def expected_Aija(U, V, W):  # TODO: future refactoring ticket: use a similar function from pgm,
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
    assert np.isclose(np.sum(matrix), expected_sum,
                      atol=TOLERANCE_1), f"Expected sum of {matrix_name} to be {expected_sum}, but got {np.sum(matrix)}"
