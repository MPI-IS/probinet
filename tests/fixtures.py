"""
This file contains the fixtures for the tests.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from probinet.model_selection.cross_validation import CrossValidation

from .constants import INIT_STR, K_NEW, PATH_FOR_INIT, RANDOM_SEED_REPROD, TOLERANCE_1

ALGORITHM = "CRep"
MODEL_PARAMETERS = {}
CV_PARAMETERS = {}


class BaseTest(unittest.TestCase):
    """
    This is a base test class that sets up a temporary directory for each test run.
    It also provides helper methods for fitting models to data, loading models results,
    and asserting models information. It does not have any test methods of its own.
    """

    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.folder = temp_output_folder + "/"
            # Call the parent class's run method to execute the test
            super().run(result)

    def _fit_model_to_data(self, conf):
        _ = self.model.fit(
            self.gdata,
            **conf,
        )

    def _load_model_results(self, path=None):
        if path is None:
            path = Path(self.model.out_folder) / str("theta" + self.model.end_file)
        return np.load(path.with_suffix(".npz"))

    def _load_ground_truth_results(self, thetaGT_path=None):
        if thetaGT_path is None:
            thetaGT_path = (
                Path(__file__).parent / "outputs" / ("theta_GT_" + self.algorithm)
            )
        return np.load(thetaGT_path.with_suffix(".npz"))

    def _assert_model_information(self, theta):
        self.assertTrue(np.allclose(self.model.u_f, theta["u"]))
        self.assertTrue(np.allclose(self.model.v_f, theta["v"]))
        self.assertTrue(np.allclose(self.model.w_f, theta["w"]))
        if self.algorithm == "CRep" or self.algorithm == "JointCRep":
            self.assertTrue(np.allclose(self.model.eta_f, theta["eta"]))
        elif self.algorithm == "MTCOV":
            self.assertTrue(np.allclose(self.model.beta_f, theta["beta"]))

    def _assert_dictionary_keys(self, theta):
        self.assertTrue(
            all(key in theta for key in self.keys_in_thetaGT),
            "Some keys are missing in the theta dictionary",
        )

    def _assert_ground_truth_information(self, theta, thetaGT):
        self.assertTrue(np.allclose(thetaGT["u"], theta["u"]))
        self.assertTrue(np.allclose(thetaGT["v"], theta["v"]))
        self.assertTrue(np.allclose(thetaGT["w"], theta["w"]))
        # Check if the name of the algorithm used is contained in either CRep or JointCRep
        if (
            self.algorithm.startswith("CRep") or self.algorithm.startswith("JointCRep")
        ) and "DynCRep" not in self.algorithm:
            self.assertTrue(np.allclose(thetaGT["eta"], theta["eta"]))
        # If not, then it should be MTCOV
        elif "MTCOV" in self.algorithm:
            self.assertTrue(np.allclose(thetaGT["beta"], theta["beta"]))


class ModelTestMixin:
    """
    A mixin class that provides common test methods for validating models.

    This class is designed to be used as a mixin, adding its methods to those of a unittest.TestCase subclass.
    It provides methods to fit a models to data, load models results, load ground truth results, and assert
    various conditions about the models and its results.

    The methods in this class should be used in the validation tests for each models to ensure consistency
    and reduce code duplication.

    Methods:
        test_running_algorithm_from_mixin: Tests the algorithm by fitting the models to data, loading the
        models results, and asserting various conditions about the models and its results.

        test_running_algorithm_initialized_from_file_from_mixin: Similar to test_running_algorithm_from_mixin,
        but the models is initialized from a file before fitting it to the data.
    """

    def running_algorithm_from_mixin(self, path=None):
        """
        Test running algorithm function.
        """

        # Fit the models to the data
        self._fit_model_to_data(self.conf)

        # Load the models results
        theta = self._load_model_results(path)

        # Load the ground truth results
        thetaGT = self._load_ground_truth_results(path)

        # Assert the models information
        self._assert_model_information(theta)

        # Assert the dictionary keys
        self._assert_dictionary_keys(theta)

        # Asserting GT information
        self._assert_ground_truth_information(theta, thetaGT)

    def running_algorithm_initialized_from_file_from_mixin(self):
        with open(
            PATH_FOR_INIT / ("setting_" + self.algorithm + ".yaml"), encoding="utf-8"
        ) as fp:
            self.conf = yaml.safe_load(fp)
        # Saving the outputs of the tests inside the tests dir
        self.conf["out_folder"] = self.folder

        self.conf["end_file"] = (
            "_OUT_" + self.algorithm
        )  # Adding a suffix to the evaluation files

        self.conf["initialization"] = 1
        self.conf["files"] = self.files
        self.conf["rng"] = np.random.default_rng(seed=RANDOM_SEED_REPROD)

        self._fit_model_to_data(self.conf)

        # Load the models results
        theta_path = Path(self.model.out_folder) / str("theta" + self.model.end_file)
        theta = self._load_model_results(theta_path)

        # Load the ground truth results
        thetaGT_path = (
            Path(__file__).parent
            / "outputs"
            / ("theta_GT_" + self.algorithm + INIT_STR)
        )
        thetaGT = self._load_ground_truth_results(thetaGT_path)

        # Assert the models information
        self._assert_model_information(theta)

        # Assert the dictionary keys
        self._assert_dictionary_keys(theta)

        # Asserting GT information
        self._assert_ground_truth_information(theta, thetaGT)

    def model_parameter_change_with_config_file(self):
        # Load the configuration file
        with open(PATH_FOR_INIT / ("setting_" + self.algorithm + ".yaml")) as fp:
            self.conf = yaml.safe_load(fp)

        # Change the value of K in the configuration
        self.conf["K"] = K_NEW

        # Change the random seed in the configuration
        self.conf["rng"] = np.random.default_rng(seed=RANDOM_SEED_REPROD)

        # Fit the models to the data using the modified configuration
        self._fit_model_to_data(self.conf)

        # Assert that the models's K parameter matches the value in the configuration
        assert self.model.K == self.conf["K"]


class ConcreteCrossValidation(CrossValidation):
    """
    This is a dummy class that inherits from the abstract class CrossValidation.
    """

    def __init__(
        self,
        algorithm=ALGORITHM,
        model_parameters=MODEL_PARAMETERS,
        cv_parameters=CV_PARAMETERS,
    ):
        super().__init__(algorithm, model_parameters, cv_parameters)

    def calculate_performance_and_prepare_comparison(self):
        # Placeholder implementation
        pass

    def extract_mask(self, fold):
        # Placeholder implementation
        pass

    def load_data(self):
        # Placeholder implementation
        pass

    def prepare_and_run(self):
        # Placeholder implementation
        pass

    def save_results(self):
        # Placeholder implementation
        pass


def check_shape_and_sum(matrix, expected_shape, expected_sum, matrix_name):
    assert (
        matrix.shape == expected_shape
    ), f"Expected {matrix_name} to have shape {expected_shape}, but got {matrix.shape}"
    assert np.isclose(
        np.sum(matrix), expected_sum, atol=TOLERANCE_1
    ), f"Expected sum of {matrix_name} to be {expected_sum}, but got {np.sum(matrix)}"
