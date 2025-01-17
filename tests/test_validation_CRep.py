"""
This is the test module for the CRep algorithm.
"""

from importlib.resources import files

import numpy as np
import yaml

from probinet.evaluation.likelihood import PSloglikelihood, calculate_opt_func
from probinet.input.loader import build_adjacency_from_file
from probinet.models.crep import CRep

from .constants import DECIMAL, PATH_FOR_INIT, RANDOM_SEED_REPROD
from .fixtures import BaseTest, ModelTestMixin


class BaseTestCase(BaseTest, ModelTestMixin):
    """
    The basic class that inherits unittest.TestCase
    """

    def setUp(self) -> None:
        """
        Set up the test case.
        """

        # Test case parameters
        self.algorithm = "CRep"
        self.keys_in_thetaGT = ["u", "v", "w", "eta", "final_it", "maxPSL", "nodes"]
        self.adj = "syn111.dat"
        self.ego = "source"
        self.alter = "target"
        self.force_dense = False

        # Import data

        with files("probinet.data.input").joinpath(self.adj).open("rb") as network:
            self.gdata = build_adjacency_from_file(
                network.name,
                ego=self.ego,
                alter=self.alter,
                force_dense=self.force_dense,
                binary=False,
                header=0,
            )

        # Setting to run the algorithm
        with open(
            PATH_FOR_INIT / ("setting_" + self.algorithm + ".yaml"), encoding="utf-8"
        ) as fp:
            conf = yaml.safe_load(fp)

        # Saving the outputs of the tests into the temp folder created in the BaseTest
        conf["out_folder"] = self.folder
        conf["end_file"] = (
            "_OUT_" + self.algorithm
        )  # Adding a suffix to the evaluation files
        self.conf = conf
        self.conf["rng"] = np.random.default_rng(seed=RANDOM_SEED_REPROD)
        self.files = PATH_FOR_INIT / "theta_GT_CRep_for_initialization.npz"

        # Run model
        self.model = CRep()

    # test case function to check the crep.set_name function
    def test_import_data(self):
        """
        Test import data function
        """

        if self.force_dense:
            self.assertTrue(self.gdata.adjacency_tensor.sum() > 0)
        else:
            self.assertTrue(self.gdata.adjacency_tensor.data.sum() > 0)

    def test_calculate_opt_func(self):
        """
        # Test calculate_opt_func function in the case where data is dense
        """
        self.force_dense = True

        # Import data
        with files("probinet.data.input").joinpath(self.adj).open("rb") as network:
            self.gdata = build_adjacency_from_file(
                network.name,
                ego=self.ego,
                alter=self.alter,
                force_dense=self.force_dense,
                binary=False,
                header=0,
            )

        # Running the algorithm
        self._fit_model_to_data(self.conf)

        # Call the function
        opt_func_result = calculate_opt_func(
            self.gdata.adjacency_tensor, algo_obj=self.model, assortative=True
        )

        # Check if the result is a number
        self.assertIsInstance(opt_func_result, float)

        # Check if the result is what expected
        opt_func_expected = -21204.389389
        np.testing.assert_almost_equal(
            opt_func_result, opt_func_expected, decimal=DECIMAL
        )

    def test_PSloglikelihood(self):
        """
        Test PSloglikelihood function
        """

        self.force_dense = True

        # Import data
        with files("probinet.data.input").joinpath(self.adj).open("rb") as network:
            self.gdata = build_adjacency_from_file(
                network.name,
                ego=self.ego,
                alter=self.alter,
                force_dense=self.force_dense,
                binary=False,
                header=0,
            )

        # Running the algorithm

        self._fit_model_to_data(self.conf)

        # Calculate pseudo log-likelihood
        psloglikelihood_result = PSloglikelihood(
            self.gdata.adjacency_tensor,
            self.model.u,
            self.model.v,
            self.model.w,
            self.model.eta,
        )

        # Check that it is what expected
        psloglikelihood_expected = -21204.38938

        np.testing.assert_almost_equal(
            psloglikelihood_result, psloglikelihood_expected, decimal=DECIMAL
        )

        # Check if psloglikelihood_result is a number
        self.assertIsInstance(psloglikelihood_result, float)

    def test_running_algorithm_from_mixin(self):
        self.running_algorithm_from_mixin()

    def test_running_algorithm_initialized_from_file_from_mixin(self):
        self.running_algorithm_initialized_from_file_from_mixin()

    def test_model_parameter_change_with_config_file(self):
        self.model_parameter_change_with_config_file()
