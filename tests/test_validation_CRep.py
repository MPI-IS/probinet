"""
This is the test module for the CRep algorithm.
"""

from importlib.resources import files

import numpy as np
import yaml

from pgm.input.loader import import_data
from pgm.model.crep import CRep
from pgm.output.evaluate import calculate_opt_func, PSloglikelihood

from .fixtures import BaseTest, DECIMAL, ModelTestMixin


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

        with files("pgm.data.input").joinpath(self.adj).open("rb") as network:
            self.A, self.B, self.B_T, self.data_T_vals = import_data(
                network.name,
                ego=self.ego,
                alter=self.alter,
                force_dense=self.force_dense,
                binary=False,
                header=0,
            )

        self.nodes = self.A[0].nodes()

        # Setting to run the algorithm
        with (
            files("pgm.data.model")
            .joinpath("setting_" + self.algorithm + ".yaml")
            .open("rb") as fp
        ):
            conf = yaml.safe_load(fp)

        # Saving the outputs of the tests into the temp folder created in the BaseTest
        conf["out_folder"] = self.folder

        conf["end_file"] = (
            "_OUT_" + self.algorithm
        )  # Adding a suffix to the output files

        self.conf = conf

        self.model = CRep() # type: ignore

    # test case function to check the crep.set_name function
    def test_import_data(self):
        """
        Test import data function
        """

        if self.force_dense:
            self.assertTrue(self.B.sum() > 0)
        else:
            self.assertTrue(self.B.vals.sum() > 0)

    def test_calculate_opt_func(self):
        """
        # Test calculate_opt_func function
        """
        self.force_dense = True

        # Import data

        with files("pgm.data.input").joinpath(self.adj).open("rb") as network:
            self.A, self.B, self.B_T, self.data_T_vals = import_data(
                network.name,
                ego=self.ego,
                alter=self.alter,
                force_dense=self.force_dense,
                binary=False,
                header=0,
            )

        # Running the algorithm
        #
        self._fit_model_to_data(self.conf)

        # Call the function
        opt_func_result = calculate_opt_func(
            self.B, algo_obj=self.model, assortative=True
        )

        # Check if the result is a number
        self.assertIsInstance(opt_func_result, float)

        # Check if the result is what expected
        opt_func_expected = -20916.774960752904
        np.testing.assert_almost_equal(
            opt_func_result, opt_func_expected, decimal=DECIMAL
        )

    def test_PSloglikelihood(self):
        # Test PSloglikelihood function

        self.force_dense = True

        # Import data

        with files("pgm.data.input").joinpath(self.adj).open("rb") as network:
            self.A, self.B, self.B_T, self.data_T_vals = import_data(
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
            self.B, self.model.u, self.model.v, self.model.w, self.model.eta
        )

        # Check that it is what expected
        psloglikelihood_expected = -21975.622428762843

        np.testing.assert_almost_equal(
            psloglikelihood_result, psloglikelihood_expected, decimal=DECIMAL
        )

        # Check if psloglikelihood_result is a number
        self.assertIsInstance(psloglikelihood_result, float)

