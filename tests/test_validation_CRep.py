"""
This is the test module for the CRep algorithm.
"""
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml

from pgm.input.loader import import_data
from pgm.model.crep import CRep
from pgm.output.evaluate import calculate_opt_func, PSloglikelihood

from .fixtures import BaseTest, decimal

# pylint: disable=missing-function-docstring, too-many-locals, too-many-instance-attributes


class BaseTestCase(BaseTest):
    """
    The basic class that inherits unittest.TestCase
    """

    def setUp(self) -> None:
        """
        Set up the test case.
        """

        # Test case parameters
        self.algorithm = 'CRep'
        self.adj = 'syn111.dat'
        self.ego = 'source'
        self.alter = 'target'
        self.force_dense = False

        # Import data

        with (files('pgm.data.input').joinpath(self.adj).open('rb') as network):
            self.A, self.B, self.B_T, self.data_T_vals = import_data(
                network.name,
                ego=self.ego,
                alter=self.alter,
                force_dense=self.force_dense,
                binary=False,
                header=0
            )

        self.nodes = self.A[0].nodes()

        # Setting to run the algorithm
        with (files('pgm.data.model').joinpath('setting_' + self.algorithm + '.yaml').open('rb')
              as fp):
            conf = yaml.safe_load(fp)

        # Saving the outputs of the tests into the temp folder created in the BaseTest
        conf['out_folder'] = self.folder

        conf['end_file'] = '_OUT_CRep'  # Adding a suffix to the output files

        self.conf = conf

        self.model = CRep()

    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.temp_output_folder = Path(temp_output_folder)
            # Call the parent class's run method to execute the test
            super().run(result)

    # test case function to check the crep.set_name function
    def test_import_data(self):
        """
        Test import data function
        """

        if self.force_dense:
            self.assertTrue(self.B.sum() > 0)
        else:
            self.assertTrue(self.B.vals.sum() > 0)

    # test case function to check the Person.get_name function
    def test_running_algorithm(self):
        """
        Test running algorithm function.
        """

        _ = self.model.fit(data=self.B,
                           data_T=self.B_T,
                           data_T_vals=self.data_T_vals,
                           nodes=self.nodes,
                           **self.conf)

        theta = np.load((self.temp_output_folder / self.model.out_folder / str('theta' +
                                                                               self.model.end_file)).with_suffix(
            '.npz'))

        # This reads the synthetic data Ground Truth output
        thetaGT_path = Path(__file__).parent /'outputs' / 'theta_GT_CRep'
        thetaGT = np.load(thetaGT_path.with_suffix('.npz'))
        
        # Asserting the model information

        # Assert that the model's u_f attribute is close to the 'u' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.u_f, theta['u']))

        # Assert that the model's v_f attribute is close to the 'v' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.v_f, theta['v']))

        # Assert that the model's w_f attribute is close to the 'w' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.w_f, theta['w']))

        # Assert that the model's eta_f attribute is close to the 'eta' value in the theta dictionary
        self.assertTrue(np.allclose(self.model.eta_f, theta['eta']))

        # Asserting GT information

        # Assert that the 'u' value in the thetaGT dictionary is close to the 'u' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['u'], theta['u']))

        # Assert that the 'v' value in the thetaGT dictionary is close to the 'v' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['v'], theta['v']))

        # Assert that the 'w' value in the thetaGT dictionary is close to the 'w' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['w'], theta['w']))

        # Assert that the 'eta' value in the thetaGT dictionary is close to the 'eta' value in the theta dictionary
        self.assertTrue(np.allclose(thetaGT['eta'], theta['eta']))

    def test_calculate_opt_func(self):
        """
        # Test calculate_opt_func function
        """
        self.force_dense = True

        # Import data

        with (files('pgm.data.input').joinpath(self.adj).open('rb') as network):
            self.A, self.B, self.B_T, self.data_T_vals = import_data(
                network.name,
                ego=self.ego,
                alter=self.alter,
                force_dense=self.force_dense,
                binary=False,
                header=0
            )

        # Running the algorithm

        _ = self.model.fit(data=self.B,
                           data_T=self.B_T,
                           data_T_vals=self.data_T_vals,
                           nodes=self.nodes,
                           **self.conf)

        # Call the function
        opt_func_result = calculate_opt_func(self.B, algo_obj=self.model, assortative=True)

        # Check if the result is a number
        self.assertIsInstance(opt_func_result, float)

        # Check if the result is what expected
        opt_func_expected = -20916.774960752904
        np.testing.assert_almost_equal(opt_func_result, opt_func_expected, decimal=decimal)

    def test_PSloglikelihood(self):
        # Test PSloglikelihood function

        self.force_dense = True

        # Import data

        with (files('pgm.data.input').joinpath(self.adj).open('rb') as network):
            self.A, self.B, self.B_T, self.data_T_vals = import_data(
                network.name,
                ego=self.ego,
                alter=self.alter,
                force_dense=self.force_dense,
                binary=False,
                header=0
            )

        # Running the algorithm

        _ = self.model.fit(
            data=self.B,
            data_T=self.B_T,
            data_T_vals=self.data_T_vals,
            nodes=self.nodes,
            **self.conf)

        # Calculate pseudo log-likelihood
        psloglikelihood_result = PSloglikelihood(
            self.B, self.model.u, self.model.v, self.model.w, self.model.eta)

        # Check that it is what expected
        psloglikelihood_expected = -21975.622428762843

        np.testing.assert_almost_equal(
            psloglikelihood_result,
            psloglikelihood_expected,
            decimal=decimal)

        # Check if psloglikelihood_result is a number
        self.assertIsInstance(psloglikelihood_result, float)

