"""
This is the test module for the CRep algorithm.
"""
from importlib.resources import files
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from pgm.input.loader import import_data
from pgm.model.crep import CRep
from pgm.output.evaluate import calculate_opt_func, PSloglikelihood

from .fixtures import decimal

GT_OUTPUT_DIR = Path(__file__).parent / 'outputs'
class CRepValidationTestCase(unittest.TestCase):
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

        # Saving the outputs of the tests inside the tests dir
        conf['out_folder'] = self.temp_output_folder / conf['out_folder']

        conf['end_file'] = '_OUT_CRep'  # Adding a suffix to the output files

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
        thetaGT = np.load((GT_OUTPUT_DIR / ('theta_GT_' + self.algorithm)).with_suffix(
            '.npz'))

        self.assertTrue(np.array_equal(self.model.u_f, theta['u']))
        self.assertTrue(np.array_equal(self.model.v_f, theta['v']))
        self.assertTrue(np.array_equal(self.model.w_f, theta['w']))
        self.assertTrue(np.array_equal(self.model.eta_f, theta['eta']))

        self.assertTrue(np.array_equal(thetaGT['u'], theta['u']))
        self.assertTrue(np.array_equal(thetaGT['v'], theta['v']))
        self.assertTrue(np.array_equal(thetaGT['w'], theta['w']))
        self.assertTrue(np.array_equal(thetaGT['eta'], theta['eta']))

        # Remove output npz files after testing using os module
        os.remove((self.model.out_folder / str('theta' + self.model.end_file)).with_suffix('.npz'))

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

        # Remove output npz files after testing using os module
        os.remove((self.model.out_folder / str('theta' + self.model.end_file)).with_suffix('.npz'))

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

        # Remove output npz files after testing using os module
        os.remove((self.model.out_folder / str('theta' + self.model.end_file)).with_suffix('.npz'))
