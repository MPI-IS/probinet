from importlib.resources import files
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from pgm.input.loader import import_data_mtcov
from pgm.model.mtcov import MTCOV

GT_OUTPUT_DIR = Path(__file__).parent / 'outputs'
class MTCovValidationTestCase(unittest.TestCase):

    def setUp(self):
        """
        Set up the test case.
        """
        self.algorithm = 'MTCOV'
        self.C = 2
        self.gamma = 0.5
        self.out_folder = 'outputs/'
        self.end_file = '_test'
        self.adj_name = Path('adj.csv')
        self.cov_name = Path('X.csv')
        self.ego = 'source'
        self.alter = 'target'
        self.egoX = 'Name'

        self.attr_name = 'Metadata'
        self.undirected = False
        self.flag_conv = 'log'
        self.force_dense = False
        self.batch_size = None

        # Import data
        pgm_data_input_path = files('pgm.data.input')
        self.A, self.B, self.X, self.nodes = import_data_mtcov(pgm_data_input_path,
                                                               adj_name=self.adj_name,
                                                               cov_name=self.cov_name,
                                                               ego=self.ego,
                                                               alter=self.alter,
                                                               egoX=self.egoX,
                                                               attr_name=self.attr_name,
                                                               undirected=self.undirected,
                                                               force_dense=self.force_dense)

        self.Xs = np.array(self.X)

        with (files('pgm.data.model').joinpath('setting_' + self.algorithm + '.yaml').open('rb')
              as fp):
            self.conf = yaml.safe_load(fp)

        # Saving the outputs of the tests inside the tests dir
        self.conf['out_folder'] = self.temp_output_folder / self.conf['out_folder']

        self.conf['end_file'] = '_OUT_' + self.algorithm  # Adding a suffix to the output files

        self.model = MTCOV()

    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.temp_output_folder = Path(temp_output_folder)
            # Call the parent class's run method to execute the test
            super().run(result)


    def test_import_data(self):
        print("Start import data test\n")
        if self.force_dense:
            self.assertTrue(self.B.sum() > 0)
            print('B has ', self.B.sum(), ' total weight.')
        else:
            self.assertTrue(self.B.vals.sum() > 0)
            print('B has ', self.B.vals.sum(), ' total weight.')

    def test_running_algorithm(self):
        print("\nStart running algorithm test\n")

        _ = self.model.fit(data=self.B,
                           data_X=self.Xs,
                           flag_conv=self.flag_conv,
                           nodes=self.nodes,
                           batch_size=self.batch_size,
                           **self.conf)

        theta = np.load((self.temp_output_folder / self.model.out_folder / str('theta' +
                                                         self.model.end_file)).with_suffix(
            '.npz'))
        # This reads the synthetic data Ground Truth output
        thetaGT = np.load((GT_OUTPUT_DIR / ('theta_GT_' + self.algorithm)).with_suffix(
            '.npz'))

        self.assertTrue(
            np.allclose(
                self.model.u_f,
                theta['u'],
                rtol=1e-12,
                atol=1e-12))
        self.assertTrue(
            np.allclose(
                self.model.v_f,
                theta['v'],
                rtol=1e-12,
                atol=1e-12))
        self.assertTrue(
            np.allclose(
                self.model.w_f,
                theta['w'],
                rtol=1e-12,
                atol=1e-12))
        self.assertTrue(
            np.allclose(
                self.model.beta_f,
                theta['beta'],
                rtol=1e-12,
                atol=1e-12
            ))

        self.assertTrue(np.allclose(thetaGT['u'], theta['u'], rtol=1e-12, atol=1e-12))
        self.assertTrue(np.allclose(thetaGT['v'], theta['v'], rtol=1e-12, atol=1e-12))
        self.assertTrue(np.allclose(thetaGT['w'], theta['w'], rtol=1e-12, atol=1e-12))
        self.assertTrue(np.allclose(thetaGT['beta'], theta['beta'], rtol=1e-12, atol=1e-12))

